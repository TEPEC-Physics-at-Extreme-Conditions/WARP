# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Symmetry projection of TDEP first-order displacements onto Wyckoff site parameters.

Reads a VASP5 unit cell and TDEP's ``outfile.uc_displacements`` (one line per atom:
``index dx dy dz``, fractional), identifies the space group and Wyckoff orbits with spglib,
solves each orbit's free site parameters from the bundled symmetry table
(``warpt/data/symmetry_basics.json`` via :func:`warpt.resources.load_symmetry_basics`), fits
one parameter increment per orbit to the displacements of all its members by least squares,
applies ``p_new = p - damping * dp`` and regenerates every member from the new parameters.
Three guardrails (maximum displacement, minimum interatomic distance and, when
``enforce_input_spacegroup`` is set, unchanged space-group number) run on the staged
positions before an orbit is committed. Writes the projected structure (default
``outfile.uc_projected_structure``) and prints a per-orbit report to stdout.

Public functions: ``get_spglib_data``, ``get_spglib_number_from_structure``, ``load_spacegroup``,
``build_A``, ``extract_wyckoff_data``, ``apply_orbit``, ``extract_rotations``, ``match_orbit``,
``choose_canonical_rep``, ``orbit_reproduces_positions``, ``select_orbit_operator``,
``build_index_map``, ``solve_orbit_parameters``, ``fit_orbit_parameters``,
``fit_orbit_displacement_all_atoms``, ``check_displacement_magnitude``,
``check_displacement_magnitude_staged``, ``check_atoms_too_close``, ``check_spacegroup_preserved``,
``generate_projected_structure``.

Command line::

    python3 -m warpt.Wyckoff_Projector POSCAR [--displacements F] [--out F] [--symprec X]
        [--max-disp-angstrom X] [--min-dist-angstrom X] [--orbit-tol X]
        [--allow-spacegroup-change]

The command line uses the default damping of 0.8 and an ``--orbit-tol`` default of 1e-6
(the function default is 1e-5).
"""
import re
from fractions import Fraction
from collections import defaultdict

import itertools
import numpy as np
import spglib as spg

from .resources import load_symmetry_basics
from .io_poscar import read_ucposcar_minimal_vasp5, write_ucposcar_minimal


# ============================================================
# PBC helpers
# ============================================================

def _mod1(x):
    """Map fractional coordinates into [0, 1)."""
    return np.asarray(x, dtype=float) % 1.0


def _wrap(u):
    """
    Minimum-image convention for a fractional-coordinate *difference*.

    Maps any component into [-0.5, 0.5] (u - round(u); ties round to even). Apply it only to
    *differences* of positions (delta = r_a - r_b), never to positions themselves, because
    positions have a well-defined unit-cell origin.
    """
    u = np.asarray(u, dtype=float)
    return u - np.round(u)


# ============================================================
# spglib interface
# ============================================================

def get_spglib_data(path, symprec=1e-12):
    """
    Read a VASP5 POSCAR and query spglib for its symmetry.

    Parameters
    ----------
    path : str
        VASP5 POSCAR, read with :func:`warpt.io_poscar.read_ucposcar_minimal_vasp5`.
    symprec : float
        spglib tolerance in Å (default 1e-12).

    Returns
    -------
    spg_number : int
        International Tables space-group number.
    wy_letters : list[str]
        Wyckoff letter per atom, in POSCAR order (integer indices converted to letters).
    eq : list[int]
        spglib ``equivalent_atoms``: atoms with the same value form one orbit.
    (lat, pos, types, meta) : tuple
        Lattice, fractional positions, integer types and header data, as returned by
        read_ucposcar_minimal_vasp5.

    Raises
    ------
    RuntimeError
        If spglib cannot identify the symmetry.
    """
    lattice, pos, types, meta = read_ucposcar_minimal_vasp5(path)

    ds = spg.get_symmetry_dataset((lattice, pos, types), symprec=symprec)
    if ds is None:
        raise RuntimeError(f"spglib failed to identify symmetry for {path}")

    spg_number = int(ds.number)
    wy = list(ds.wyckoffs)
    # Integer Wyckoff indices, if spglib returns them, are converted to letters.
    wy_letters = [w if isinstance(w, str) else chr(ord("a") + int(w)) for w in wy]
    eq = list(map(int, ds.equivalent_atoms))

    return spg_number, wy_letters, eq, (lattice, pos, types, meta)


def get_spglib_number_from_structure(lattice, pos, types, symprec=1e-12):
    """
    Space-group number of an in-memory structure.

    Parameters
    ----------
    lattice : (3, 3) array
        Lattice vectors (rows) in Å.
    pos : (N, 3) array
        Fractional positions.
    types : sequence of int
        Integer atom types.
    symprec : float
        spglib tolerance in Å (default 1e-12).

    Returns
    -------
    int
        International Tables space-group number.

    Raises
    ------
    RuntimeError
        If spglib cannot identify the symmetry.
    """
    ds = spg.get_symmetry_dataset((lattice, pos, types), symprec=symprec)
    if ds is None:
        raise RuntimeError("spglib failed to identify symmetry for staged structure")
    return int(ds.number)


# ============================================================
# Symmetry database (JSON)
# ============================================================

def _spacegroup_entries(db):
    """The list of setting records, for either table schema: ``db["data"]["spacegroups"]``
    (JSON-LD schema, the shipped file) or ``db["spacegroups"]`` (flat schema).
    """
    if "data" in db and isinstance(db["data"], dict) and "spacegroups" in db["data"]:
        return db["data"]["spacegroups"]
    return db["spacegroups"]


def load_spacegroup(spg_number, hall_number=None):
    """
    Load the record of ONE setting of a space group from the bundled table.

    The table holds every setting of every group (527 records). The record's operators are
    applied to the structure as given, without a setting transformation, so the structure is
    taken to be in spglib's default setting (origin choice 1 for the 24 two-origin groups;
    unique axis b, cell choice 1 for the monoclinic groups), which is not always the ITA
    reference setting. The record is selected in this order:

      1. if ``hall_number`` (spglib's) is given, the record with this ``it_number`` whose
         ``spglib_hall_numbers`` contains it;
      2. else the record named by the table index
         ``indicies.index_it_number_to_spglib_default_spacegroups`` (JSON-LD schema only),
         provided its ``it_number`` matches;
      3. else the first record with this ``it_number``.

    Per Wyckoff record the JSON-LD schema stores ``orbit`` and ``orbit_mod_centering`` as
    lists of ``{matrix, vector, xyz}``; the flat schema stores ``orbit_affine`` /
    ``orbit_mod_centering_affine`` as 4x4 rational matrices.

    Parameters
    ----------
    spg_number : int
        International Tables space-group number.
    hall_number : int or None
        spglib Hall number of the setting, if known.

    Returns
    -------
    dict
        The setting record.

    Raises
    ------
    KeyError
        If no record has this ``it_number``.
    """
    db = load_symmetry_basics()
    sgs = _spacegroup_entries(db)
    if hall_number is not None:
        for sg in sgs:
            if sg["it_number"] == spg_number and int(hall_number) in [int(h) for h in (sg.get("spglib_hall_numbers") or [])]:
                return sg
    idx = db.get("indicies") or db.get("indices") or {}
    m = idx.get("index_it_number_to_spglib_default_spacegroups") if isinstance(idx, dict) else None
    if m and str(spg_number) in m:
        sg = sgs[int(m[str(spg_number)])]
        if sg["it_number"] == spg_number:
            return sg
    for sg in sgs:
        if sg["it_number"] == spg_number:
            return sg
    raise KeyError(f"Space group {spg_number} not found in symmetry database")


def _wyckoff_records(sg):
    """``{letter: record}`` for both schemas: JSON-LD (list of records with a ``letter`` field,
    the shipped file) or flat (dict keyed by letter).
    """
    w = sg["wyckoff"]
    if isinstance(w, dict):
        return dict(w)
    return {rec["letter"]: rec for rec in w}


def _record_affine(rec, key_new, key_old):
    """(m,4,4) float affine operators of one orbit set, from either schema."""
    if key_old in rec:
        return _orbit_affine_to_float(rec[key_old])
    ops = rec[key_new]
    out = np.zeros((len(ops), 4, 4), dtype=float)
    for k, o in enumerate(ops):
        out[k, :3, :3] = _orbit_affine_to_float(o["matrix"])
        out[k, :3, 3] = _orbit_affine_to_float(o["vector"]).reshape(3)
        out[k, 3, 3] = 1.0
    return out


def _record_first_xyz(rec):
    """Symbolic coordinates of orbit member 0, the site that A and mapping[0] refer to.

    JSON-LD schema: ``orbit[0]["xyz"]``, which need not equal ``first_orbit_ita`` (site d of
    space group 164 lists '2/3,1/3,-z' before '1/3,2/3,z'). Flat schema: ``first_orbit_ita``.
    """
    if "orbit" in rec and rec["orbit"] and isinstance(rec["orbit"][0], dict) and "xyz" in rec["orbit"][0]:
        return rec["orbit"][0]["xyz"]
    return rec["first_orbit_ita"]


# ============================================================
# Parsing the ITA symbolic coordinate string → displacement basis A
# ============================================================

def _orbit_affine_to_float(oa):
    """
    Convert the nested rational arrays from the JSON to float64 numpy arrays.

    The JSON stores exact fractions such as '1/3' or '-1/2' as strings (or as numbers where
    the value is exact). Every entry is converted to float64, strings through
    :class:`fractions.Fraction` so that they are parsed exactly.
    """
    arr = np.array(oa, dtype=object)

    def _to_float(x):
        """Coerce an int/float/Fraction-like orbit-operator entry to ``float``."""
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        return float(Fraction(str(x)))

    return np.vectorize(_to_float, otypes=[float])(arr)


def _parse_linear_token(tok):
    """
    Parse one comma-separated token of an ITA Wyckoff coordinate string.

    Examples of valid tokens: 'x', '-y', '2x', '1/2', 'x+1/4', '-3/2y'.

    Parameters
    ----------
    tok : str
        One coordinate of the string, e.g. 'x+1/4'.

    Returns
    -------
    const : Fraction
        Constant offset (the translational part).
    coeff : dict[str, Fraction]
        Coefficients of x, y, z.
    """
    tok = tok.replace(" ", "")
    parts = re.findall(r"[+-]?[^+-]+", tok)

    const = Fraction(0)
    coeff = {"x": Fraction(0), "y": Fraction(0), "z": Fraction(0)}

    for p in parts:
        if not p:
            continue

        sgn = -1 if p[0] == "-" else 1
        t = p[1:] if p[0] in "+-" else p

        # Variable term: 'x', 'y', 'z', '2x', '1/2x', etc.
        m = re.fullmatch(r'(?:(\d+(?:/\d+)?)?)?([xyz])', t)
        if m:
            a, var = m.groups()
            a = Fraction(a) if a else Fraction(1)
            coeff[var] += sgn * a
        else:
            # Pure rational constant: '1/3', '2', '-1/4', …
            const += sgn * Fraction(t)

    return const, coeff


def build_A(first_orbit_ita):
    """
    Build the (3 × n_dof) displacement basis matrix A for the representative
    Wyckoff site from its ITA symbolic coordinate string.

    A encodes WHICH fractional directions the representative is free to move:
    each column of A is one independent degree of freedom.

    Parameters
    ----------
    first_orbit_ita : str
        Symbolic coordinates of the site, e.g. 'x,y,1/2'.

    Returns
    -------
    A : (3, n_dof) array
        One column per free parameter, ordered by first appearance in the string (x, y, z
        within one coordinate); no columns for a fixed site.

    Examples
    --------
    'x,y,1/2'  →  representative can move in x and y  →  A is 3×2:
        [[1, 0],
         [0, 1],
         [0, 0]]

    '0,0,z'    →  only z is free  →  A is 3×1:
        [[0],
         [0],
         [1]]

    'x,x,0'    →  one DoF, but shared between the first two coords  →  A is 3×1:
        [[1],
         [1],
         [0]]
    """
    toks = [t.strip() for t in first_orbit_ita.split(",")]
    parsed = [_parse_linear_token(t) for t in toks]

    # Collect the variables that appear, in order of first appearance (x, y, z within one
    # coordinate).
    vars_used = []
    for _, c in parsed:
        for v in ("x", "y", "z"):
            if c.get(v, 0) != 0 and v not in vars_used:
                vars_used.append(v)

    A = np.zeros((3, len(vars_used)), dtype=float)
    for i, (_, c) in enumerate(parsed):
        for j, v in enumerate(vars_used):
            A[i, j] = float(c.get(v, 0))

    return A


def extract_wyckoff_data(spg_number, letters=None, hall_number=None):
    """
    For each Wyckoff letter in ``letters`` (None = every letter of the group), extract

      - A   : displacement basis (3 x n_dof) at the orbit[0] site (build_A)
      - oa  : orbit operators (m x 4 x 4) as float64
      - omc : orbit_mod_centering operators (m' x 4 x 4) as float64

    The two operator sets differ only for centred lattices (F, I, C, R), where the
    conventional and primitive cells have different multiplicities;
    generate_projected_structure tries ``oa`` and then ``omc`` against the actual atoms.
    spglib labels the general position of Pmmm (47) 'A'; the table uses the Greek letter
    (U+03B1). Both are accepted.

    Parameters
    ----------
    spg_number : int
        International Tables space-group number.
    letters : iterable of str or None
        Wyckoff letters to extract; None extracts every letter of the selected record.
    hall_number : int or None
        Passed to load_spacegroup.

    Returns
    -------
    dict
        ``{letter: {"A": ..., "oa": ..., "omc": ...}}``.

    Raises
    ------
    KeyError
        If a requested letter is not in the selected setting record.
    """
    sg = load_spacegroup(spg_number, hall_number=hall_number)
    recs = _wyckoff_records(sg)
    alias = {"A": "\u03b1", "\u03b1": "A"}
    out = {}
    for L in (recs.keys() if letters is None else letters):
        rec = recs.get(L)
        if rec is None and L in alias:
            rec = recs.get(alias[L])
        if rec is None:
            raise KeyError(f"Wyckoff letter '{L}' not in the table record of space group {spg_number} "
                           f"(setting {sg.get('setting_it_nc', sg.get('hall', '?'))}); letters there: {sorted(recs)}")
        out[L] = {
            "A":   build_A(_record_first_xyz(rec)),
            "oa":  _record_affine(rec, "orbit", "orbit_affine"),
            "omc": _record_affine(rec, "orbit_mod_centering", "orbit_mod_centering_affine"),
        }
    return out


# ============================================================
# Orbit arithmetic
# ============================================================

def apply_orbit(oa, x):
    """
    Apply all affine orbit operators to a single fractional POSITION x.

    Each operator in `oa` is a (4×4) augmented matrix [R | t; 0 0 0 1].
    The result is mapped back into [0,1)^3 (i.e. the unit cell).

    Parameters
    ----------
    oa : (m, 4, 4) array
        Affine orbit operators in fractional coordinates.
    x : (3,) array
        Fractional position of the input atom.

    Returns
    -------
    positions : (m, 3) array
        Fractional positions in [0, 1), one per operator.

    Raises
    ------
    ValueError
        If ``x`` does not have shape (3,).
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError(f"apply_orbit expects shape (3,), got {x.shape}")
    xh = np.array([x[0], x[1], x[2], 1.0], dtype=float)
    return _mod1((oa @ xh)[:, :3])


def extract_rotations(oa):
    """
    Return the (m, 3, 3) rotation sub-blocks of the affine orbit operators.

    Used by fit_orbit_displacement_all_atoms: a displacement at the orbit[0] site
    transforms to the orbit[k] site with R_k (the linear part of the operator), since
    translations do not act on displacements.
    """
    return np.asarray(oa, dtype=float)[:, :3, :3]


def match_orbit(ref, gen, tol=1e-6):
    """
    Find the bijective permutation that maps generated positions to reference
    atom positions, using the minimum-image convention.

    Parameters
    ----------
    ref : (m, 3) array
        Fractional positions of the m atoms in the orbit (from the POSCAR).
    gen : (m, 3) array
        Fractional positions generated by apply_orbit from some seed atom.
    tol : float
        Maximum allowed fractional-coordinate distance for a match.

    Returns
    -------
    perm : list[int]
        perm[k] = index in ref that matches gen[k].

    Raises
    ------
    RuntimeError
        If any generated point has 0 or more than 1 candidate within tol
        (indicates a symmetry mismatch or tolerance problem).
    """
    rem = set(range(len(ref)))
    perm = []

    for g in gen:
        candidates = [k for k in rem if np.linalg.norm(_wrap(ref[k] - g)) < tol]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Orbit matching failed for generated point {g}: "
                f"{len(candidates)} candidates within tol={tol}. "
                f"Check symprec or orbit_tol."
            )
        j = candidates[0]
        perm.append(j)
        rem.remove(j)

    return perm


def _orbit_match_error(ref, gen, tol=1e-6):
    """Return (max_error, perm) for a ref/gen pair."""
    perm = match_orbit(ref, gen, tol=tol)
    err = max(np.linalg.norm(_wrap(ref[j] - gen[k])) for k, j in enumerate(perm))
    return err, perm


# ============================================================
# Choosing representative atoms and orbit operators
# ============================================================

def choose_canonical_rep(pos, idxs, orbit, tol=1e-5):
    """
    Among the atoms in ``idxs``, find a valid generator: an atom whose image under ``orbit``
    reproduces the full set of orbit positions.

    Preference: the atom fixed by orbit[0]. If no valid generator passes that fixed-point test,
    the valid generator with the smallest 1-based index is returned. An orbit of one atom
    returns that atom without any test. Not used by generate_projected_structure, which
    solves site parameters (solve_orbit_parameters).

    Parameters
    ----------
    pos : (N, 3) array
        All fractional positions in the unit cell (0-based).
    idxs : list[int]
        1-based atom indices belonging to this orbit.
    orbit : (m, 4, 4) array
        Affine orbit operators.
    tol : float
        Fractional-coordinate tolerance for position matching.

    Returns
    -------
    rep : int
        1-based index of the chosen representative atom.

    Raises
    ------
    RuntimeError
        If no atom of the orbit is a valid generator.
    """
    if len(idxs) == 1:
        return idxs[0]

    ref = np.array([pos[i - 1] for i in idxs])
    valid = []

    for cand in idxs:
        gen = apply_orbit(orbit, pos[cand - 1])
        try:
            err, _ = _orbit_match_error(ref, gen, tol=tol)
            if err < tol:
                valid.append(cand)
        except Exception:
            pass

    if not valid:
        raise RuntimeError(
            f"No valid generator found for orbit {idxs}. "
            f"No atom's orbit reproduces the full set within tol={tol}. "
            f"Increase orbit_tol or check symprec."
        )

    # Prefer the atom fixed by orbit[0] (its position maps to itself under operator 0).
    for i in valid:
        coord = pos[i - 1]
        gen = apply_orbit(orbit, coord)
        if np.linalg.norm(_wrap(coord - gen[0])) < tol:
            return i

    # Fallback: the valid generator with the smallest index.
    return min(valid)


def orbit_reproduces_positions(pos, idxs, orbit, tol=1e-6):
    """
    Check whether ``orbit`` reproduces the atom positions in ``idxs``.

    Every atom of the orbit is tried as generator and the one with the smallest matching
    error is kept.

    Parameters
    ----------
    pos : (N, 3) array
        All fractional positions in the unit cell (0-based).
    idxs : list[int]
        1-based atom indices belonging to this orbit.
    orbit : (m, 4, 4) array
        Affine orbit operators.
    tol : float
        Fractional-coordinate tolerance for position matching.

    Returns
    -------
    (ok, best_cand) : tuple[bool, int or None]
        ``ok`` is True only if the best generator reproduces all positions within ``tol``;
        ``best_cand`` is its 1-based index. ``(False, None)`` is returned immediately when the
        operator count ``m`` differs from ``len(idxs)``, and when no atom can be matched at all.
    """
    if orbit.shape[0] != len(idxs):
        return False, None

    ref = np.array([pos[i - 1] for i in idxs])
    best_err, best_cand = np.inf, None

    for cand in idxs:
        gen = apply_orbit(orbit, pos[cand - 1])
        try:
            err, _ = _orbit_match_error(ref, gen, tol=tol)
            if err < best_err:
                best_err, best_cand = err, cand
        except Exception:
            pass

    return (best_err < tol, best_cand)


def select_orbit_operator(pos, idxs, data, tol=1e-6):
    """
    Choose between the ``oa`` and ``omc`` operator sets of a Wyckoff record by testing which
    one reproduces the atom positions (orbit_reproduces_positions).

    For primitive lattices both sets are identical. For centred lattices (F, I, C, R) the
    conventional cell has a higher multiplicity than the primitive cell, so only one of the
    two sets matches the actual atom count.

    Preference: ``oa`` when both work. Not used by generate_projected_structure, which tries
    both sets with solve_orbit_parameters.

    Parameters
    ----------
    pos : (N, 3) array
        All fractional positions in the unit cell (0-based).
    idxs : list[int]
        1-based atom indices belonging to this orbit.
    data : dict
        One entry of extract_wyckoff_data: ``{"A", "oa", "omc"}`` with ``oa`` and ``omc`` of
        shape (m, 4, 4).
    tol : float
        Fractional-coordinate tolerance for position matching.

    Returns
    -------
    (m, 4, 4) array
        The chosen operator set (``data["oa"]`` or ``data["omc"]``).

    Raises
    ------
    RuntimeError
        If neither set reproduces the positions, which indicates a mismatch between the
        structure and the symmetry table.
    """
    candidates = []
    for name in ("oa", "omc"):
        ok, cand = orbit_reproduces_positions(pos, idxs, data[name], tol=tol)
        if ok:
            candidates.append((name, data[name]))

    if not candidates:
        raise RuntimeError(
            f"Neither oa nor omc reproduces atom positions for orbit {idxs}. "
            f"oa mult={data['oa'].shape[0]}, omc mult={data['omc'].shape[0]}, "
            f"actual mult={len(idxs)}"
        )

    # Prefer oa (the conventional-cell set); omc applies only to a centred lattice given in a
    # primitive cell.
    for name, orbit in candidates:
        if name == "oa":
            return orbit
    return candidates[0][1]


def build_index_map(pos, idxs, orbit, rep, tol=1e-6):
    """
    Build the mapping mapping[k] = 1-based index of the atom that orbit operator k sends
    ``rep`` to.

    mapping[0] is the image of ``rep`` under operator 0 (``rep`` itself when operator 0 is the
    identity); mapping[k] for k > 0 are the remaining symmetry-related atoms. Not used by
    generate_projected_structure, which takes the mapping from solve_orbit_parameters.

    Parameters
    ----------
    pos : (N, 3) array
        All fractional positions (0-based indexing).
    idxs : list[int]
        1-based indices of the atoms in this orbit.
    orbit : (m, 4, 4) array
        Affine orbit operators.
    rep : int
        1-based index of any valid generator (from choose_canonical_rep).
        A different rep yields a permuted mapping over the same atoms.
    tol : float
        Matching tolerance.

    Returns
    -------
    mapping : list[int], length m
        mapping[k] = 1-based atom index corresponding to orbit operator k.
    """
    gen0 = apply_orbit(orbit, pos[rep - 1])
    ref  = np.array([pos[i - 1] for i in idxs])
    perm = match_orbit(ref, gen0, tol=tol)
    return [idxs[p] for p in perm]



# ============================================================
# Parameter-based orbit handling
# ============================================================
# The table's orbit operators are PARAMETER maps: member k of a site with free parameters p
# (the x, y, z in its coordinate string) sits at
#     x_k = M_k[:, used] p + v_k        (M_k, v_k from the record's orbit list).
# Member 0 is not always the ITA representative with the identity parameterisation (site d of
# space group 164 lists '2/3,1/3,-z' first, and no member of a site such as 'x,2x,z' has the
# identity parameterisation), so the parameters are solved for explicitly from any
# member, the orbit is generated from them, and the displacement fit works on the parameter
# increment directly (B_k = M_k[:, used]).

def _used_columns(oa):
    """Indices of the free parameters of a site: columns that are non-zero in ANY member."""
    M = np.asarray(oa, dtype=float)[:, :3, :3]
    return [j for j in range(3) if np.any(np.abs(M[:, :, j]) > 1e-12)]


def _orbit_from_parameters(oa, used, p):
    """(m,3) fractional positions of all members for parameter vector p (wrapped to [0,1))."""
    oa = np.asarray(oa, dtype=float)
    M = oa[:, :3, :][:, :, used] if used else np.zeros((oa.shape[0], 3, 0))
    v = oa[:, :3, 3]
    return _mod1(M @ np.asarray(p, dtype=float) + v) if used else _mod1(v)


def solve_orbit_parameters(pos, idxs, oa, tol=1e-6):
    """
    Find the site parameters p and the member->atom mapping for the atoms ``idxs``.

    Tries every atom of the orbit as member 0 (with the 27 integer shifts that make the
    fractional coordinates consistent), solves M_0[:, used] p = x - v_0 in the least-squares
    sense, and accepts when the generated orbit is a bijection onto the atoms (match_orbit
    within ``tol``).

    Parameters
    ----------
    pos : (N, 3) array
        All fractional positions in the unit cell (0-based).
    idxs : list[int]
        1-based atom indices belonging to this orbit.
    oa : (m, 4, 4) array
        Affine orbit operators of the record (parameter maps, see above).
    tol : float
        Fractional-coordinate tolerance, for the least-squares residual and for match_orbit.

    Returns
    -------
    (p, mapping, used) or None
        ``p`` the site parameters, ``mapping[k]`` the 1-based atom index at member k and
        ``used`` the indices of the free parameters; ``None`` when the operator count differs
        from ``len(idxs)`` or no atom/shift reproduces the orbit.
    """
    oa = np.asarray(oa, dtype=float)
    if oa.shape[0] != len(idxs):
        return None
    used = _used_columns(oa)
    M0 = oa[0, :3, :][:, used] if used else np.zeros((3, 0))
    v0 = oa[0, :3, 3]
    ref = np.array([pos[i - 1] for i in idxs], dtype=float)
    shifts = [np.array(n, dtype=float) for n in itertools.product((-1, 0, 1), repeat=3)]
    for a in range(len(idxs)):
        for n in shifts:
            rhs = ref[a] - v0 + n
            if used:
                pv, *_ = np.linalg.lstsq(M0, rhs, rcond=None)
                if np.linalg.norm(M0 @ pv - rhs) > tol:
                    continue
            else:
                pv = np.zeros(0)
                if np.linalg.norm(_wrap(rhs)) > tol:
                    continue
            gen = _orbit_from_parameters(oa, used, pv)
            try:
                perm = match_orbit(ref, gen, tol=tol)
            except Exception:
                continue
            return pv, [idxs[q] for q in perm], used
    return None


def fit_orbit_parameters(disp, mapping, oa, used):
    """
    Least-squares fit of ONE parameter increment dp to the raw displacements of all orbit
    members: M_k[:, used] dp = u_k.

    Parameters
    ----------
    disp : dict[int, (3,) array]
        Fractional displacements by 1-based atom index (from ``outfile.uc_displacements``).
    mapping : list[int]
        mapping[k] = 1-based atom index of orbit member k (from solve_orbit_parameters).
    oa : (m, 4, 4) array
        Affine orbit operators of the record (parameter maps).
    used : list[int]
        Indices of the free parameters (from solve_orbit_parameters).

    Returns
    -------
    dp : (len(used),) array
        Fitted parameter increment; empty when the site has no free parameter.
    U_raw : (m, 3) array
        Raw displacements of the members, fractional.
    U_fit : (m, 3) array
        Fitted displacements M_k[:, used] dp, fractional; zero when the site has no free
        parameter.
    rms : float
        RMS residual of the fit, fractional; the RMS of U_raw when the site has no free
        parameter.

    Raises
    ------
    KeyError
        If an atom of ``mapping`` is absent from ``disp``.
    """
    oa = np.asarray(oa, dtype=float)
    U = np.array([disp[i] for i in mapping], dtype=float)
    if not used:
        return np.zeros(0), U, np.zeros_like(U), float(np.sqrt(np.mean(U ** 2))) if U.size else 0.0
    B = np.concatenate([oa[k, :3, :][:, used] for k in range(oa.shape[0])], axis=0)
    dp, *_ = np.linalg.lstsq(B, U.reshape(-1), rcond=None)
    Ufit = (B @ dp).reshape(U.shape)
    rms = float(np.sqrt(np.mean((U - Ufit) ** 2)))
    return dp, U, Ufit, rms

# ============================================================
# Orbit-wide displacement fit
# ============================================================

def fit_orbit_displacement_all_atoms(disp, mapping, orbit, A):
    """
    Fit one shared Wyckoff-DoF parameter vector p for the entire orbit (position-space fit;
    generate_projected_structure uses fit_orbit_parameters instead).

    Physical model
    --------------
    The allowed displacement of atom k in the orbit is:

        u_k  =  R_k  A  p   +  noise

    where:

      - R_k  is the 3×3 rotation that maps the representative site to atom k
             (extracted from orbit[k]). Translations do not affect
             displacements, only rotations do.
      - A    is the (3 × n_dof) basis of allowed displacements at the orbit[0] site,
             built by build_A from the coordinate string of orbit member 0.
      - p    is the (n_dof,) vector of Wyckoff DoF parameters — the unknowns.

    Stacking all m atoms gives the overdetermined system:

        B  p  ≈  U

    with B of shape (3m × n_dof) and U the stacked raw displacements.
    The least-squares solution averages out TDEP numerical noise across all
    orbit members symmetrically, rather than relying on a single atom.

    Note on coordinate frames
    -------------------------
    Everything here is in FRACTIONAL coordinates (both displacements and A).
    The orbit rotation matrices R_k are the 3×3 upper-left blocks of the
    affine operators, which act on fractional coordinates. No Cartesian
    conversion is needed: fractional in, fractional out throughout.

    Parameters
    ----------
    disp : dict[int → (3,)]
        Per-atom fractional displacements from outfile.uc_displacements.
        Atoms absent from the dict are treated as zero displacement.
    mapping : list[int]
        mapping[k] = 1-based atom index for orbit operator k.
        Constructed by build_index_map; mapping[0] is the orbit[0] atom.
    orbit : (m, 4, 4) array
        Full affine orbit operators (fractional coordinates).
    A : (3, n_dof) array
        Wyckoff displacement basis at the orbit[0] site.

    Returns
    -------
    p : (n_dof,) array
        Fitted Wyckoff DoF parameters.
    urep_proj : (3,) array
        Projected displacement at the orbit[0] site: A @ p.
    U_raw : (m, 3) array
        Raw displacements from TDEP, one row per orbit member.
    U_fit : (m, 3) array
        Model displacements R_k @ urep_proj, one row per orbit member.
    rms : float
        RMS residual |U_raw - U_fit| in fractional units.
    """
    m = len(mapping)
    A = np.asarray(A, dtype=float)

    # Fixed site (no free parameter): nothing to fit.
    if A.shape[1] == 0:
        U_raw = np.array(
            [disp.get(i, np.zeros(3, dtype=float)) for i in mapping], dtype=float
        )
        U_fit = np.zeros_like(U_raw)
        rms = float(np.sqrt(np.mean((U_raw - U_fit) ** 2)))
        return np.zeros(0, dtype=float), np.zeros(3, dtype=float), U_raw, U_fit, rms

    # Build the stacked system  B p = U
    R = extract_rotations(orbit)   # shape (m, 3, 3)

    B = np.zeros((3 * m, A.shape[1]), dtype=float)
    U = np.zeros(3 * m, dtype=float)

    for k, atom_idx in enumerate(mapping):
        # Row block for atom k: R_k A is the (3 × n_dof) sensitivity matrix
        B[3*k : 3*k+3, :] = R[k] @ A
        U[3*k : 3*k+3]    = disp.get(atom_idx, np.zeros(3, dtype=float))

    # Overdetermined least-squares: (3m equations, n_dof unknowns)
    p, _, _, _ = np.linalg.lstsq(B, U, rcond=None)

    # Reconstruct the projected displacement at the orbit[0] site
    urep_proj = A @ p

    # Reconstruct model displacements for all orbit members
    U_fit = np.array([R[k] @ urep_proj for k in range(m)], dtype=float)
    U_raw = U.reshape(m, 3)
    rms   = float(np.sqrt(np.mean((U_raw - U_fit) ** 2)))

    return p, urep_proj, U_raw, U_fit, rms


# ============================================================
# Guardrails (sanity checks before committing new positions)
# ============================================================

def check_displacement_magnitude(
    lat, pos_old, pos_new, idxs, mapping, orbit, seed_atom,
    max_disp_angstrom=2.0,
):
    """
    Guardrail 1 (position path): reject unphysically large displacements. Not called by
    generate_projected_structure, which uses check_displacement_magnitude_staged.

    Regenerates the new orbit from ``seed_atom`` (the orbit[0] atom in the new structure) and
    compares it with the old positions using the minimum-image convention. The comparison is
    made in Cartesian coordinates (Å), so the threshold is a physical distance.

    Parameters
    ----------
    lat : (3, 3) array
        Lattice matrix (rows = lattice vectors) in Å.
    pos_old : (N, 3) array
        Original fractional positions.
    pos_new : (N, 3) array
        New fractional positions after projection.
    idxs : list[int]
        1-based atom indices in this orbit.
    mapping : list[int]
        mapping[k] = 1-based atom index for orbit operator k.
    orbit : (m, 4, 4) array
        Affine orbit operators.
    seed_atom : int
        1-based index of the orbit[0] atom in the new structure (= mapping[0]).
    max_disp_angstrom : float
        Abort threshold in Å.

    Raises
    ------
    RuntimeError
        Listing every atom that moved more than the threshold.
    """
    gen_new = apply_orbit(orbit, pos_new[seed_atom - 1])

    violations = []
    for k, i in enumerate(mapping):
        delta_frac = _wrap(gen_new[k] - pos_old[i - 1])
        delta_cart = delta_frac @ lat
        mag = np.linalg.norm(delta_cart)
        if mag > max_disp_angstrom:
            violations.append((i, mag))

    if violations:
        lines = [
            f"[guardrail:displacement] Orbit {idxs} has unphysically large "
            f"displacements (threshold = {max_disp_angstrom:.2f} Å):"
        ]
        for atom_idx, mag in violations:
            lines.append(f"  atom {atom_idx}: |u| = {mag:.4f} Å")
        raise RuntimeError("\n".join(lines))



def check_displacement_magnitude_staged(lat, pos_old, pos_new, idxs, max_disp_angstrom=2.0):
    """Guardrail 1 (parameter path): no atom of the orbit may move more than the threshold.

    Parameters
    ----------
    lat : (3, 3) array
        Lattice matrix (rows = lattice vectors) in Å.
    pos_old, pos_new : (N, 3) array
        Fractional positions before and after the update (0-based).
    idxs : list[int]
        1-based atom indices of the orbit.
    max_disp_angstrom : float
        Abort threshold in Å (minimum-image displacement).

    Raises
    ------
    RuntimeError
        Listing every atom that moved more than the threshold.
    """
    violations = []
    for i in idxs:
        mag = float(np.linalg.norm(_wrap(pos_new[i - 1] - pos_old[i - 1]) @ lat))
        if mag > max_disp_angstrom:
            violations.append((i, mag))
    if violations:
        lines = [f"[guardrail:displacement] Orbit {idxs} has unphysically large displacements (threshold = {max_disp_angstrom:.2f} Å):"]
        lines += [f"  atom {i}: |u| = {m:.4f} Å" for i, m in violations]
        raise RuntimeError("\n".join(lines))

def check_atoms_too_close(lat, pos_new, all_idxs_so_far, new_idxs, min_dist_angstrom=0.3):
    """
    Guardrail 2: reject configurations where atoms have moved too close.

    Checks minimum-image pairwise Cartesian distances between every atom of the current orbit
    and every atom placed so far, including the other atoms of the current orbit.

    Parameters
    ----------
    lat : (3, 3) array
        Lattice matrix (rows = lattice vectors) in Å.
    pos_new : (N, 3) array
        Current fractional positions (0-based).
    all_idxs_so_far : list[int]
        1-based indices of atoms placed in previous orbits.
    new_idxs : list[int]
        1-based indices of atoms in the current orbit.
    min_dist_angstrom : float
        Abort threshold in Å.

    Raises
    ------
    RuntimeError
        Listing every pair closer than ``min_dist_angstrom``.
    """
    check_against = list(all_idxs_so_far) + list(new_idxs)
    violations, seen = [], set()

    for i in new_idxs:
        for j in check_against:
            pair = (min(i, j), max(i, j))
            if pair in seen or i == j:
                continue
            seen.add(pair)

            dist = np.linalg.norm(_wrap(pos_new[i - 1] - pos_new[j - 1]) @ lat)
            if dist < min_dist_angstrom:
                violations.append((i, j, dist))

    if violations:
        lines = [
            f"[guardrail:proximity] Atoms are unphysically close "
            f"(threshold = {min_dist_angstrom:.2f} Å):"
        ]
        for i, j, d in violations:
            lines.append(f"  atoms {i} and {j}: d = {d:.4f} Å")
        raise RuntimeError("\n".join(lines))


def check_spacegroup_preserved(lat, pos_new, types, expected_spg, symprec=1e-12):
    """
    Guardrail 3: the staged structure must keep the expected space-group number.

    Called on the staged positions of each orbit before it is committed, when
    ``enforce_input_spacegroup`` is set.

    Parameters
    ----------
    lat, pos_new, types
        Lattice (Å), fractional positions and integer types of the staged structure.
    expected_spg : int
        Space-group number the staged structure must keep.
    symprec : float
        spglib tolerance in Å.

    Raises
    ------
    RuntimeError
        If spglib reports a different number.
    """
    found = get_spglib_number_from_structure(lat, pos_new, types, symprec=symprec)
    if found != expected_spg:
        raise RuntimeError(
            f"[guardrail:spacegroup] Projection changed space group: "
            f"expected {expected_spg}, got {found}"
        )


# ============================================================
# Main entry point
# ============================================================

def generate_projected_structure(
    refined_ucposcar_path,
    displacements_path="outfile.uc_displacements",
    out_path="outfile.uc_projected_structure",
    symprec=1e-12,
    max_disp_angstrom=2.0,
    min_dist_angstrom=0.3,
    orbit_tol=1e-5,
    enforce_input_spacegroup=True,
    damping=0.8,
):
    """
    Project TDEP first-order displacements onto Wyckoff site parameters and write the updated
    structure.

    Workflow per orbit (atoms grouped by spglib's ``equivalent_atoms``)
    -------------------------------------------------------------------
    1. solve_orbit_parameters with the ``oa`` operators of the spglib Wyckoff letter, then
       ``omc``; if neither reproduces the atoms, every other record of the group is tried and
       the first that does is used (its letter replaces spglib's).
    2. fit_orbit_parameters: one parameter increment dp fitted to the displacements of all
       members (least squares).
    3. p_new = p_site - damping * dp; all members regenerated from p_new.
    4. Guardrails on the staged positions: check_displacement_magnitude_staged,
       check_atoms_too_close (against all orbits committed so far) and, if
       ``enforce_input_spacegroup``, check_spacegroup_preserved.
    5. The orbit is committed.

    Parameters
    ----------
    refined_ucposcar_path : str
        VASP5 unit cell to update (``infile.ucposcar`` in a WARP step).
    displacements_path : str
        TDEP ``outfile.uc_displacements``: one line per atom, ``index dx dy dz``, fractional;
        blank lines, ``#`` lines and lines without exactly four fields are skipped.
    out_path : str
        Where the projected structure is written (VASP5, scale 1.0, Direct).
    symprec : float
        spglib symmetry tolerance in Å (default 1e-12; :func:`warpt.tdep_extract_fc.run_extract_fc`
        passes 1e-5).
    max_disp_angstrom : float
        Abort if any atom moves more than this, in Å (default 2.0).
    min_dist_angstrom : float
        Abort if any atom pair comes closer than this, in Å (default 0.3). In a WARP step
        the committed structure is checked again, with 0.5 Å, by
        :func:`warpt.tdep_extract_fc.assert_structure_physical`.
    orbit_tol : float
        Fractional-coordinate tolerance for orbit matching (default 1e-5).
    enforce_input_spacegroup : bool
        Run guardrail 3 (default True).
    damping : float
        Fraction of the fitted parameter increment applied (default 0.8).

    Returns
    -------
    out_path : str
        The path the projected structure was written to.
    rep_info : list[dict]
        One dict per orbit: eqid, letter, seed_atom (mapping[0]), site_parameters,
        free_parameters, members, mapping, fit_rms_fractional, rep_displacement_projected
        (fitted displacement of member 0) and fit_parameters (dp).

    Side effects
    ------------
    Writes ``out_path``; prints the per-orbit report to stdout.

    Raises
    ------
    RuntimeError
        If spglib cannot identify the symmetry, if an orbit is reproduced by no record of the
        group, or if a guardrail trips (message prefixed ``[guardrail:...]``).
    KeyError
        If a Wyckoff letter is absent from the table record, or if an atom of an orbit has no
        line in ``displacements_path``.
    """
    spg_number, wy, eq, (lat, pos, types, meta) = \
        get_spglib_data(refined_ucposcar_path, symprec)

    print("=== WYCKOFF PROJECTOR INPUT ===")
    print(f"  space group : {spg_number}")
    print(f"  n_atoms     : {len(pos)}")

    # Group atoms by their equivalent_atoms index (= orbit ID).
    groups = defaultdict(list)
    for i, e in enumerate(eq, 1):
        groups[e].append(i)

    print(f"  n_orbits    : {len(groups)}")

    # Wyckoff letter for each orbit (all atoms in one orbit share a letter).
    letters = {eqid: wy[idxs[0] - 1] for eqid, idxs in groups.items()}

    # Load A, oa and omc for each Wyckoff letter that occurs.
    wy_data = extract_wyckoff_data(spg_number, set(letters.values()))

    # Solve each orbit's site parameters and member->atom mapping (parameter-based path).
    # The Wyckoff letter is only a lookup key: if the record spglib's letter points to does
    # not reproduce the atoms (the six two-origin groups where spglib's and the table's
    # labelling differ by a normaliser translation), every other record of the group is tried
    # and the first that reproduces the orbit is used; solve_orbit_parameters returns None at
    # once for a record whose multiplicity differs from the orbit's.
    solved = {}
    all_records = None
    for eqid, idxs in groups.items():
        L = letters[eqid]
        res = None
        for name in ("oa", "omc"):
            res = solve_orbit_parameters(pos, idxs, wy_data[L][name], tol=orbit_tol)
            if res is not None:
                solved[eqid] = (name,) + res
                break
        if res is None:
            if all_records is None:
                all_records = extract_wyckoff_data(spg_number, None)
            for L2, d2 in all_records.items():
                if L2 == L:
                    continue
                for name in ("oa", "omc"):
                    res = solve_orbit_parameters(pos, idxs, d2[name], tol=orbit_tol)
                    if res is not None:
                        break
                if res is not None:
                    print(f"[wyckoff] spglib letter '{L}' does not reproduce orbit {idxs}; table record '{L2}' "
                          f"does (labelling convention differs between spglib and the table) -> using '{L2}'")
                    letters[eqid] = L2
                    wy_data[L2] = d2
                    solved[eqid] = (name,) + res
                    break
        if res is None:
            raise RuntimeError(f"orbit {idxs} (spglib letter '{L}') is reproduced by no Wyckoff record of "
                               f"space group {spg_number} in the table (multiplicity {len(idxs)})")
    reps = {eqid: solved[eqid][2][0] for eqid in groups}
    print(f"  representative atoms: {reps}")

    # Read the TDEP fractional displacements: index → (dx, dy, dz).
    disp = {}
    with open(displacements_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 4:
                continue
            disp[int(parts[0])] = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])], dtype=float
            )

    new_pos     = pos.copy()
    rep_info    = []
    placed_idxs = []   # grows orbit by orbit; used by the proximity guardrail

    print("\n=== ORBIT PROCESSING ===")

    for eqid, idxs in groups.items():
        L      = letters[eqid]
        opname, p_site, mapping, used = solved[eqid]
        orbit  = wy_data[L][opname]
        print(f"\nOrbit {eqid} | Wyckoff {L} | atoms {idxs} | multiplicity {len(idxs)} | free parameters {['xyz'[j] for j in used]}")
        print(f"  site parameters p = {np.round(p_site, 6)} (member 0 = atom {mapping[0]})")

        # Fit ONE parameter increment dp to the raw displacements of all members (LS).
        dp, U_raw, U_fit, rms = fit_orbit_parameters(disp, mapping, orbit, used)
        print("  raw displacements (fractional):")
        for k, atom_idx in enumerate(mapping):
            print(f"    atom {atom_idx:3d}: {U_raw[k]}")
        print("  fitted displacements (fractional):")
        for k, atom_idx in enumerate(mapping):
            print(f"    atom {atom_idx:3d}: {U_fit[k]}")
        urep_proj = U_fit[0] if len(U_fit) else np.zeros(3)
        p = dp
        print(f"  projected displacement at orbit[0] site: {urep_proj}")
        print(f"  RMS fit residual (fractional): {rms:.6e}")

        # Structure update in parameter space. Sign convention: TDEP writes d, then applies
        # r <- r - d.
        seed_atom = mapping[0]
        p_new = np.asarray(p_site, dtype=float) - damping * dp
        gen = _orbit_from_parameters(orbit, used, p_new)
        staged_pos = new_pos.copy()
        for k, atom_idx in enumerate(mapping):
            staged_pos[atom_idx - 1] = gen[k]

        # --- guardrail 1: no atom moves more than max_disp_angstrom ---
        check_displacement_magnitude_staged(lat, pos, staged_pos, idxs, max_disp_angstrom=max_disp_angstrom)
        # --- guardrail 2: no pair comes closer than min_dist_angstrom ---
        check_atoms_too_close(
            lat, staged_pos, placed_idxs, idxs,
            min_dist_angstrom=min_dist_angstrom,
        )

        # --- guardrail 3: space group must be preserved ---
        if enforce_input_spacegroup:
            check_spacegroup_preserved(
                lat, staged_pos, types,
                expected_spg=spg_number,
                symprec=symprec,
            )

        # All checks passed: commit this orbit.
        new_pos = staged_pos
        placed_idxs.extend(idxs)

        rep_info.append({
            "eqid":                      eqid,
            "letter":                    L,
            "seed_atom":                 seed_atom,   # mapping[0], the member-0 atom
            "site_parameters":           np.asarray(p_site, dtype=float),
            "free_parameters":           ["xyz"[j] for j in used],
            "members":                   idxs,
            "mapping":                   mapping,
            "fit_rms_fractional":        rms,
            "rep_displacement_projected": urep_proj,
            "fit_parameters":            p,
        })

    write_ucposcar_minimal(out_path, lat, new_pos, meta)

    print(f"\n=== WYCKOFF PROJECTOR DONE | output: {out_path} ===")
    return out_path, rep_info


# ============================================================
# Command line
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Project TDEP displacements onto the Wyckoff site parameters and write "
                    "the updated structure."
    )
    parser.add_argument("poscar",
                        help="unit-cell POSCAR (VASP5) to update, e.g. infile.ucposcar")
    parser.add_argument("--displacements", default="outfile.uc_displacements",
                        help="TDEP displacement file (default: outfile.uc_displacements)")
    parser.add_argument("--out", default="outfile.uc_projected_structure",
                        help="file the projected structure is written to "
                             "(default: outfile.uc_projected_structure)")
    parser.add_argument("--symprec", type=float, default=1e-12,
                        help="spglib symmetry tolerance in Å (default: 1e-12)")
    parser.add_argument("--max-disp-angstrom", type=float, default=2.0,
                        help="abort if any atom moves more than this distance, in Å "
                             "(default: 2.0)")
    parser.add_argument("--min-dist-angstrom", type=float, default=0.3,
                        help="abort if any pair of atoms ends up closer than this distance, "
                             "in Å (default: 0.3)")
    parser.add_argument("--orbit-tol", type=float, default=1e-6,
                        help="fractional-coordinate tolerance for matching generated orbit "
                             "members to atoms (default: 1e-6)")
    parser.add_argument("--allow-spacegroup-change", action="store_true",
                        help="skip guardrail 3, the check that the output keeps the input "
                             "space-group number (default: enforced)")
    args = parser.parse_args()

    generate_projected_structure(
        refined_ucposcar_path=args.poscar,
        displacements_path=args.displacements,
        out_path=args.out,
        symprec=args.symprec,
        max_disp_angstrom=args.max_disp_angstrom,
        min_dist_angstrom=args.min_dist_angstrom,
        orbit_tol=args.orbit_tol,
        enforce_input_spacegroup=not args.allow_spacegroup_change,
    )
