#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""
Reference-free, step-by-step tracking of the structure and convergence of a WARP run.

Walks every ``iter_<N>/preconditioning/geom_*``, ``iter_<N>/position_*`` and
``iter_<N>/lattice_`` directory of a run, in execution order, and records per step: cell
parameters, volume per atom, c/a, space group and the tightest symprec at which it holds,
first-order residual forces (``outfile.forceconstant_firstorder``), sample count
(``infile.stat``), minimum phonon frequency (``outfile.dispersion_relations``), the internal
stress assembled from ``outfile.stress_dft/fc2/fc3`` and ``outfile.sigma_ph`` (steps without
these files are reported with ``complete=False`` and left out of the stress panels), the
symmetry-forbidden part of that stress, the step RMSD, and the fractional coordinates of one
atom per symmetry orbit (in the cell of the run) with the Wyckoff site it occupies at each
step. The target pressure is read from ``warpt_run.log``, ``run_warpt.sh``,
``run.sh`` or ``job*`` in the run root. Only directories named exactly ``iter_<N>`` are read, and a step is listed as soon as its ``infile.ucposcar``
exists; :mod:`warpt.get_frames` also reads sidelined ``iter_*`` copies but lists a step only
once it has written an ``outfile.new*ucposcar``, so its frame count can differ from the step
count here.

Writes a CSV (``--csv``, default ``warpt_trajectory.csv``) with every recorded quantity and a
six-panel figure of the cell and the moving atomic coordinates (``--plot``, default
``warpt_trajectory.png``) in the current directory, and prints a summary. Requires
ASE and numpy; spglib and matplotlib are optional (symmetry columns and the figure are
skipped without them). Safe to re-run on a partial run.

Command line::

    python3 -m warpt.trajectory [run_dir] [--csv F] [--plot F] [--no-plot] [--no-show]
        [--n-dofs N]

``run_dir`` is the run root (the directory WARP was launched in, holding ``iter_*``) and
defaults to the current directory; ``--no-plot`` writes the CSV and summary only;
``--no-show`` keeps the figure from being opened in a viewer; ``--n-dofs`` (default 3) is the
number of moving coordinates plotted. The convergence quantities (residual forces,
stress residual, phonon-stress change, symmetry-forbidden stress) are in the CSV.

Public functions: ``read_poscar``, ``step_dirs``, ``symmetry``, ``spacegroup_at``,
``site_symmetry``, ``orbit_sites``, ``fc1``, ``min_freq``, ``voigt``, ``stresses``,
``voigt_to_matrix``, ``cartesian_ops``, ``forbidden_stress``, ``nconfigs``, ``press_ext``,
``harvest``, ``dof_label``, ``pretty_group``, ``group_segments``, ``active_dofs``, ``unwrap``,
``show_file``, ``plot``, ``main``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

# SYMPREC_REF: symprec (Å) of the Wyckoff site labels and the shaded space-group bands
# (site_symmetry).
# SYMPREC_LADDER: symprecs tried by symmetry(), which reports the highest space-group number
# found and the tightest symprec that still yields it.
SYMPREC_REF = 1e-2
SYMPREC_LADDER = (0.3, 0.2, 0.15, 0.1, 0.06, 0.04, 0.02, 0.015, 0.01,
                  0.008, 0.006, 0.004, 0.002, 0.001)


def read_poscar(path):
    """Read the POSCAR file at ``path`` with ASE (format ``vasp``) and return an ``ase.Atoms``.

    Distinct from :func:`warpt.io_poscar.read_poscar` (lattice and raw lines) and
    :func:`warpt.structconv.read_poscar` (text to ``Structure``). A path containing ``@`` is
    opened as a file handle because ASE parses ``name@index`` as slice syntax. Errors from
    ``ase.io.read`` (missing or malformed file) propagate.
    """
    from ase.io import read
    if "@" in str(path):
        with open(path) as fh:
            return read(fh, format="vasp")
    return read(str(path), format="vasp")


def step_dirs(root):
    """Every step directory of the run, in the order WARP visits them.

    Only directories named exactly ``iter_<N>`` are read; sidelined copies such as
    ``iter_6.stale`` are ignored (:func:`warpt.get_frames.stage_dirs` includes them).
    Within an iteration: ``preconditioning/geom_*``, then
    ``position_*``, then ``lattice_``. Returns a list of ``(iter, kind, index, path)`` with
    ``kind`` in ``precond``, ``position``, ``lattice``.
    """
    out = []
    def _n(p):
        """Iteration number of a directory named exactly ``iter_<N>``, else None."""
        m = re.fullmatch(r"iter_(\d+)", os.path.basename(p))
        return None if m is None else int(m.group(1))
    cands = [p for p in glob.glob(os.path.join(root, "iter_*")) if _n(p) is not None]
    for it in sorted(cands, key=_n):
        n = _n(it)
        for g in sorted(glob.glob(os.path.join(it, "preconditioning", "geom_*")),
                        key=lambda p: int(p.rsplit("_", 1)[1])):
            out.append((n, "precond", int(g.rsplit("_", 1)[1]), g))
        for p in sorted(glob.glob(os.path.join(it, "position_*")),
                        key=lambda p: int(p.rsplit("_", 1)[1])):
            out.append((n, "position", int(p.rsplit("_", 1)[1]), p))
        lat = os.path.join(it, "lattice_")
        if os.path.isdir(lat):
            out.append((n, "lattice", 0, lat))
    return out


def symmetry(atoms):
    """Highest space-group number found over SYMPREC_LADDER, and the tightest symprec (Å) at
    which that number is still found.

    The tightest symprec falls as a structure settles onto a higher-symmetry parent, so it
    measures how well the symmetry is satisfied without naming a target group. Returns
    ``(None, None)`` without spglib or when every symprec fails.
    """
    try:
        import spglib
    except ImportError:
        return None, None
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    found = {}
    for s in SYMPREC_LADDER:
        try:
            d = spglib.get_symmetry_dataset(cell, symprec=s)
        except Exception:  # noqa: BLE001
            continue
        if d is not None:
            found[s] = int(d.number)
    if not found:
        return None, None
    best = max(found.values())
    tightest = min(s for s, n in found.items() if n == best)
    return best, tightest


def spacegroup_at(atoms, symprec):
    """Space-group number at one symprec (Å), or None without spglib or on failure."""
    try:
        import spglib
        d = spglib.get_symmetry_dataset(
            (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
            symprec=symprec)
        return None if d is None else int(d.number)
    except Exception:  # noqa: BLE001
        return None


def site_symmetry(atoms):
    """Space group and the Wyckoff site of every atom, at SYMPREC_REF.

    Returns ``(number, symbol, sites)``: the space-group number, its Hermann-Mauguin symbol
    (spglib's, e.g. ``P2_1/m``) and, per atom in file order, the site of its orbit as element,
    multiplicity and Wyckoff letter, e.g. ``H32f``; a site shared by several orbits gets ``-2``,
    ``-3``, ... on its later orbits. The multiplicity is that of the conventional cell (the
    number of the orbit's atoms in spglib's standardised cell), so it is the tabulated one also
    when the run uses a primitive cell or a supercell. ``(None, None, [])`` without spglib or
    when spglib fails.
    """
    try:
        import spglib
    except ImportError:
        return None, None, []
    try:
        ds = spglib.get_symmetry_dataset(
            (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
            symprec=SYMPREC_REF)
    except Exception:  # noqa: BLE001
        return None, None, []
    if ds is None:
        return None, None, []
    letters = list(ds.wyckoffs)
    # crystallographic_orbits (spglib >= 2) groups atoms by the full space group; the older
    # equivalent_atoms uses only the operations of the input cell and can split an orbit of a
    # supercell in two.
    orbits = getattr(ds, "crystallographic_orbits", None)
    equiv = np.asarray(orbits if orbits is not None else ds.equivalent_atoms)
    to_prim = np.asarray(ds.mapping_to_primitive)
    std_to_prim = np.asarray(ds.std_mapping_to_primitive)
    symbols = atoms.get_chemical_symbols()
    name, seen = {}, {}
    for i in sorted(set(int(e) for e in equiv)):
        prim = set(to_prim[equiv == i].tolist())
        mult = int(np.isin(std_to_prim, list(prim)).sum())
        site = f"{symbols[i]}{mult}{letters[i]}"
        seen[site] = seen.get(site, 0) + 1
        name[i] = site if seen[site] == 1 else f"{site}-{seen[site]}"
    return int(ds.number), str(ds.international), [name[int(e)] for e in equiv]


def orbit_sites(atoms):
    """One representative atom per symmetry orbit and the Wyckoff site it occupies.

    Returns a list of ``(index, site)``, ``index`` the 0-based position of the first atom of the
    orbit in ``atoms`` and ``site`` as in :func:`site_symmetry`. Empty without spglib or when
    spglib fails.
    """
    out, seen = [], set()
    for i, site in enumerate(site_symmetry(atoms)[2]):
        if site not in seen:
            seen.add(site)
            out.append((i, site))
    return out


def fc1(path):
    """``(mean, max)`` residual force in eV/Å from ``path/outfile.forceconstant_firstorder``
    (workflow_vc_relax.fc1_stats, run inside ``path``); ``(0.0, 0.0)`` when the file is absent
    or has no data rows (fc1_stats); ``(None, None)`` on any error."""
    try:
        from .workflow_vc_relax import fc1_stats
        cwd = os.getcwd()
        os.chdir(path)
        try:
            return fc1_stats()
        finally:
            os.chdir(cwd)
    except Exception:  # noqa: BLE001
        return None, None


def min_freq(path):
    """Minimum frequency (THz) over all bands in ``path/outfile.dispersion_relations``, or None."""
    f = os.path.join(path, "outfile.dispersion_relations")
    if not os.path.exists(f):
        return None
    try:
        return float(np.loadtxt(f)[:, 1:].min())
    except Exception:  # noqa: BLE001
        return None


def voigt(path, name):
    """First six numbers of ``path/name`` as a Voigt vector, skipping ``#`` lines; None if the
    file is absent, unparsable or holds fewer than six numbers."""
    f = os.path.join(path, name)
    if not os.path.exists(f):
        return None
    try:
        # outfile.stress_* holds one row of six after its # header lines; outfile.sigma_ph
        # holds six rows of one. The first six tokens are taken either way.
        toks = [x for l in open(f) if not l.lstrip().startswith("#")
                for x in l.split()]
        v = np.array([float(x) for x in toks[:6]])
        return v if v.size == 6 else None
    except Exception:  # noqa: BLE001
        return None


def stresses(path):
    """Internal stress of a lattice step in WARP's tension-positive convention.

    Returns ``{sigma0, sigma_ph, sigma_int}`` (Voigt, GPa) or None when any of
    ``outfile.stress_dft/fc2/fc3`` is missing. ``sigma0 = -(dft + fc2 + fc3)`` because those
    files are compression-positive; ``outfile.sigma_ph`` is written by workflow_vc_relax
    already tension-positive and is added without negation. ``sigma_ph`` and ``sigma_int`` are
    None when ``outfile.sigma_ph`` is absent.
    """
    d, f2, f3 = (voigt(path, f"outfile.stress_{k}") for k in ("dft", "fc2", "fc3"))
    if d is None or f2 is None or f3 is None:
        return None
    s0 = -(d + f2 + f3)
    ph = voigt(path, "outfile.sigma_ph")
    if ph is not None and ph.size >= 6:
        # Not negated: outfile.sigma_ph is written tension-positive by workflow_vc_relax.
        ph = ph[:6]
        sint = s0 + ph
    else:
        ph, sint = None, None
    return dict(sigma0=s0, sigma_ph=ph, sigma_int=sint)


def voigt_to_matrix(v):
    """Voigt (xx, yy, zz, yz, xz, xy) -> 3x3."""
    return np.array([[v[0], v[5], v[4]],
                     [v[5], v[1], v[3]],
                     [v[4], v[3], v[2]]], float)


def cartesian_ops(atoms, symprec=1e-5):
    """The crystal's point operations as Cartesian (3, 3) matrices.

    spglib returns rotations acting on FRACTIONAL column vectors. With lattice
    vectors as ROWS of L, a Cartesian row vector transforms as r' = r L^-1 R^T L,
    so the operation acting on column vectors is O = (L^-1 R^T L)^T -- which is
    what a rank-2 tensor needs.

    Parameters
    ----------
    atoms : ase.Atoms
    symprec : float
        spglib tolerance in Å (default 1e-5).

    Returns
    -------
    list of (3, 3) arrays, or None
        None without spglib or when spglib cannot identify the symmetry.
    """
    try:
        import spglib
    except ImportError:
        return None
    ds = spglib.get_symmetry_dataset(
        (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
        symprec=symprec)
    if ds is None:
        return None
    L = np.asarray(atoms.cell[:], float)
    Linv = np.linalg.inv(L)
    rots = ds.rotations if hasattr(ds, "rotations") else ds["rotations"]
    out = []
    for R in rots:
        W = Linv @ np.asarray(R, float).T @ L
        out.append(W.T)
    return out


def forbidden_stress(sigma_voigt, ops):
    """Largest symmetry-forbidden component of a stress tensor, in GPa.

    A stress must satisfy sigma = O sigma O^T for every point operation O, so the group
    average is the projection onto the allowed subspace and ``max|sigma - average|`` is the
    forbidden part. Works for any space group without case analysis.

    ``ops`` must come from a fixed reference structure, not from the step being tested:
    recomputed per step, a drifting cell loses operations, the average is taken over the
    reduced group and the forbidden part shrinks exactly when the structure is losing symmetry.

    Parameters
    ----------
    sigma_voigt : array (6,)
        Stress in Voigt order, GPa.
    ops : list of (3, 3) arrays
        Cartesian point operations from cartesian_ops.

    Returns
    -------
    float or None
        None when ``ops`` is empty or ``sigma_voigt`` is None.
    """
    if not ops or sigma_voigt is None:
        return None
    s = voigt_to_matrix(sigma_voigt)
    sym = np.zeros((3, 3))
    for O in ops:
        sym += O @ s @ O.T
    sym /= len(ops)
    return float(np.abs(s - sym).max())


def nconfigs(path):
    """Number of lines in ``path/infile.stat`` (one per sampled configuration), or None."""
    f = os.path.join(path, "infile.stat")
    try:
        return sum(1 for _ in open(f))
    except Exception:  # noqa: BLE001
        return None


def press_ext(root):
    """Target pressure in GPa, or None.

    Searches, in order, ``warpt_run.log`` (``press_ext=<value>`` or ``press_ext <value>``),
    ``run_warpt.sh``, ``run.sh`` and ``job*`` (``--press_ext
    <value>``) in ``root``; as a last resort the absolute value of the first component of the last
    ``σ_ext: [...]`` line of the stress diagnostic block in ``warpt_run.log``.

    Only the first number is read, so for a three- or six-component ``--press_ext`` the result
    is the xx component. The patterns match unsigned numbers only and the ``σ_ext`` fallback
    takes the absolute value, so the result is always a compressive (positive) pressure; a
    tensile target is not recovered with its sign.
    """
    for pat, rx in ((os.path.join(root, "warpt_run.log"), r"press_ext[= ]+([\d.]+)"),
                    (os.path.join(root, "run_warpt.sh"), r"--press_ext\s+([\d.]+)"),
                    (os.path.join(root, "run.sh"), r"--press_ext\s+([\d.]+)"),
                    (os.path.join(root, "job*"), r"--press_ext\s+([\d.]+)")):
        for f in glob.glob(pat):
            try:
                m = re.search(rx, open(f, errors="ignore").read())
                if m:
                    return float(m.group(1))
            except Exception:  # noqa: BLE001
                pass
    # Last resort: the stress diagnostic block (warpt.phonon_stress) prints the applied
    # external stress as "σ_ext: [...]"; the pattern matches its "_ext:" suffix.
    f = os.path.join(root, "warpt_run.log")
    if os.path.exists(f):
        m = re.findall(r"_ext:\s*\[\s*(-?[\d.]+)", open(f, errors="ignore").read())
        if m:
            return abs(float(m[-1]))
    return None


def harvest(root, pext=None):
    """Collect one row of metrics per step directory under ``root``.

    Returns ``(rows, pext)``: ``rows`` is a list of dicts (the coordinate columns are
    ``w_<atom>_x/y/z`` and ``site_<atom>``, e.g. ``w_H4_z``), ``pext`` the target pressure from
    ``pext`` when given, else from press_ext, or None. Steps without a readable ``infile.ucposcar`` are skipped;
    ``press_resid_GPa`` (``max|sigma_int - sigma_ext|`` against the hydrostatic target
    ``[-pext, -pext, -pext, 0, 0, 0]``) needs ``pext``, and ``dsigma_ph_GPa`` a previous step
    with ``outfile.sigma_ph``.
    """
    rows, prev, first = [], None, None
    # Symmetry anchor for the forbidden-stress check: root/start.vasp, else root/infile.ucposcar,
    # else the first structure seen. It stays fixed, so a drifting cell shows up as growing
    # forbidden stress instead of lowering the bar it is measured against.
    ref_ops = None
    for cand in ("start.vasp", "infile.ucposcar"):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            try:
                ref_ops = cartesian_ops(read_poscar(p))
                break
            except Exception:  # noqa: BLE001
                pass
    for it, kind, idx, path in step_dirs(root):
        uc = os.path.join(path, "infile.ucposcar")
        if not os.path.exists(uc):
            continue
        try:
            at = read_poscar(uc)
        except Exception:  # noqa: BLE001
            continue
        cp = at.cell.cellpar()
        spg, tight = symmetry(at)
        m, mx = fc1(path)
        st = stresses(path)
        # Cartesian RMSD (Å) against the previous step from minimum-image fractional differences.
        rmsd = None
        if prev is not None and len(prev[1]) == len(at):
            fa, fb = prev[1].get_scaled_positions(), at.get_scaled_positions()
            dd = fa - fb
            dd -= np.round(dd)
            rmsd = float(np.sqrt((((dd @ np.array(at.cell)) ** 2).sum(1)).mean()))
        r = dict(
            iter=it, kind=kind, index=idx, path=os.path.relpath(path, root),
            natoms=len(at), a=cp[0], b=cp[1], c=cp[2],
            alpha=cp[3], beta=cp[4], gamma=cp[5],
            vol_per_atom=at.get_volume() / len(at), c_over_a=cp[2] / cp[0],
            spacegroup=spg, tightest_symprec=tight,
            fc1_mean_meV_A=None if m is None else m * 1000.0,
            fc1_max_meV_A=None if mx is None else mx * 1000.0,
            n_configs=nconfigs(path), min_freq_THz=min_freq(path),
            step_rmsd_A=rmsd, complete=bool(st and st.get("sigma_int") is not None),
        )
        if ref_ops is None:
            ref_ops = cartesian_ops(at)
        if st and st["sigma_int"] is not None:
            s = st["sigma_int"]
            r["P_int_GPa"] = float(-s[:3].mean())
            r["dev_int_GPa"] = float(s[0] - s[2])
            r["forbidden_GPa"] = forbidden_stress(s, ref_ops)
            r["_sint"], r["_sph"] = s, st["sigma_ph"]
        else:
            r["P_int_GPa"] = r["dev_int_GPa"] = r["forbidden_GPa"] = None
            r["_sint"] = r["_sph"] = None
        r["_frac"] = at.get_scaled_positions(wrap=False)
        r["site_spg"], r["site_group"], r["_sites"] = site_symmetry(at)
        rows.append(r)
        prev = (path, at)
        if first is None:
            first = at

    # Coordinates of one atom per orbit, taken in the cell of the run rather than in a
    # standardised setting: WARP never reorders atoms and only displaces them, so the same atom
    # is followed through every step, also across a change of space group, and no choice of
    # origin or axes can show up as motion. One atom is taken per orbit of the first and of the
    # last step, so orbits that merge when the symmetry rises (or split when it falls) keep a
    # representative each. Columns are named after the atom (element and 1-based position in
    # the file, e.g. H4); site_<atom> records the site that atom occupies at each step.
    reps = sorted({i for i, _ in orbit_sites(first)} | {i for i, _ in orbit_sites(prev[1])}) \
        if prev else []
    symbols = prev[1].get_chemical_symbols() if prev else []
    for r in rows:
        f, here = r.pop("_frac"), r.pop("_sites")
        for i in reps:
            if i < len(f):
                atom = f"{symbols[i]}{i + 1}"
                r[f"w_{atom}_x"], r[f"w_{atom}_y"], r[f"w_{atom}_z"] = (float(v) for v in f[i])
                r[f"site_{atom}"] = here[i] if i < len(here) else None

    # The two convergence quantities WARP tests, evaluated on every step that has
    # outfile.sigma_ph (lattice steps); preconditioning steps have outfile.stress_* but no
    # sigma_ph, position steps have neither.
    if pext is None:
        pext = press_ext(root)
    sext = None if pext is None else np.array([-pext, -pext, -pext, 0.0, 0.0, 0.0])
    prev_ph = None
    for r in rows:
        r["press_resid_GPa"] = r["dsigma_ph_GPa"] = None
        s, ph = r.pop("_sint", None), r.pop("_sph", None)
        if s is not None and sext is not None:
            r["press_resid_GPa"] = float(np.abs(s - sext).max())
        if ph is not None:
            if prev_ph is not None:
                r["dsigma_ph_GPa"] = float(np.abs(ph - prev_ph).max())
            prev_ph = ph
    return rows, pext


def _site_text(site):
    """``H18h`` -> ``("H", "18h")``, ``H6c-2`` -> ``("H", "6c(2)")``; ``(site, "")`` otherwise."""
    m = re.fullmatch(r"([A-Z][a-z]?)(\d+)([A-Za-z])(?:-(\d+))?", site or "")
    if not m:
        return site, ""
    el, mult, letter, n = m.groups()
    return el, f"{mult}{letter}{f'({n})' if n else ''}"


def dof_label(key, rows=()):
    """Legend text of a coordinate column: the atom (element and its number in the file), every
    site it occupied in ``rows`` (its ``site_<atom>`` column, each site once, in the order first
    seen) and the coordinate, e.g. ``w_H1_z`` -> ``H#1 6c → 8c z``."""
    atom, coord = key[2:].rsplit("_", 1)
    m = re.fullmatch(r"([A-Z][a-z]?)(\d+)", atom)
    name = f"{m.group(1)}#{m.group(2)}" if m else atom
    path = []
    for r in rows:
        here = r.get(f"site_{atom}")
        if here and here not in path:
            path.append(here)
    sites = " → ".join(_site_text(h)[1] for h in path)
    return f"{name} {sites} {coord}" if sites else f"{name} {coord}"


def pretty_group(symbol):
    """spglib's Hermann-Mauguin symbol with subscripts, e.g. ``P2_1/m`` -> ``P2₁/m``."""
    return re.sub(r"_(\d)", lambda m: "₀₁₂₃₄₅₆₇₈₉"[int(m.group(1))], symbol or "")


def group_segments(rows):
    """Runs of consecutive steps with the same ``site_spg``, as ``(first, last, number, symbol)``
    step indices; steps without a group are attached to the run before them."""
    out = []
    for i, r in enumerate(rows):
        n = r.get("site_spg")
        if out and (n is None or n == out[-1][2]):
            out[-1][1] = i
        else:
            out.append([i, i, n, r.get("site_group")])
    return [tuple(seg) for seg in out]


def active_dofs(rows, n_max=3, tol=1e-6):
    """The atomic coordinates that move, most active first.

    A coordinate pinned by site symmetry stays constant; a free one varies. Values are
    unwrapped across 0/1 step by step (:func:`unwrap`) before the span is taken.

    Returns
    -------
    keys : list[str]
        Up to ``n_max`` column names (``w_<atom>_<xyz>``) with span above ``tol``, largest
        span first.
    span : dict
        Span (max - min, fractional) of every moving coordinate.
    """
    keys = [k for k in rows[-1] if k.startswith("w_")] if rows else []
    span = {}
    for k in keys:
        v = unwrap([r.get(k) for r in rows])
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        s = float(v.max() - v.min())
        if s > tol:
            span[k] = s
    return sorted(span, key=span.get, reverse=True)[:n_max], span


def unwrap(values):
    """Fractional coordinates as a float array (NaN where missing), with every step-to-step
    jump larger than one half taken across the cell boundary."""
    v = np.array([np.nan if x is None else float(x) for x in values])
    fin = np.flatnonzero(np.isfinite(v))
    for a, b in zip(fin[:-1], fin[1:]):
        v[b] -= np.round(v[b] - v[a])
    return v


# Okabe-Ito qualitative palette: colourblind-safe, and the colours stay distinguishable in
# greyscale because their luminances differ.
BLUE, VERM, GREEN, ORANGE, PURPLE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"


def show_file(path):
    """Open ``path`` in the OS viewer (``open``, ``xdg-open`` or ``start``); failures are
    ignored."""
    import platform, subprocess
    cmd = {"Darwin": ["open"], "Linux": ["xdg-open"], "Windows": ["cmd", "/c", "start", ""]}
    c = cmd.get(platform.system())
    if not c:
        return
    try:
        subprocess.Popen(c + [str(path)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001 - a viewer that fails to start is not an error
        pass


def plot(rows, out, dofs=(), show=False):
    """Draw the six-panel figure of the cell and the free coordinates, and save it to ``out``.

    Left column: the lattice parameters a, b and c (Å), each on its own axis so that small
    changes stay visible. Right column: the cell angles alpha, beta and gamma (degrees) on one
    axis, the volume per atom (Å³), and the ``dofs`` atomic coordinates relative to the first
    step. Every lattice step is marked by a vertical line, and the title gives the space group
    at the first and at the last step. Saved as PNG at 150 dpi; with ``show`` the file is also
    opened via show_file. Skipped with a message when matplotlib is unavailable.

    Parameters
    ----------
    rows : list of dict
        Output of :func:`harvest`.
    out : str
        Path of the PNG to write.
    dofs : sequence of str
        Coordinate columns to plot (``w_<atom>_<x|y|z>``), from :func:`active_dofs`.
    show : bool
        Open the written file in a viewer.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[trajectory] matplotlib not available; skipping the figure")
        return
    x = np.arange(len(rows))
    lat = [i for i, r in enumerate(rows) if r["kind"] == "lattice"]

    def col(k):
        """Column ``k`` of ``rows`` as a float array, NaN where the value is missing."""
        return np.array([np.nan if r.get(k) is None else float(r[k]) for r in rows])

    fig, ax = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
    # Background: one band per run of steps in the same space group (at SYMPREC_REF). Each
    # group has its own tint (the first one none), so mono -> tet -> cubic shows three bands;
    # the groups and their tints are listed under the title.
    segs = group_segments(rows)
    groups = list(dict.fromkeys(num for _, _, num, _ in segs if num is not None))
    tints = [(None, "white"), (BLUE, "blue"), (ORANGE, "orange"), (GREEN, "green"),
             (PURPLE, "purple"), (VERM, "red")]
    tint = {g: tints[i % len(tints)] for i, g in enumerate(groups)}
    for i0, i1, num, _ in segs:
        color = tint.get(num, (None, ""))[0]
        if color:
            for a in ax.ravel():
                a.axvspan(i0 - .5, i1 + .5, color=color, alpha=.08, lw=0, zorder=0)
    spg0, spg1 = rows[0].get("spacegroup"), rows[-1].get("spacegroup")
    spg = "" if not spg0 else (f", space group {spg0}" if spg0 == spg1
                               else f", space group {spg0} → {spg1}")
    head = f"WARP trajectory — {len(rows)} steps, {rows[-1]['iter'] + 1} iterations{spg}"
    if len(segs) > 1:
        runs = [f"{pretty_group(sym)} ({num}) at steps {i0}–{i1} [{tint[num][1]}]"
                for i0, i1, num, sym in segs if num is not None]
        head += f"\nat symprec {SYMPREC_REF:g} Å: " + ", ".join(runs)
    fig.suptitle(head, y=0.995)

    for a, key, color in ((ax[0, 0], "a", BLUE), (ax[1, 0], "b", VERM), (ax[2, 0], "c", GREEN)):
        a.plot(x, col(key), ".-", color=color)
        a.set_ylabel(f"{key} (Å)")
        a.ticklabel_format(axis="y", useOffset=False)
    ax[0, 0].set_title("lattice parameters")

    a = ax[0, 1]
    for key, label, color in (("alpha", "α", BLUE), ("beta", "β", VERM), ("gamma", "γ", GREEN)):
        a.plot(x, col(key), ".-", color=color, label=label)
    a.set_ylabel("angle (°)"); a.set_title("cell angles")
    a.ticklabel_format(axis="y", useOffset=False)
    a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(x, col("vol_per_atom"), ".-", color=PURPLE)
    a.set_ylabel("V/atom (Å$^3$)"); a.set_title("volume per atom")
    a.ticklabel_format(axis="y", useOffset=False)

    a = ax[2, 1]
    if dofs:
        for k, c in zip(dofs, (BLUE, VERM, GREEN, ORANGE, PURPLE)):
            v = unwrap([r.get(k) for r in rows])
            a.plot(x, v - v[np.isfinite(v)][0], ".-", color=c, label=dof_label(k, rows))
        a.axhline(0, color="k", lw=.5, alpha=.4)
        a.set_ylabel("Δ from first step (frac.)")
        a.set_title("atomic coordinates that move")
        # Headroom above the curves keeps the legend off the data.
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.45 * (hi - lo))
        a.legend(fontsize=7, title=f"atom, sites it occupied (symprec {SYMPREC_REF:g} Å)",
                 title_fontsize=7, loc="upper left", ncol=min(3, len(dofs)), columnspacing=1.0,
                 handlelength=1.5)
    else:
        a.text(.5, .5, "no atomic coordinate moves\n(all occupied orbits are\n"
                       "pinned by site symmetry)",
               ha="center", va="center", transform=a.transAxes, fontsize=9, color="0.4")
        a.set_title("atomic coordinates")

    for a in ax[2]:
        a.set_xlabel("step")
    for a in ax.ravel():
        for i in lat:
            a.axvline(i, color="k", lw=.4, alpha=.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[trajectory] wrote {out}")
    if show:
        try:
            plt.show(block=False) if matplotlib.get_backend().lower() != "agg" else None
        except Exception:  # noqa: BLE001
            pass
        show_file(out)


def main():
    """Command-line entry point: harvest ``run_dir`` (the run root), write the CSV, print the
    summary and, unless ``--no-plot``, draw the figure. Exits with a message when no step is
    found."""
    # The description is line 1 of the module docstring, i.e. its summary sentence: this
    # docstring therefore has to keep its blank first line.
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default=".")
    p.add_argument("--csv", default="warpt_trajectory.csv")
    p.add_argument("--plot", default="warpt_trajectory.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-show", action="store_true",
                   help="do not open the figure in a viewer after writing it (default: opened)")
    p.add_argument("--press-ext", "--press_ext", dest="press_ext", type=float, default=None,
                   help="target pressure in GPa for the residual column; default: read from the run "
                        "files (see press_ext)")
    p.add_argument("--n-dofs", type=int, default=3,
                   help="how many of the most active coordinates to plot (default 3)")
    a = p.parse_args()

    rows, pext = harvest(a.run_dir, a.press_ext)
    if not rows:
        sys.exit(f"no WARP steps found under {a.run_dir}")

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[trajectory] {len(rows)} steps -> {a.csv}")

    inc = [r["path"] for r in rows if not r["complete"] and r["kind"] == "lattice"]
    if inc:
        print(f"[trajectory] in-progress lattice step(s), stress left out: {', '.join(inc)}")
    spgs = [r["spacegroup"] for r in rows if r["spacegroup"]]
    if len(set(spgs)) > 1:
        print(f"[trajectory] SPACE GROUP CHANGED during the run: "
              f"{' -> '.join(str(s) for i, s in enumerate(spgs) if i == 0 or s != spgs[i - 1])}")

    lat = [r for r in rows if r["press_resid_GPa"] is not None]
    if lat:
        print(f"[trajectory] target {pext:g} GPa | last 5 lattice steps: "
              + "  ".join(f"{r['press_resid_GPa']:.3f}" for r in lat[-5:])
              + " GPa residual")
        dp = [r["dsigma_ph_GPa"] for r in rows if r["dsigma_ph_GPa"] is not None]
        if dp:
            print("[trajectory] |Δσ_ph| last 5: "
                  + "  ".join(f"{v:.3f}" for v in dp[-5:]) + " GPa")
    last = rows[-1]
    print(f"[trajectory] latest: iter_{last['iter']} {last['kind']}_{last['index']} | "
          f"V/atom {last['vol_per_atom']:.4f} | c/a {last['c_over_a']:.4f} | "
          f"spg {last['spacegroup']} (to symprec {last['tightest_symprec']})")
    dofs, span = active_dofs(rows, a.n_dofs)
    if span:
        print(f"[trajectory] occupied orbits with a moving DoF "
              f"({len(span)} found, tracking {len(dofs)}):")
        for k in dofs:
            print(f"    {dof_label(k, rows):18s} total drift {span[k]:.5f} frac")
    else:
        print("[trajectory] no atomic coordinate moved -- every occupied orbit "
              "is pinned by site symmetry (nothing to track)")
    if not a.no_plot:
        plot(rows, a.plot, dofs, show=not a.no_show)


if __name__ == "__main__":
    main()
