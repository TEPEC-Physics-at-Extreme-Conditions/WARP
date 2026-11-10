# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Crystal-structure conversion between POSCAR and Quantum ESPRESSO input/output.

Used by :mod:`warpt.tdep_sampling` for one conversion: a sampled supercell
(``Clean_<n>.pos``) plus the seed (the run's master QE input deck) becomes the QE input
for that sample; ``nat``, ``CELL_PARAMETERS`` and ``ATOMIC_POSITIONS`` come from the
POSCAR and every other line of the seed is streamed through. No files are read or
written except by :func:`poscar_to_pwscf` (reads the two inputs, writes ``out_path``
when given) and the command-line front end (stdin to stdout, plus the optional seed
file named by its third argument).

Public API
----------
    read_poscar(text)      -> Structure
    read_pwscf(text)       -> Structure
    read_pwscf_out(text)   -> Structure
    write_poscar(s, banner=None) -> str
    write_pwscf(s, seed_text, banner=None) -> str
    convert(text, fmt_in, fmt_out, seed_text=None, banner=None) -> str
    poscar_to_pwscf(poscar_path, seed_path, out_path=None, banner=None) -> str
    assert_same_order(src, symbols_out, frac_out) -> None

``Structure`` is a named tuple: ``lattice`` (3x3 list, Angstrom, rows are vectors),
``symbols`` (per atom, file order) and ``frac`` (per-atom fractional coordinates,
file order).

Invariant: atoms are never reordered. Every reader keeps file order and every
writer emits it unchanged; the POSCAR writer uses repeated species blocks
(``Mg H O H`` / ``1 1 2 1``, valid VASP5) rather than regrouping by species. WARP
matches the calculator's forces to the sampled configuration by position, so this
is a correctness requirement. ``assert_same_order`` runs on every write.

Output rules of ``write_pwscf``:
  * fractional coordinates are printed with C ``%.17g`` and lattice vectors with
    ``%15.10f``; a negative zero is printed as ``0``;
  * the seed's first line is replaced by a timestamp banner unless it starts with
    ``&`` or contains ``=``, in which case it is kept after the banner;
  * after ``ATOMIC_POSITIONS`` the seed lines are dropped until ``K_POINTS``; a
    ``CELL_PARAMETERS``, ``ibrav`` or ``nat`` line, or (with ``ibrav = 1``) a ``celldm``
    line, met while dropping is still substituted and ends the dropping (after
    ``CELL_PARAMETERS`` only its three vector lines are dropped); with ``ibrav = 0`` a
    ``celldm`` line met while dropping is dropped;
  * the ``ibrav`` line is taken as ``ibrav = 1`` when it contains the character ``1``
    anywhere; with ``ibrav = 1`` the ``celldm(1..3)`` lines are rewritten from the
    lattice and ``CELL_PARAMETERS`` is dropped.
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import List, NamedTuple, Optional, Sequence

from .constants import ANG_TO_BOHR, BOHR_TO_ANG

BOHR = BOHR_TO_ANG   # Angstrom per bohr; ANG_TO_BOHR serves the celldm(1) substitution


class Structure(NamedTuple):
    """A structure in file order: ``lattice`` (3x3, rows are vectors, Å), ``symbols`` and
    ``frac`` (fractional coordinates), one entry per atom."""
    lattice: List[List[float]]   # rows are lattice vectors, Angstrom
    symbols: List[str]           # per atom, file order
    frac: List[List[float]]      # fractional coordinates, file order


# --------------------------------------------------------------------------- helpers
def _inv3(m: Sequence[Sequence[float]]) -> List[List[float]]:
    """Inverse of a 3x3 matrix by cofactors; raises ValueError on a singular lattice."""
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if det == 0:
        raise ValueError("structconv: singular lattice (zero volume)")
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def _cart_to_frac(x: float, y: float, z: float, inv: Sequence[Sequence[float]]) -> List[float]:
    """Fractional coordinates of a Cartesian point (Å), given the inverse lattice ``inv``."""
    # row-vector convention: f_j = sum_i r_i * inv[i][j]
    return [x * inv[0][j] + y * inv[1][j] + z * inv[2][j] for j in range(3)]


def _nz(x: float) -> float:
    """Return ``x`` with a negative zero mapped to +0.0 so it prints as ``0``."""
    return x + 0.0 if x == 0 else x


def _banner(kind: str, when: Optional[_dt.datetime] = None) -> str:
    """Timestamp title line, e.g. ``structconv: PWSCF file converted at 12:34:56 (GMT+2.0) 3rd
    March 2025``; ``when`` defaults to the current local time."""
    t = when or _dt.datetime.now().astimezone()
    off = t.utcoffset() or _dt.timedelta(0)
    sign = "-" if off.total_seconds() < 0 else "+"
    secs = abs(int(off.total_seconds()))
    gmt = f"(GMT{sign}{secs // 3600}.{(secs % 3600) // 60 // 10})"
    d = t.day
    sfx = "th" if 11 <= d % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    return f"structconv: {kind} file converted at {t:%H:%M:%S} {gmt} {d}{sfx} {t:%B} {t:%Y}"


# --------------------------------------------------------------------------- readers
def read_poscar(text: str) -> Structure:
    """Parse VASP5 POSCAR text (the file content, not a path) into a :class:`Structure`.

    Accepts an optional ``Selective dynamics`` line and Direct or Cartesian coordinates
    (converted to fractional). The scale factor multiplies the lattice only; Cartesian
    coordinates are converted with the scaled lattice and are not themselves scaled.
    Returns a :class:`Structure` in file order. Raises ValueError when line 6 holds
    counts instead of species symbols (VASP4 format) or, for Cartesian input, when the
    lattice is singular.
    """
    L = text.splitlines()
    scale = float(L[1].split()[0])
    lat = [[float(v) * scale for v in L[2 + r].split()[:3]] for r in range(3)]
    sp = L[5].split()
    if sp and sp[0].isdigit():
        raise ValueError("structconv: POSCAR must list species symbols on line 6 (VASP5 format)")
    cnt = [int(v) for v in L[6].split()]
    base = 8 if L[7].strip()[:1].upper() == "S" else 7          # 'Selective dynamics' line
    cart = L[base].strip()[:1].upper() in ("C", "K")
    inv = _inv3(lat) if cart else None
    symbols, frac, idx = [], [], base
    for k, n in enumerate(cnt):
        for _ in range(n):
            idx += 1
            x, y, z = (float(v) for v in L[idx].split()[:3])
            frac.append(_cart_to_frac(x, y, z, inv) if cart else [x, y, z])
            symbols.append(sp[k])
    return Structure(lat, symbols, frac)


def _kv(line: str, key: str) -> Optional[str]:
    """Value of ``key=val`` / ``key = val`` on a stripped line, or None."""
    w = line.split()
    if not w:
        return None
    if w[0].startswith(key + "="):
        v = w[0][len(key) + 1:]
        return v if v else (w[1] if len(w) > 1 else None)
    if w[0] == key and len(w) > 2 and w[1] == "=":
        return w[2]
    return None


def read_pwscf(text: str) -> Structure:
    """Parse a Quantum ESPRESSO ``pw.x`` input.

    Supports ``ibrav = 0`` (lattice from ``CELL_PARAMETERS`` in angstrom, bohr or alat
    units) and ``ibrav = 1`` (cubic cell from ``celldm(1)``); atomic positions in
    crystal, angstrom, bohr or alat units are converted to fractional. Returns a
    :class:`Structure` in file order. Raises ValueError for another ``ibrav``, a
    missing ``CELL_PARAMETERS``, ``ATOMIC_POSITIONS`` or ``nat``, or a missing
    ``celldm(1)`` where it is needed.
    """
    L = text.splitlines()
    ibrav, alatb, nat, cprow, aprow, cpu, apu = 0, 0.0, 0, None, None, "", ""
    for r, raw in enumerate(L):
        t = raw.strip()
        for key, cast in (("ibrav", int), ("celldm(1)", float), ("nat", int)):
            v = _kv(t, key)
            if v is not None:
                v = v.rstrip(",")
                if key == "ibrav":
                    ibrav = int(float(v))
                elif key == "celldm(1)":
                    alatb = float(v.replace("d", "e").replace("D", "e"))
                else:
                    nat = int(float(v))
        if t.startswith("CELL_PARAMETERS"):
            cprow, cpu = r, t.lower()
        elif t.startswith("ATOMIC_POSITIONS"):
            aprow, apu = r, t.lower()
    alata = alatb * BOHR
    if ibrav == 0:
        if cprow is None:
            raise ValueError("structconv: ibrav=0 but no CELL_PARAMETERS found")
        f = 1.0 if "angstrom" in cpu else (BOHR if "bohr" in cpu else alata)
        if f == 0:
            raise ValueError("structconv: alat (celldm(1)) required to scale CELL_PARAMETERS")
        lat = [[float(v) * f for v in L[cprow + 1 + r].split()[:3]] for r in range(3)]
    elif ibrav == 1:
        if alata == 0:
            raise ValueError("structconv: ibrav=1 needs celldm(1)")
        lat = [[alata, 0, 0], [0, alata, 0], [0, 0, alata]]
    else:
        raise ValueError(f"structconv: only ibrav=0 and ibrav=1 are supported (got {ibrav})")
    if nat == 0 or aprow is None:
        raise ValueError("structconv: could not locate nat / ATOMIC_POSITIONS")
    cartpos = any(u in apu for u in ("angstrom", "bohr", "alat"))
    posf = BOHR if "bohr" in apu else (alata if "alat" in apu else 1.0)
    inv = _inv3(lat) if cartpos else None
    symbols, frac = [], []
    for k in range(nat):
        p = L[aprow + 1 + k].split()
        x, y, z = (float(v) * posf for v in p[1:4])
        frac.append(_cart_to_frac(x, y, z, inv) if cartpos else [x, y, z])
        symbols.append(p[0])
    return Structure(lat, symbols, frac)


def read_pwscf_out(text: str) -> Structure:
    """Parse the final structure of a ``pw.x`` vc-relax output.

    Reads ``number of atoms/cell`` and the ``CELL_PARAMETERS``/``ATOMIC_POSITIONS``
    blocks between ``Begin final coordinates`` and ``End final coordinates`` (lattice
    in angstrom, bohr or alat units; positions in crystal, angstrom or bohr units,
    converted to fractional; an ``alat`` position block is read as angstrom). Returns
    a :class:`Structure` in file order. Raises ValueError when the atom count, the
    final-coordinates block or its parts are missing; a fixed-cell relax output has no
    final ``CELL_PARAMETERS`` block and raises ValueError.
    """
    L = text.splitlines()
    imin, nat = None, 0
    for r, raw in enumerate(L):
        t = raw.strip()
        if t == "Begin final coordinates":
            imin = r
        w = t.split()
        if nat == 0 and len(w) > 4 and w[2] == "atoms/cell":
            nat = int(w[4])
    if imin is None:
        raise ValueError('structconv: no "Begin final coordinates" block (not a relax/vc-relax output?)')
    if nat == 0:
        raise ValueError("structconv: could not read number of atoms/cell")
    lat, aprow, apu, apend = None, None, "", None
    for r in range(imin, len(L)):
        t = L[r].strip()
        w = t.split()
        if w and w[0] == "CELL_PARAMETERS":
            cpu = t.lower()
            al = 0.0
            if "alat=" in t:
                al = float(t.split("alat=")[1].strip().split()[0].strip(")"))
            f = 1.0 if "angstrom" in cpu else (BOHR if "bohr" in cpu else al * BOHR)
            lat = [[float(v) * f for v in L[r + 1 + k].split()[:3]] for k in range(3)]
        elif w and w[0] == "ATOMIC_POSITIONS":
            aprow, apu = r, t.lower()
        elif t == "End final coordinates":
            apend = r
            break
    if lat is None or aprow is None:
        raise ValueError("structconv: final CELL_PARAMETERS / ATOMIC_POSITIONS not found")
    if apend is None:
        apend = aprow + nat + 1
    cartpos = any(u in apu for u in ("angstrom", "bohr", "alat"))
    posf = BOHR if "bohr" in apu else 1.0
    inv = _inv3(lat) if cartpos else None
    symbols, frac = [], []
    for r in range(aprow + 1, apend):
        p = L[r].split()
        if len(p) < 4:
            continue
        x, y, z = (float(v) * posf for v in p[1:4])
        frac.append(_cart_to_frac(x, y, z, inv) if cartpos else [x, y, z])
        symbols.append(p[0])
    return Structure(lat, symbols, frac)


# --------------------------------------------------------------------------- writers
def assert_same_order(src: Structure, symbols_out: Sequence[str], frac_out: Sequence[Sequence[float]]) -> None:
    """Raise unless the written atoms are the source atoms, one to one, in the same order."""
    if list(symbols_out) != list(src.symbols) or len(frac_out) != len(src.frac):
        raise RuntimeError("structconv: atom order would change on write -- refusing")
    for a, b in zip(frac_out, src.frac):
        if any(abs(x - y) > 1e-12 for x, y in zip(a, b)):
            raise RuntimeError("structconv: atom order would change on write -- refusing")


def write_poscar(s: Structure, banner: Optional[str] = None) -> str:
    """Render ``s`` as VASP5 POSCAR text in file order.

    Consecutive atoms of one species form a block, so an interleaved sequence is
    written with repeated species symbols rather than regrouped. Scale 1.0, Direct
    coordinates, ``%15.10f`` formatting. ``banner`` replaces the timestamp title line.
    Returns the text; raises RuntimeError from :func:`assert_same_order`.
    """
    out = [banner or _banner("POSCAR"), "   %.10f" % 1.0]
    out += ["%15.10f%15.10f%15.10f" % tuple(_nz(x) for x in v) for v in s.lattice]
    runs: List[List[object]] = []                      # [symbol, count] per consecutive run
    for sym in s.symbols:
        if runs and runs[-1][0] == sym:
            runs[-1][1] += 1
        else:
            runs.append([sym, 1])
    out.append(" " + " ".join(str(r[0]) for r in runs))
    out.append(" " + " ".join(str(r[1]) for r in runs))
    out.append(" Direct")
    assert_same_order(s, s.symbols, s.frac)
    out += ["%15.10f%15.10f%15.10f" % tuple(_nz(x) for x in f) for f in s.frac]
    return "\n".join(out) + "\n"


def write_pwscf(s: Structure, seed_text: str, banner: Optional[str] = None) -> str:
    """Render a QE input by streaming ``seed_text`` with the structure of ``s`` substituted.

    Replaces ``nat``, the ``CELL_PARAMETERS`` block (``ibrav = 0``) or the
    ``celldm(1..3)`` lines (``ibrav = 1``) and the ``ATOMIC_POSITIONS`` block (written
    as ``(crystal)`` with ``%.17g``); drops the seed's old coordinate lines up to
    ``K_POINTS``; replaces the first seed line by ``banner`` (default: a timestamp)
    unless it starts with ``&`` or contains ``=``. Returns the text; raises
    RuntimeError from :func:`assert_same_order`.
    """
    lat, na = s.lattice, len(s.symbols)
    out: List[str] = []
    seed = seed_text.splitlines()
    head = seed[0].strip() if seed else ""
    ban = banner or _banner("PWSCF")
    if head.startswith("&") or "=" in head:
        out += [ban, " " + head]          # line 1 is real content: keep it after the banner
    else:
        out.append(ban)                   # line 1 was a title: replaced by the banner
    ibrav_zero, skip = True, 0
    for raw in seed[1:]:
        t = raw.lstrip()
        if t.startswith("ibrav") and t[5:].lstrip().startswith("="):
            ibrav_zero = "1" not in t     # any '1' on the line selects the ibrav=1 branch
            out.append(" " + t); skip = 0; continue
        if t.startswith("nat") and t[3:].lstrip().startswith("="):
            out.append(" nat= %d" % na); skip = 0; continue
        if not ibrav_zero and t.startswith("celldm(1)") and t[9:].lstrip().startswith("="):
            out.append(" celldm(1)= %.8f" % (lat[0][0] * ANG_TO_BOHR)); skip = 0; continue
        if not ibrav_zero and t.startswith("celldm(2)") and t[9:].lstrip().startswith("="):
            out.append(" celldm(2)= %.8f" % (lat[1][1] / lat[0][0])); skip = 0; continue
        if not ibrav_zero and t.startswith("celldm(3)") and t[9:].lstrip().startswith("="):
            out.append(" celldm(3)= %.8f" % (lat[2][2] / lat[0][0])); skip = 0; continue
        if t.startswith("CELL_PARAMETERS"):
            if ibrav_zero:
                out.append(" " + t)
                out += ["%15.10f%15.10f%15.10f" % tuple(_nz(x) for x in v) for v in lat]
            skip = 3                      # drop the three old vectors
            continue
        if t.startswith("ATOMIC_POSITIONS"):
            assert_same_order(s, s.symbols, s.frac)
            out.append(" ATOMIC_POSITIONS (crystal)")
            out += [" %s %.17g %.17g %.17g" % (sym, _nz(f[0]), _nz(f[1]), _nz(f[2])) for sym, f in zip(s.symbols, s.frac)]
            skip = 100000                 # drop all old coordinate lines until K_POINTS
            continue
        if t.startswith("K_POINTS"):
            out.append(" " + t); skip = 0; continue
        if skip > 0:
            skip -= 1; continue
        out.append(" " + t)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- front ends
_READERS = {"poscar": read_poscar, "pwscf": read_pwscf, "pwscf_out": read_pwscf_out}


def convert(text: str, fmt_in: str, fmt_out: str, seed_text: Optional[str] = None,
            banner: Optional[str] = None) -> str:
    """Convert structure text between formats.

    ``fmt_in`` is ``poscar``, ``pwscf`` or ``pwscf_out``; ``fmt_out`` is ``poscar`` or
    ``pwscf`` (the latter needs ``seed_text``). ``banner`` replaces the timestamp line.
    Returns the converted text; raises ValueError for an unsupported format or a missing
    seed.
    """
    if fmt_in not in _READERS:
        raise ValueError(f"structconv: unsupported input format: '{fmt_in}'")
    s = _READERS[fmt_in](text)
    if fmt_out == "poscar":
        return write_poscar(s, banner)
    if fmt_out == "pwscf":
        if seed_text is None:
            raise ValueError("structconv: output 'pwscf' needs a seed/master QE input")
        return write_pwscf(s, seed_text, banner)
    raise ValueError(f"structconv: unsupported output format: '{fmt_out}'")


def poscar_to_pwscf(poscar_path: str, seed_path: str, out_path: Optional[str] = None,
                    banner: Optional[str] = None) -> str:
    """Convert a POSCAR file plus the seed file at ``seed_path`` into a QE input.

    Reads ``poscar_path`` and ``seed_path``; writes the result to ``out_path`` when
    given; ``banner`` replaces the timestamp line. Returns the QE input text. Raises
    ValueError from :func:`read_poscar`, RuntimeError from :func:`assert_same_order` and
    OSError when a file cannot be opened.
    """
    with open(poscar_path) as f:
        pos = f.read()
    with open(seed_path) as f:
        seed = f.read()
    txt = write_pwscf(read_poscar(pos), seed, banner)
    if out_path:
        with open(out_path, "w") as f:
            f.write(txt)
    return txt


if __name__ == "__main__":       # CLI: structconv.py <in-fmt> <out-fmt> [seed] < input > output
    import sys
    if len(sys.argv) < 3:
        sys.exit("usage: structconv.py <poscar|pwscf|pwscf_out> <poscar|pwscf> [seed] < input > output")
    seed = open(sys.argv[3]).read() if len(sys.argv) > 3 else None
    sys.stdout.write(convert(sys.stdin.read(), sys.argv[1], sys.argv[2], seed))
