# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""POSCAR / VASP5 structure I/O for WARP.

Readers and writers for the unit-cell and supercell structures exchanged between WARP, TDEP
and the calculator, without ASE or pymatgen. Lattices are returned in Å with the POSCAR scale
applied; positions are fractional. The writers emit scale 1.0;
``normalize_poscar_to_direct_scale1`` and ``write_ucposcar_minimal`` also emit Direct
coordinates wrapped to [0, 1), whereas ``write_poscar_with_new_lattice`` copies the
species/coordinate block of the source file verbatim. ``wrap_to_pm_half`` applies the
minimum-image convention to fractional differences.

Public functions: ``read_poscar_frac``, ``normalize_poscar_to_direct_scale1``, ``read_poscar``,
``write_poscar_with_new_lattice``, ``compute_volume``, ``wrap_to_pm_half``,
``read_ucposcar_minimal_vasp5``, ``write_ucposcar_minimal``.
"""
import numpy as np

def read_poscar_frac(filename: str):
    """Read fractional positions and the lattice from a POSCAR-like file.

    Handles Direct and Cartesian coordinates and an optional Selective dynamics line. Cartesian
    coordinates are converted with the scaled lattice; the scale factor is not applied to the
    coordinates themselves. Lines are read by fixed position (no blank-line skipping).

    Parameters
    ----------
    filename : str
        Path to a POSCAR-like file.

    Returns
    -------
    frac : numpy.ndarray, shape (N, 3)
        Fractional positions, not wrapped.
    A : numpy.ndarray, shape (3, 3)
        Lattice vectors (rows) in Å, scale applied.
    lines : list[str]
        The raw file lines.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    scale = float(lines[1])
    A = np.array([
        list(map(float, lines[2].split())),
        list(map(float, lines[3].split())),
        list(map(float, lines[4].split()))
    ]) * scale

    coord_line_idx = 7
    if lines[7].strip().lower().startswith("s"):
        coord_line_idx = 8

    mode = lines[coord_line_idx].strip().lower()
    start = coord_line_idx + 1

    species_counts = list(map(int, lines[6].split()))
    nat = sum(species_counts)

    frac = []
    for i in range(nat):
        parts = lines[start + i].split()
        xyz = np.array(list(map(float, parts[:3])))
        if mode.startswith("c"):      # Cartesian -> fractional
            frac.append(xyz @ np.linalg.inv(A))
        else:                         # Direct
            frac.append(xyz)

    return np.asarray(frac), A, lines


def normalize_poscar_to_direct_scale1(src: str, dst: str):
    """Rewrite a POSCAR with scale 1.0, Direct coordinates and positions wrapped to [0, 1).

    Reads ``src`` with :func:`read_poscar_frac` and writes ``dst`` (may equal ``src``): title,
    ``1.0``, the scaled lattice, the species and count lines copied verbatim, ``Direct``, and
    the positions with 16 decimals. A Selective dynamics line and its flags are dropped.
    """
    frac, A, lines = read_poscar_frac(src)
    frac = (frac % 1.0 + 1.0) % 1.0

    new_lines = []
    new_lines.append(lines[0])
    new_lines.append("1.0\n")
    for vec in A:                      # A already includes the original scale factor
        new_lines.append(f"{vec[0]:.16f} {vec[1]:.16f} {vec[2]:.16f}\n")

    new_lines.extend(lines[5:7])
    new_lines.append("Direct\n")

    nat = sum(map(int, lines[6].split()))
    for i in range(nat):
        new_lines.append(f"{frac[i,0]:.16f} {frac[i,1]:.16f} {frac[i,2]:.16f}\n")

    with open(dst, "w") as f:
        f.writelines(new_lines)


def read_poscar(filename):
    """Read only the lattice (scale already applied) and the raw lines.

    A minimal reader used when a routine needs to rewrite a POSCAR while
    keeping the species/coordinate block verbatim (see
    :func:`write_poscar_with_new_lattice`).

    Parameters
    ----------
    filename : str
        Path to a POSCAR-like file.

    Returns
    -------
    lattice : numpy.ndarray, shape (3, 3)
        Lattice vectors (rows) in Ångström, with the POSCAR scale applied.
    lines : list[str]
        The original file lines, unmodified.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    scale = float(lines[1])
    lattice = np.array(
        [list(map(float, lines[i].split())) for i in range(2, 5)]
    ) * scale
    return lattice, lines


def write_poscar_with_new_lattice(filename: str, original_lines, new_lattice: np.ndarray):
    """Write ``original_lines`` to ``filename`` with the lattice replaced and scale forced to 1.0.

    Line 0 (title) and lines 5 onwards (species, counts, coordinate mode, positions) are copied
    verbatim, so ``new_lattice`` (rows, Å) must already include any scale factor and the
    coordinate block must already be consistent with scale 1.0 (Direct, or Cartesian in Å).
    """
    new_lines = original_lines[:1] + ["1.0\n"]
    for vec in new_lattice:
        new_lines.append(f"{vec[0]:.16f} {vec[1]:.16f} {vec[2]:.16f}\n")
    new_lines.extend(original_lines[5:])

    with open(filename, "w") as f:
        f.writelines(new_lines)


def compute_volume(lattice: np.ndarray) -> float:
    """Cell volume (Å³) as the absolute scalar triple product of the rows.

    Parameters
    ----------
    lattice : numpy.ndarray, shape (3, 3)
        Lattice vectors as rows, in Ångström.

    Returns
    -------
    float
        The cell volume in Å³.
    """
    return float(abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2]))))


def wrap_to_pm_half(delta_frac: np.ndarray) -> np.ndarray:
    """Wrap fractional differences (array of any shape) into [-0.5, 0.5); returns a new array."""
    return (delta_frac + 0.5) % 1.0 - 0.5

def read_ucposcar_minimal_vasp5(path: str):
    """VASP5 POSCAR reader for unit cells.

    Blank lines are skipped and undecodable bytes ignored; only the first three fields of each
    lattice and position line are read. Supports Direct and Cartesian coordinates (Cartesian
    converted with the scaled lattice) and an optional Selective dynamics line.

    Parameters
    ----------
    path : str
        Path to a VASP5 POSCAR / ucposcar file.

    Returns
    -------
    lattice : (3, 3) array
        Lattice vectors (rows) in Å, scale applied.
    frac_pos : (N, 3) array
        Fractional positions wrapped to [0, 1).
    types : (N,) int array
        Species index 1..nspecies in file order.
    meta : dict
        ``comment`` (str), ``species`` (list[str]), ``counts`` (list[int]).
    """
    with open(path, "r", errors="ignore") as f:
        raw = [ln.strip() for ln in f if ln.strip()]

    comment = raw[0]
    scale = float(raw[1])

    lattice = np.array([list(map(float, raw[i].split()[:3])) for i in (2, 3, 4)], float) * scale

    species = raw[5].split()
    counts = list(map(int, raw[6].split()))
    nat = sum(counts)

    idx = 7
    if raw[idx].lower().startswith("s"):
        idx += 1

    mode = raw[idx].lower()
    idx += 1

    pos = np.array([list(map(float, raw[idx + i].split()[:3])) for i in range(nat)], float)

    if mode.startswith("c"):
        pos = pos @ np.linalg.inv(lattice)

    pos = pos % 1.0

    # types: 1..nspecies repeated by counts
    types = np.concatenate([[t + 1] * c for t, c in enumerate(counts)])

    meta = {"comment": comment, "species": species, "counts": counts}
    return lattice, pos, types, meta


def write_ucposcar_minimal(path: str, lattice, frac_pos, meta):
    """Write a VASP5 POSCAR with scale 1.0 and Direct coordinates.

    ``frac_pos`` is wrapped to [0, 1); ``meta`` supplies ``species`` and ``counts`` (required)
    and ``comment`` (default ``"WARP"``). Values are written with 16 decimals.
    """
    lattice = np.asarray(lattice, float)
    frac_pos = (np.asarray(frac_pos, float) % 1.0)

    comment = meta.get("comment", "WARP")
    species = meta["species"]
    counts = meta["counts"]

    with open(path, "w") as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        for v in lattice:
            f.write(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")
        f.write(" ".join(species) + "\n")
        f.write(" ".join(map(str, counts)) + "\n")
        f.write("Direct\n")
        for p in frac_pos:
            f.write(f"{p[0]:.16f} {p[1]:.16f} {p[2]:.16f}\n")