# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Quantum ESPRESSO input/output helpers for WARP.

Provides :func:`read_qe_stress_gpa`, which extracts the ``total stress`` block from a ``pw.x``
output (or from :mod:`warpt.MLIP_interface` / :mod:`warpt.VASP_interface`, which print the same
block) and returns it in GPa. QE prints the stress compression-positive; the sign is preserved
here and handled by the callers.

Also provides the two pieces every non-QE backend shares: :func:`read_qe_input_file` (the
sample structure WARP wrote) and :func:`print_qe_like` (the QE-shaped block the collector
parses). They live here, not in ``MLIP_interface``, so a backend can use them without
importing torch and UPET.

Public functions: ``read_qe_stress_gpa``, ``read_qe_input_file``, ``print_qe_like``.
"""
import numpy as np
import re

from ase.io.espresso import read_espresso_in

from .constants import KBAR_TO_GPA, RYBOHR3_TO_GPA

KBAR_PER_RY_BOHR3 = RYBOHR3_TO_GPA / KBAR_TO_GPA  # kbar per Ry/bohr^3


def read_qe_input_file(filename: str):
    """Read a Quantum ESPRESSO ``pw.x`` input file.

    Parameters
    ----------
    filename : str
        Path to the input file.

    Returns
    -------
    ase.Atoms
        The structure, as returned by ``ase.io.espresso.read_espresso_in``, atoms in
        file order.
    """
    with open(filename, "r") as f:
        return read_espresso_in(f)


def print_qe_like(atoms, energy_ry, forces_ry_bohr, stress_ry_bohr3, source="MLIP_interface"):
    """Print energy, forces and stress in a minimal QE-like format.

    Emits the lines the WARP collector parses (atom count; total energy on a ``!``
    line; per-atom ``force =`` lines; the ``total stress (Ry/bohr**3)`` block, each row
    followed by its kbar values) together with QE-style header lines (title, forces
    header, stress header with ``P=`` in kbar). The caller prints ``JOB DONE.``
    afterwards. No other line may contain ``!``: the collector takes the energy from
    the last line that does.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure.
    energy_ry : float
        Total energy in Ry.
    forces_ry_bohr : numpy.ndarray, shape (N, 3)
        Forces in Ry/bohr (force-on-atom sign).
    stress_ry_bohr3 : numpy.ndarray, shape (3, 3)
        Stress in Ry/bohr³, compression-positive (QE convention).
    source : str
        Backend named on the title line (default ``MLIP_interface``).
    """
    print(f"Fake QE output generated with {source}")
    print(f"number of atoms/cell = {len(atoms)}")
    print()

    print(f"! total energy = {energy_ry:18.8f} Ry")
    print()

    print(" Forces acting on atoms (cartesian axes, Ry/au):\n")
    for i, f in enumerate(forces_ry_bohr, 1):
        print(f" atom {i:4d} type 1 force = {f[0]:12.8f} {f[1]:12.8f} {f[2]:12.8f}")
    print()

    pressure_kbar = float(np.trace(stress_ry_bohr3) / 3.0 * KBAR_PER_RY_BOHR3)
    print(" Computing stress (cart. axes) and pressure\n")
    print(f" total stress (Ry/bohr**3) (kbar) P= {pressure_kbar:10.2f}")
    for i in range(3):
        row_ry = " ".join(f"{stress_ry_bohr3[i, j]:12.8f}" for j in range(3))
        row_kbar = " ".join(f"{(stress_ry_bohr3[i, j] * KBAR_PER_RY_BOHR3):10.2f}" for j in range(3))
        print(" " + row_ry + " " + row_kbar)

def read_qe_stress_gpa(qe_out_path: str, rybohr3_to_gpa: float, kbar_to_gpa: float = 0.1) -> np.ndarray:
    """Return the 3x3 total stress in GPa from a Quantum ESPRESSO-style output file.

    Finds the first line containing ``total stress`` (any whitespace between the words,
    case-insensitive) and reads the next three lines. If the header mentions ``Ry/bohr`` (the QE
    header does) or mentions neither unit, the first three numbers of each row are multiplied
    by ``rybohr3_to_gpa``; if it mentions only ``kbar`` they are multiplied by ``kbar_to_gpa``.

    Parameters
    ----------
    qe_out_path : str
        Output file to parse: a ``pw.x`` output or the block printed by
        :mod:`warpt.MLIP_interface` or :mod:`warpt.VASP_interface`.
    rybohr3_to_gpa : float
        Ry/bohr³ to GPa (``warpt.constants.RYBOHR3_TO_GPA``).
    kbar_to_gpa : float
        kbar to GPa (default 0.1).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Stress in GPa, sign as printed (compression-positive).

    Raises
    ------
    RuntimeError
        If no header is found, if fewer than three lines follow it, or if a row has fewer
        than three fields.
    ValueError
        If one of the first three fields of a row is not a number.
    """
    with open(qe_out_path, "r", errors="ignore") as f:
        lines = f.readlines()

    for i, hdr in enumerate(lines):
        if re.search(r"total\s+stress", hdr, flags=re.IGNORECASE):
            has_ry = "ry/bohr" in hdr.lower()
            has_kbar = "kbar" in hdr.lower()

            rows = lines[i + 1 : i + 4]
            if len(rows) < 3:
                break

            mat = []
            for r in rows:
                vals = r.split()
                if len(vals) < 3:
                    raise RuntimeError(f"Unexpected stress row format: {r.strip()}")

                triplet = list(map(float, vals[:3]))

                if has_ry:
                    gpa_row = [v * rybohr3_to_gpa for v in triplet]
                elif has_kbar and (not has_ry):
                    gpa_row = [v * kbar_to_gpa for v in triplet]
                else:
                    # header names neither unit: Ry/bohr^3
                    gpa_row = [v * rybohr3_to_gpa for v in triplet]

                mat.append(gpa_row)

            return np.array(mat, dtype=float)

    raise RuntimeError("Could not find 'total stress' block in QE output")
