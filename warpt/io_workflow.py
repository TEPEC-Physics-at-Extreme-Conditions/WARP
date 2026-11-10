# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Geometry and force-constant handoff between WARP stages.

Moves structures and force constants from one step directory to the next: picks the best
unit-cell POSCAR of a directory, copies the master QE input deck into the current directory,
builds ``infile.ucposcar`` / ``infile.ssposcar`` from a unit cell with TDEP's
``generate_structure``, and copies ``outfile.forceconstant`` (and
``outfile.forceconstant_thirdorder``) forward as ``infile.*``. ``write_isotopes`` writes the
``infile.isotopes`` that gives TDEP non-natural masses (``--masses``).
``latest_uc`` reads the directory it is given; the other functions write into the current
working directory. A missing input or a failed ``generate_structure`` exits via ``die``.

Public functions: ``latest_uc``, ``copy_master_scf``, ``prepare_infiles_from_uc``,
``copy_previous``, ``write_isotopes``.
"""
import os
import glob
import shutil
from typing import Optional, Sequence, Tuple

from .utilities import die, safe_run
from .io_poscar import normalize_poscar_to_direct_scale1, read_ucposcar_minimal_vasp5


def latest_uc(directory: str = ".") -> str:
    """Path of the best unit-cell POSCAR in ``directory``.

    Priority: ``outfile.new_vc_ucposcar``, ``outfile.new_ucposcar``, ``infile.ucposcar``.
    Exits via ``die`` if none exists.
    """
    for name in ("outfile.new_vc_ucposcar", "outfile.new_ucposcar", "infile.ucposcar"):
        p = os.path.join(directory, name)
        if os.path.exists(p):
            return p
    die("No UC POSCAR found")


def copy_master_scf(master_scf: str) -> str:
    """Copy ``master_scf`` into the current directory unless a file of that basename exists.

    Prints the basename and returns it.
    """
    ms = os.path.basename(master_scf)
    if not os.path.exists(ms):
        shutil.copy(master_scf, ms)
    print(f"  Using master scf file {ms}")
    return ms


def prepare_infiles_from_uc(src_uc: str, sc_opts: Sequence[str]) -> None:
    """Create the TDEP structure infiles in the current directory from a unit-cell POSCAR.

    Copies ``src_uc`` to ``infile.ucposcar``, runs ``generate_structure`` with ``sc_opts`` (the
    flags from :func:`warpt.utilities.parse_supercell`) and copies ``outfile.ssposcar`` to
    ``infile.ssposcar``. Exits via ``die`` if ``generate_structure`` fails or writes no
    ``outfile.ssposcar``.
    """
    shutil.copy(src_uc, "infile.ucposcar")

    rc, out, err = safe_run("generate_structure " + " ".join(sc_opts))
    if rc != 0 or (not os.path.exists("outfile.ssposcar")):
        die("generate_structure failed (no outfile.ssposcar)")

    shutil.copy("outfile.ssposcar", "infile.ssposcar")


def copy_previous(prev_dir: str, sc_opts: Sequence[str]) -> None:
    """Hand geometry and force constants over from ``prev_dir`` to the current directory.

    Rebuilds ``infile.ucposcar`` and ``infile.ssposcar`` from the best unit cell of ``prev_dir``
    (:func:`latest_uc`, then :func:`prepare_infiles_from_uc`). If ``infile.forceconstant`` is
    absent from the current directory, ``prev_dir/outfile.forceconstant`` is copied to it; the
    run exits via ``die`` when that file is missing.
    ``prev_dir/outfile.forceconstant_thirdorder`` is copied to
    ``infile.forceconstant_thirdorder`` when it exists and the target does not.
    """
    src_uc = latest_uc(prev_dir)
    if not os.path.exists(src_uc):
        die(f"Could not find {src_uc}")

    prepare_infiles_from_uc(src_uc, sc_opts)

    if not os.path.exists("infile.forceconstant"):
        pf = os.path.join(prev_dir, "outfile.forceconstant")
        if os.path.exists(pf):
            shutil.copy(pf, "infile.forceconstant")
            print("  Copied previous forceconstant")
        else:
            die("Previous forceconstant missing!")

    pf3 = os.path.join(prev_dir, "outfile.forceconstant_thirdorder")
    if os.path.exists(pf3) and not os.path.exists("infile.forceconstant_thirdorder"):
        shutil.copy(pf3, "infile.forceconstant_thirdorder")
        print("  Copied previous thirdorder forceconstant")


def write_isotopes(masses: dict, poscar: str = "infile.ucposcar", out: str = "infile.isotopes") -> str:
    """Write TDEP's ``infile.isotopes`` for the atoms of ``poscar`` with the masses in ``masses``.

    The file has, for every atom in the order of ``poscar``, a line with the number of isotopes
    (always 1 here) followed by one line ``concentration mass``. Elements absent from
    ``masses`` (``{symbol: mass in amu}``) get their standard atomic weight (ASE). TDEP reads
    the file when ``--readiso`` is passed to ``canonical_configuration`` or
    ``phonon_dispersion_relations``. Returns ``out``.
    """
    from ase.data import atomic_masses, atomic_numbers
    _, _, _, meta = read_ucposcar_minimal_vasp5(poscar)
    lines = []
    for sym, count in zip(meta["species"], meta["counts"]):
        mass = masses.get(sym, float(atomic_masses[atomic_numbers[sym]]))
        lines += ["1", f"1.0 {mass:.6f}"] * count
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out
