#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""VASP backend that mimics Quantum ESPRESSO output.

Runs one VASP single point for a structure WARP sampled and prints energy, forces
and stress in the same QE-like block as :mod:`warpt.MLIP_interface`, so sampling,
collection (:mod:`warpt.tdep_collect`), resume and the convergence logic treat VASP
exactly like QE or an MLIP.

For each sample ``<input>`` (a ``pw.x`` input written by WARP) the run uses the
directory ``<input>.vasp/`` next to it:

* ``POSCAR`` from the input, atoms in input order (the collector reads the
  positions from the QE input, so the order must not change; it is checked
  against the OUTCAR after the run);
* ``INCAR`` and ``KPOINTS`` copied from the template directory (``KPOINTS`` may be
  omitted when ``INCAR`` sets ``KSPACING``);
* ``POTCAR`` concatenated from ``<template>/POTCAR_<element>`` in the order of the
  POSCAR's species line.

The VASP command comes from ``--vasp_cmd`` or, when that is absent, from the
``WARPT_VASP_CMD`` environment variable. The orchestrator passes it through the
environment because the runner splits the calculator command on whitespace, which
would break up a quoted ``mpirun ...`` argument. The run counts as successful only
when the OUTCAR is complete ("General timing" line) and the electronic loop
converged ("EDIFF is reached", and no "EDIFF was not reached"). Then the free
energy (TOTEN; the forces are its negative gradient), forces and stress are printed
in QE units and ``JOB DONE.`` follows; any failure prints ``JOB FAILED. <reason>``
instead, on which the runner and the collector abort the workflow. ``WAVECAR``,
``CHG``, ``CHGCAR`` and the other large files are deleted afterwards unless
``--keep_all`` is given.

Run as ``python3 -m warpt.VASP_interface -input <file> --template <dir>``.

Public functions: ``read_incar``, ``check_template``, ``prepare_run``, ``run_vasp``,
``read_outcar``, ``run_once``, ``main``.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.stress import voigt_6_to_full_3x3_stress
from .constants import EV_TO_GPA_A3, RY_TO_EV, RYBOHR3_TO_GPA, RYBOHR_TO_EV_ANG
from .io_qe import print_qe_like, read_qe_input_file

LARGE_FILES = ("WAVECAR", "CHG", "CHGCAR", "PROCAR", "DOSCAR", "EIGENVAL", "vasprun.xml",
               "XDATCAR", "LOCPOT", "ELFCAR", "WAVEDER")
STALE_FILES = ("OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml", "vasp.stdout")


def read_incar(path):
    """Parse an INCAR into ``{TAG: value-string}`` (upper-case tags; comments dropped)."""
    tags = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].split("!", 1)[0]
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                tags[k.strip().upper()] = v.strip()
    return tags


def check_template(template, elements):
    """Check the template directory.

    Requires ``INCAR``; ``KPOINTS`` unless ``INCAR`` sets ``KSPACING``; one
    ``POTCAR_<el>`` per element. Refuses an INCAR that would not give a single-point
    stress: ``NSW`` > 0 or ``ISIF`` < 2. The orchestrator calls it once before any
    sample runs and prints the returned warnings; :func:`run_once` repeats the check
    for every sample.

    Parameters
    ----------
    template : str or Path
        Template directory.
    elements : iterable of str
        Chemical symbols that occur in the structure.

    Returns
    -------
    list of str
        Warnings (e.g. ``LWAVE``/``LCHARG`` not off); empty when none.

    Raises
    ------
    ValueError
        On a missing file or an INCAR that cannot give a single-point stress.
    """
    t = Path(template)
    if not (t / "INCAR").is_file():
        raise ValueError(f"VASP template {t} has no INCAR")
    tags = read_incar(t / "INCAR")
    if "KSPACING" not in tags and not (t / "KPOINTS").is_file():
        raise ValueError(f"VASP template {t} has no KPOINTS and INCAR sets no KSPACING")
    missing = [el for el in sorted(set(elements)) if not (t / f"POTCAR_{el}").is_file()]
    if missing:
        raise ValueError(f"VASP template {t} lacks " + ", ".join(f"POTCAR_{el}" for el in missing))
    try:
        nsw = int(float(tags.get("NSW", "0")))
    except ValueError:
        raise ValueError(f"INCAR NSW = {tags['NSW']!r} is not a number")
    if nsw > 0:
        raise ValueError(f"INCAR has NSW = {nsw}; WARP needs single points (NSW = 0)")
    if "ISIF" in tags and int(float(tags["ISIF"])) < 2:
        raise ValueError(f"INCAR has ISIF = {tags['ISIF']}; the stress needs ISIF >= 2")
    warn = []
    for tag in ("LWAVE", "LCHARG"):
        if tags.get(tag, ".TRUE.").upper().strip(".") not in ("FALSE", "F"):
            warn.append(f"INCAR {tag} is not .FALSE.; the file is written and then deleted per sample")
    return warn


def prepare_run(atoms, rundir, template):
    """Write POSCAR, INCAR, KPOINTS and POTCAR for one sample; clear stale outputs.

    Returns the species groups of the POSCAR's species line (the POTCAR order).
    Raises RuntimeError when the POSCAR, read back, has a different atom order from
    ``atoms``.
    """
    rundir.mkdir(parents=True, exist_ok=True)
    for name in STALE_FILES:
        (rundir / name).unlink(missing_ok=True)
    write(rundir / "POSCAR", atoms, format="vasp", direct=True, sort=False)
    species = (rundir / "POSCAR").read_text().splitlines()[5].split()
    back = read(rundir / "POSCAR", format="vasp")
    if back.get_chemical_symbols() != atoms.get_chemical_symbols():
        raise RuntimeError("POSCAR atom order differs from the WARP input")
    t = Path(template)
    shutil.copy(t / "INCAR", rundir / "INCAR")
    if (t / "KPOINTS").is_file():
        shutil.copy(t / "KPOINTS", rundir / "KPOINTS")
    with open(rundir / "POTCAR", "wb") as out:
        for el in species:
            out.write((t / f"POTCAR_{el}").read_bytes())
    return species


def run_vasp(rundir, vasp_cmd):
    """Run the VASP command in ``rundir``; stdout and stderr go to ``vasp.stdout``.

    Raises RuntimeError on a non-zero exit code.
    """
    with open(rundir / "vasp.stdout", "w") as fh:
        rc = subprocess.run(shlex.split(vasp_cmd), cwd=rundir, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"VASP exited with code {rc} (see {rundir / 'vasp.stdout'})")


def read_outcar(rundir, atoms):
    """Validate the OUTCAR and return (free energy eV, forces eV/Å, stress eV/Å³ ASE sign).

    Raises RuntimeError when the run did not finish, the electronic loop did not
    converge, or the atoms in the OUTCAR differ from the input (count, species,
    order, or positions by more than 1e-3 Å).
    """
    outcar = rundir / "OUTCAR"
    if not outcar.is_file():
        raise RuntimeError(f"no OUTCAR in {rundir}")
    body = outcar.read_text(errors="ignore")
    if "General timing and accounting informations" not in body:
        raise RuntimeError("OUTCAR is incomplete (VASP did not finish)")
    if "EDIFF was not reached" in body or "EDIFF is reached" not in body:
        raise RuntimeError("electronic loop not converged (raise NELM or check the setup)")
    a = read(outcar, format="vasp-out", index=-1)
    if a.get_chemical_symbols() != atoms.get_chemical_symbols():
        raise RuntimeError("OUTCAR atoms differ from the input (count, species or order)")
    d = a.get_scaled_positions(wrap=False) - atoms.get_scaled_positions(wrap=False)
    d -= np.round(d)
    if np.abs(d @ atoms.cell[:]).max() > 1e-3:
        raise RuntimeError("OUTCAR positions differ from the input by more than 1e-3 Å")
    r = a.calc.results
    if "stress" not in r:
        raise RuntimeError("OUTCAR has no stress (ISIF >= 2 needed)")
    return float(r["free_energy"]), np.asarray(r["forces"]), np.asarray(r["stress"])


def run_once(args):
    """One sample: prepare, run, validate, print the QE-like block. Raises on any failure."""
    atoms = read_qe_input_file(args.input)
    check_template(args.template, atoms.get_chemical_symbols())
    vasp_cmd = args.vasp_cmd or os.environ.get("WARPT_VASP_CMD")
    if not vasp_cmd:
        raise RuntimeError("no VASP command: pass --vasp_cmd or set WARPT_VASP_CMD")
    rundir = Path(args.input).resolve().with_name(Path(args.input).name + ".vasp")
    prepare_run(atoms, rundir, args.template)
    try:
        run_vasp(rundir, vasp_cmd)
        e_ev, f_ev_a, s_ase = read_outcar(rundir, atoms)
    finally:
        if not args.keep_all:
            for name in LARGE_FILES:
                (rundir / name).unlink(missing_ok=True)
    # eV -> QE units with warpt.constants, so the collector's conversion back is exact
    # (ase.units uses CODATA 2014, warpt.constants CODATA 2018; relative difference ~8e-9).
    # Stress: ASE tension-positive eV/Å³ -> QE compression-positive Ry/bohr³.
    s3 = voigt_6_to_full_3x3_stress(s_ase) if s_ase.shape == (6,) else s_ase
    print_qe_like(atoms, e_ev / RY_TO_EV, f_ev_a / RYBOHR_TO_EV_ANG,
                  -s3 * EV_TO_GPA_A3 / RYBOHR3_TO_GPA, source="VASP_interface")


def main():
    """Command-line entry point; prints ``JOB DONE.`` or ``JOB FAILED. <reason>`` (exit 1)."""
    p = argparse.ArgumentParser(description="QE-like runner for WARP: one VASP single point per call.")
    # runner.build_cmd appends '-input <file>' to the calculator command.
    p.add_argument("-input", required=True, help="QE input file WARP wrote for this sample.")
    p.add_argument("--template", required=True,
                   help="Directory with INCAR, KPOINTS (unless INCAR sets KSPACING) and one "
                        "POTCAR_<element> per element.")
    p.add_argument("--vasp_cmd", default=None,
                   help="Command that runs VASP in the sample directory, e.g. 'srun vasp_std'. "
                        "Default: the WARPT_VASP_CMD environment variable.")
    p.add_argument("--keep_all", action="store_true",
                   help="Keep WAVECAR, CHGCAR, vasprun.xml and the other large files "
                        "(default: delete them; OUTCAR and OSZICAR are always kept).")
    args = p.parse_args()
    try:
        run_once(args)
    except Exception as exc:
        print(f"[VASP_interface] Error: {exc}", file=sys.stderr, flush=True)
        print(f"JOB FAILED. {exc}", flush=True)
        sys.exit(1)
    print("JOB DONE.", flush=True)


if __name__ == "__main__":
    main()
