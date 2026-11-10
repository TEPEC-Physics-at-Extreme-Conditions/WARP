# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Collect calculator outputs into TDEP input files.

:func:`collect_stats` waits until every ``<master>.samples<n>.out`` in the current
working directory contains ``JOB DONE.`` and has a stable size (aborting at once on
a ``JOB FAILED.`` marker), then parses energies, forces and stresses into
``infile.positions``, ``infile.forces``, ``infile.stat`` and ``infile.meta`` and
copies the four files to ``calc_samples/``. :func:`combine_samples` appends one
directory's sampled data to another's; it serves the sliding window of
:func:`~warpt.workflow_preconditioning.run_preconditioning` (the previous geometry's
``local_samples/``) and the ``--reuse_threshold`` merging in
:func:`~warpt.workflow_vc_relax.run_vc_relax` (previous step's samples into the
current position step or into the lattice step).

Reads: ``<master>.samples<n>`` (positions) and ``<master>.samples<n>.out`` (energy,
forces, stress). Writes: the four ``infile.*`` files and their copies in
``calc_samples/``.

Units and signs in ``infile.stat``: energy in eV; the six stress columns in GPa as
printed by the calculator (compression-positive), ordered ``xx, yy, zz, xy, yz, xz``
as TDEP's reader expects; the pressure column is ``-tr(S)/3`` of that tensor, so it
is negative under compression, opposite in sign to the ``P=`` the calculator prints.

Public functions: ``collect_stats``, ``combine_samples``.
"""
import os
import re
import time
import shutil
import subprocess
from typing import Optional, Tuple

import numpy as np

from .utilities import die
from .io_qe import read_qe_stress_gpa
from .constants import RY_TO_EV, RYBOHR_TO_EV_ANG, RYBOHR3_TO_GPA
from .timing import stage

def _file_ready(path: str, settle_s: float, last_sizes: dict, stable_since: dict) -> bool:
    """True once ``path`` holds ``JOB DONE.`` and its size is unchanged for ``settle_s`` seconds.

    Aborts via :func:`die` if the file contains ``JOB FAILED.``. ``last_sizes`` and
    ``stable_since`` are per-path dictionaries updated in place.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", errors="ignore") as f:
            body = f.read()
    except Exception:
        return False
    # JOB FAILED. (written by MLIP_interface) means JOB DONE. will never appear;
    # abort with the calculator's own message rather than wait for the timeout.
    if "JOB FAILED." in body:
        reason = next((ln.split("JOB FAILED.", 1)[1].strip()
                       for ln in body.splitlines() if "JOB FAILED." in ln), "")
        die(f"Calculator failed for {path}: {reason or 'see that file'}")
    if "JOB DONE." not in body:
        return False

    size = os.path.getsize(path)
    now = time.time()
    if size != last_sizes[path]:
        last_sizes[path] = size
        stable_since[path] = now
        return False
    return (now - stable_since[path]) >= settle_s


@stage("collect + fit IFCs")
def collect_stats(
    ctx,
    state,
    *,
    # run_jobs_parallel has already waited for every writer process to exit, so a
    # short settle time suffices.
    poll_period: float = 0.5,
    settle_seconds: float = 1.0,
    timeout: Optional[float] = None,
) -> Tuple[int, int]:
    """Parse the calculator outputs of the current working directory into TDEP input files.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Uses ``master_scf`` (input files are ``<basename>.samples<n>``), ``awk_cmd`` and
        ``temp`` (K, written to ``infile.meta``).
    state : context_state.WarpState
        Not read.
    poll_period : float
        Seconds between checks of the output files.
    settle_seconds : float
        Seconds an output's size must stay unchanged after ``JOB DONE.`` appears.
    timeout : float or None
        Seconds to wait for all outputs before aborting; None waits indefinitely.

    Returns
    -------
    (natoms, nsamples) : tuple[int, int]
        Number of atoms per configuration (from the first output) and number of
        samples written to ``infile.stat``.

    Side effects
    ------------
    Writes ``infile.positions`` (fractional), ``infile.forces`` (eV/Å), ``infile.stat``
    (one row per sample: index, 0, energy in eV, 0, 0, 0, pressure in GPa, then stress
    xx, yy, zz, xy, yz, xz in GPa; the stress columns are the calculator's tensor as
    parsed by :func:`warpt.io_qe.read_qe_stress_gpa`, compression-positive, and the
    pressure column is ``-tr(S)/3`` of it, negative under compression and opposite in
    sign to the ``P=`` the calculator prints) and ``infile.meta``, and copies them into
    ``calc_samples/``.

    Raises
    ------
    SystemExit
        Via :func:`die` when no input files exist, an output contains ``JOB FAILED.``,
        the timeout expires, or the atom count, energy, positions or forces cannot be
        parsed consistently.
    RuntimeError
        From :func:`warpt.io_qe.read_qe_stress_gpa` when an output has no ``total
        stress`` block.
    subprocess.CalledProcessError
        When an awk extraction command (energy, positions or forces) exits non-zero.
    """
    master_base = os.path.basename(ctx.master_scf)

    ins = [p for p in os.listdir(".") if re.match(rf"^{re.escape(master_base)}\.samples\d+$", p)]
    if not ins:
        die("No SCF input files (*.samplesN) found; nothing to collect.")

    def _samples_index(name: str) -> int:
        """Trailing integer of a ``*.samplesN`` filename (``-1`` if none)."""
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else -1

    ins.sort(key=_samples_index)
    outs = [f"{t}.out" for t in ins]

    # Barrier: wait until every output contains JOB DONE. and has a stable size.
    last_sizes = {p: -1 for p in outs}
    stable_since = {p: 0.0 for p in outs}

    t0 = time.time()
    while True:
        ok = True
        for o in outs:
            if not _file_ready(o, settle_seconds, last_sizes, stable_since):
                ok = False
                break
        if ok:
            break
        if timeout is not None and (time.time() - t0) > timeout:
            missing = [o for o in outs if not os.path.exists(o)]
            die(f"Timeout waiting for SCF outputs. Missing: {missing}")
        time.sleep(poll_period)

    # Number of atoms, from the first output.
    cmdN = (
        f"LC_ALL=C grep -m1 'number of atoms/cell' {outs[0]} | "
        f"LC_ALL=C {ctx.awk_cmd} '{{print $5}}'"
    )
    try:
        NUM = int(subprocess.check_output(cmdN, shell=True))
    except Exception as e:
        die(f"Could not parse number of atoms from {outs[0]} ({e})")

    with open("infile.positions", "w") as POF, open("infile.forces", "w") as FOR, open("infile.stat", "w") as STA:
        sample_count = 0

        for out, input_file in zip(outs, ins):
            E_lines = subprocess.check_output(
                f"LC_ALL=C grep '!' {out} | LC_ALL=C {ctx.awk_cmd} '{{printf \"%12.8f\\n\", $5*{RY_TO_EV!r}}}'",
                shell=True
            ).decode().strip().splitlines()
            if not E_lines:
                die(f"Could not parse total energy from {out}")
            E = E_lines[-1]

            # stress (GPa)
            S = read_qe_stress_gpa(out, rybohr3_to_gpa=RYBOHR3_TO_GPA)
            SXX, SXY, SXZ = S[0, 0], S[0, 1], S[0, 2]
            SYY, SYZ = S[1, 1], S[1, 2]
            SZZ = S[2, 2]
            P = -float(np.trace(S)) / 3.0

            # positions from pwscf input
            cr = subprocess.check_output(
                f"""LC_ALL=C {ctx.awk_cmd} -v N={NUM} '
                            BEGIN {{ IGNORECASE=1; flag=0; n=0; }}
                            /^[[:space:]]*ATOMIC_POSITIONS/ {{ flag=1; next; }}
                            flag && NF>=4 {{
                            printf "%20.17f %20.17f %20.17f\\n", $2, $3, $4;
                            n++; if (n>=N) exit;
                            }}
                            ' {input_file}""",
                shell=True
            ).decode().splitlines()

            # forces: Ry/bohr -> eV/Å
            fo = subprocess.check_output(
                f"""LC_ALL=C {ctx.awk_cmd} '/^ *atom .* force =/ {{
                            printf "%20.17f %20.17f %20.17f\\n",
                            $(NF-2)*{RYBOHR_TO_EV_ANG!r},
                            $(NF-1)*{RYBOHR_TO_EV_ANG!r},
                            $NF*{RYBOHR_TO_EV_ANG!r}
                            }}' {out}""",
                shell=True
            ).decode().splitlines()

            if len(cr) != NUM or len(fo) != NUM:
                die(f"Mismatch natoms vs parsed data for {out}/{input_file}: natoms={NUM}, positions={len(cr)}, forces={len(fo)}")

            for i in range(NUM):
                POF.write(cr[i] + "\n")
                FOR.write(fo[i] + "\n")

            # TDEP (type_mdsim.f90, read_from_file) reads the six stress columns as
            # xx, yy, zz, xy, yz, xz.
            STA.write(f" {sample_count} 0 {E} 0 0 0 {P} {SXX} {SYY} {SZZ} {SXY} {SYZ} {SXZ}\n")
            sample_count += 1

    with open("infile.meta", "w") as mf:
        mf.write(f"{NUM} #natoms\n")
        mf.write(f"{sample_count} #SCF calculations\n")
        mf.write("1 timestep\n")
        mf.write(f"{ctx.temp} #temperature of thermostat\n")

    os.makedirs("calc_samples", exist_ok=True)
    for f in ["infile.positions", "infile.forces", "infile.stat", "infile.meta"]:
        shutil.copy(f, "calc_samples")

    return NUM, sample_count

def combine_samples(prev_dir, current_dir):
    """Append a previous geometry's sampled data onto the current one.

    Concatenates ``infile.positions``/``infile.forces``, appends the previous
    ``infile.stat`` rows with shifted sample indices, updates ``infile.meta``'s
    sample count, and refreshes the ``calc_samples`` copy. Two callers:
    :func:`~warpt.workflow_preconditioning.run_preconditioning` (sliding window,
    ``prev_dir`` = the previous geometry's ``local_samples/``) and
    :func:`~warpt.workflow_vc_relax.run_vc_relax`, which merges the previous step's
    samples into the current position step or into the lattice step when that step's
    structure update was below ``--reuse_threshold`` and the lattice did not just change.

    Parameters
    ----------
    prev_dir : str
        Directory whose samples are appended.
    current_dir : str
        Directory receiving the combined samples.

    Returns None.

    Side effects
    ------------
    Appends to ``current_dir/infile.positions``, ``infile.forces`` and ``infile.stat``,
    rewrites line 2 of ``current_dir/infile.meta`` as ``<total> #SCF_steps`` and
    overwrites the four files in ``current_dir/calc_samples/``.
    """
    prev_stat = os.path.join(prev_dir, "infile.stat")
    curr_stat = os.path.join(current_dir, "infile.stat")

    with open(curr_stat) as f:
        current_lines = f.readlines()
    offset = len(current_lines)

    # append positions + forces
    for name in ["positions", "forces"]:
        with open(os.path.join(prev_dir, f"infile.{name}")) as pf, \
             open(os.path.join(current_dir, f"infile.{name}"), "a") as cf:
            cf.write(pf.read())

    # append stat with shifted indices
    with open(prev_stat) as pf, open(curr_stat, "a") as cf:
        for line in pf:
            parts = line.strip().split()
            if parts:
                parts[0] = str(int(parts[0]) + offset)
                cf.write(" ".join(parts) + "\n")

    # update meta
    prev_meta = os.path.join(prev_dir, "infile.meta")
    curr_meta = os.path.join(current_dir, "infile.meta")

    with open(prev_meta) as f:
        prev_lines = f.readlines()
    with open(curr_meta) as f:
        curr_lines = f.readlines()

    total = int(prev_lines[1].split()[0]) + int(curr_lines[1].split()[0])
    curr_lines[1] = f"{total} #SCF_steps\n"

    with open(curr_meta, "w") as f:
        f.writelines(curr_lines)

    # copy to calc_samples
    calc_dir = os.path.join(current_dir, "calc_samples")
    os.makedirs(calc_dir, exist_ok=True)

    for name in ["positions", "forces", "stat", "meta"]:
        shutil.copy(
            os.path.join(current_dir, f"infile.{name}"),
            os.path.join(calc_dir, f"infile.{name}")
        )
