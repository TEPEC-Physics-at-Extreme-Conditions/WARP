# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Calculator invocation and a small parallel job pool.

Builds the ``-input <file>`` command for Quantum ESPRESSO and for the MLIP and
VASP wrappers (all three accept it), detects a finished run by the ``JOB DONE.``
marker in its output file, and runs a bounded number of jobs concurrently.
Writes one output file per job (stdout and stderr combined) and records the
batch wall time in :mod:`warpt.timing`.

Public functions: ``build_cmd``, ``job_done``, ``run_jobs_parallel``.
"""
import os
import time
import subprocess
from typing import List, Tuple

def build_cmd(pwscf_run: str, scf_in: str) -> List[str]:
    """Return the argv list ``<pwscf_run tokens> -input <scf_in>``.

    Parameters
    ----------
    pwscf_run : str
        Calculator command, split on whitespace; ``["pw.x"]`` when empty.
    scf_in : str
        Input file name passed after ``-input``.
    """
    cmd = pwscf_run.split() if pwscf_run else ["pw.x"]
    cmd += ["-input", scf_in]
    return cmd

def job_done(out_path: str) -> bool:
    """True when ``out_path`` exists and contains the string ``JOB DONE.``; False when it is
    missing or cannot be read.
    """
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", errors="ignore") as f:
            return ("JOB DONE." in f.read())
    except Exception:
        return False

def run_jobs_parallel(jobs: List[Tuple[List[str], str]], max_concurrent: int = 4, poll_s: float = 5.0) -> None:
    """Run calculator jobs concurrently and wait for all of them.

    Parameters
    ----------
    jobs : list[tuple[list[str], str]]
        ``(argv, out_path)`` pairs; each process's stdout and stderr are written
        to ``out_path`` (truncated on start).
    max_concurrent : int
        Maximum number of simultaneously running processes (default 4).
    poll_s : float
        Upper bound of the polling interval in seconds (default 5.0); polling
        starts at 0.05 s and grows by a factor 1.5 while nothing finishes.

    Side effects
    ------------
    Launches one subprocess per job; records one :func:`~warpt.timing.record_cmd`
    entry for the whole batch, keyed by the first job's command.

    Raises
    ------
    ``RuntimeError`` as soon as a finished job's output lacks ``JOB DONE.``; the
    message includes the text after ``JOB FAILED.`` when the output contains that
    marker. Jobs still running are not terminated.
    """
    pending = list(jobs)
    running: List[Tuple[subprocess.Popen, str]] = []

    def launch(cmd: List[str], out_path: str):
        """Start one subprocess, writing combined stdout/stderr to ``out_path``."""
        fout = open(out_path, "w")
        p = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT)
        return p, out_path, fout

    import time as _time
    from .timing import record_cmd
    _t0 = _time.perf_counter()
    _first_cmd = jobs[0][0] if jobs else "force-eval"

    open_files = []
    poll_min = 0.05
    cur_poll = poll_min

    try:
        while pending or running:
            while pending and len(running) < max_concurrent:
                cmd, out_path = pending.pop(0)
                p, tag, fout = launch(cmd, out_path)
                running.append((p, tag))
                open_files.append(fout)

            finished_idx = []
            for i, (p, tag) in enumerate(running):
                if p.poll() is not None:
                    if not job_done(tag):
                        # The MLIP and VASP wrappers print "JOB FAILED. <reason>"; the reason
                        # is copied into the exception message, so the cause is reported
                        # there and not only in the job's output file.
                        why = ""
                        try:
                            with open(tag, errors="ignore") as fh:
                                why = next((ln.split("JOB FAILED.", 1)[1].strip()
                                            for ln in fh if "JOB FAILED." in ln), "")
                        except OSError:
                            pass
                        raise RuntimeError(f"SCF/MLIP/VASP run failed for {tag} (no 'JOB DONE.')"
                                           + (f": {why}" if why else ""))
                    finished_idx.append(i)

            for i in reversed(finished_idx):
                running.pop(i)

            if not finished_idx:
                # Adaptive back-off: sleep cur_poll (from poll_min, x1.5 per idle poll,
                # capped at poll_s); reset to poll_min whenever a job finishes.
                time.sleep(min(cur_poll, poll_s))
                cur_poll = min(cur_poll * 1.5, poll_s)
            else:
                cur_poll = poll_min
    finally:
        # One timing record for the whole batch, keyed by the first job's command.
        record_cmd(_first_cmd, _time.perf_counter() - _t0)
        for f in open_files:
            try:
                f.close()
            except Exception:
                pass
