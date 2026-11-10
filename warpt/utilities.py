# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Process, shell and filesystem helpers used throughout WARP.

Provides a context manager that pins ``OMP_NUM_THREADS`` around a TDEP call, a fatal-error
exit (:func:`die`), a non-raising shell runner (:func:`safe_run`, which reports each command's
wall time to :mod:`warpt.timing`), a create-and-chdir helper, and the translator from the
``--supercell`` integers to ``generate_structure`` flags used by the orchestrator.

Public functions: ``omp_threads``, ``die``, ``safe_run``, ``make_and_enter``, ``parse_supercell``,
``parse_masses``.
"""
import os
import re
import sys
import subprocess
from typing import Tuple, List
from contextlib import contextmanager

@contextmanager
def omp_threads(n):
    """Temporarily set ``OMP_NUM_THREADS`` for the duration of a ``with`` block.

    The previous value is restored (or the variable unset) on exit, even if the
    body raises. Used to pin individual TDEP tool invocations to a chosen thread
    count without leaking the setting into the rest of the run.

    Parameters
    ----------
    n : int
        Thread count to export as ``OMP_NUM_THREADS`` inside the block.
    """
    old = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old

def die(msg: str, code: int = 1, prefix: str = "WARP") -> None:
    """Print a fatal error to stderr and abort the process.

    Parameters
    ----------
    msg : str
        Human-readable error message.
    code : int
        Process exit code (raised via ``SystemExit``).
    prefix : str
        Name inserted in the ``Error in <prefix>:`` header (default ``"WARP"``).

    Raises
    ------
    SystemExit
        Always.
    """
    print(f"Error in {prefix}: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)

def safe_run(cmd: str) -> Tuple[int, str, str]:
    """Run ``cmd`` through the shell and return ``(returncode, stdout, stderr)``.

    Never raises: if ``subprocess.run`` itself fails the return is ``(999, "", message)``. The
    wall time is reported to ``timing.record_cmd`` in every case.
    """
    import time as _time
    from .timing import record_cmd
    _t0 = _time.perf_counter()
    try:
        cp = subprocess.run(
            cmd, shell=True, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return cp.returncode, cp.stdout, cp.stderr
    except Exception as e:
        return 999, "", f"safe_run exception: {e}"
    finally:
        # Commands run through here are recorded by binary name. Calculator batches are
        # recorded by runner.run_jobs_parallel instead; canonical_configuration is launched by
        # tdep_sampling with subprocess.run and has no entry in the command table.
        record_cmd(cmd, _time.perf_counter() - _t0)

def make_and_enter(path: str) -> None:
    """Create ``path`` if needed and ``chdir`` into it.

    Prints whether the directory was created or already existed, then changes
    the process working directory into it.

    Parameters
    ----------
    path : str
        Directory to create and enter.
    """
    if not os.path.isdir(path):
        print(f"  Creating directory {os.path.basename(path)}!")
    else:
        print(f"  Entering existing directory {os.path.basename(path)}!")
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

def parse_supercell(sc: List[str]) -> List[str]:
    """Translate the ``--supercell`` integers into ``generate_structure`` flags.

    1 integer -> ``--desired_na N``; 3 -> ``--dimensions a b c``; 9 ->
    ``--nondiagonal_dimensions`` with the nine matrix entries. Prints the choice; exits via
    ``die`` for an empty list or any other count.
    """
    if not sc:
        die("Supercell size must be specified!")
    ints = list(map(int, sc))
    if len(ints) == 1:
        print(f"  Building supercell close to {ints[0]} atoms!")
        return ["--desired_na", str(ints[0])]
    if len(ints) == 3:
        print(f"  Using a {ints[0]}x{ints[1]}x{ints[2]} supercell")
        return ["--dimensions", *map(str, ints)]
    if len(ints) == 9:
        print("  Using matrix")
        print(f"      {ints[0]} {ints[1]} {ints[2]}")
        print(f"      {ints[3]} {ints[4]} {ints[5]}")
        print(f"      {ints[6]} {ints[7]} {ints[8]}")
        return ["--nondiagonal_dimensions", *map(str, ints)]
    die("Supercell requires 1, 3 or 9 integers")
    return []


def parse_masses(text: str) -> dict:
    """Parse the ``--masses`` option, ``ELEMENT=MASS[,ELEMENT=MASS...]`` in atomic mass units.

    Returns ``{symbol: mass}``. Exits via :func:`die` when an entry is not of the form
    ``El=number``, the symbol is not a chemical element or the mass is not positive.
    """
    from ase.data import chemical_symbols
    out = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        m = re.fullmatch(r"([A-Z][a-z]?)\s*=\s*([0-9.eE+-]+)", item)
        if not m or m.group(1) not in chemical_symbols:
            die(f"--masses: expected ELEMENT=MASS, got '{item}'")
        try:
            mass = float(m.group(2))
        except ValueError:
            die(f"--masses: '{m.group(2)}' is not a number")
        if mass <= 0:
            die(f"--masses: the mass of {m.group(1)} must be positive")
        out[m.group(1)] = mass
    if not out:
        die("--masses: no ELEMENT=MASS entry given")
    return out
