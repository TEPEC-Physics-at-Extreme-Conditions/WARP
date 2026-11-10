#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Wall-time records of external commands and workflow stages.

Two tables are kept in memory: external commands keyed by binary name and workflow
stages keyed by stage name (``sampling + forces``, ``collect + fit IFCs``,
``lattice update``, ``preconditioning``, ...). Commands are recorded by
:func:`~warpt.utilities.safe_run` (``generate_structure``, ``refine_structure``,
``extract_forceconstants``, ``ifc_pressure``, ``phonon_dispersion_relations``) and
by :func:`~warpt.runner.run_jobs_parallel`, which records one entry per calculator
batch keyed by the first job's command: the first token of ``--pwx`` for QE
(``mpirun`` with the default command), or the backend module for MLIP and VASP
(``warpt.MLIP_interface``, ``warpt.VASP_interface``). ``canonical_configuration``
is launched directly by :mod:`warpt.tdep_sampling` and has no entry in the command
table; its wall time is contained in the ``sampling + forces`` stage. Stages nest
inside each other and contain the commands they launch, so the two tables overlap
and are read separately, never added.

Reads the environment variables ``WARPT_TIMING`` (``"0"`` disables all records
and output; default ``"1"``) and ``WARPT_TIME_MIN_S`` (seconds; a per-command
line is printed only for calls at least this long; default 1.0). Writes
``[time] ...`` lines to stdout as they happen and the two tables on
:func:`summary`. Never raises and never alters control flow: every entry point
swallows its own exceptions.

Public functions and classes: ``tool_name``, ``record_cmd``, ``stage``, ``summary``.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

# name -> [total seconds, call count]
_CMD: "OrderedDict[str, list]" = OrderedDict()
_STAGE: "OrderedDict[str, list]" = OrderedDict()

# A per-command line is printed only for calls at least this long (WARPT_TIME_MIN_S).
_MIN_PRINT_S = float(os.environ.get("WARPT_TIME_MIN_S", "1.0"))
_ENABLED = os.environ.get("WARPT_TIMING", "1") != "0"


def _add(table, name, dt):
    """Add ``dt`` seconds and one call to ``table[name]``, creating the entry when needed."""
    e = table.setdefault(name, [0.0, 0])
    e[0] += dt
    e[1] += 1


def tool_name(cmd) -> str:
    """Return the binary name a command invokes.

    Parameters
    ----------
    cmd : str or sequence of str
        Shell string (only the part before the first ``|`` is used) or argv
        list. Leading ``VAR=value`` assignments are skipped; ``python* -m
        <module>`` yields ``<module>`` (e.g. ``warpt.MLIP_interface``).

    Returns
    -------
    str
        Basename of the executable, or ``"?"`` when it cannot be determined.
    """
    try:
        if isinstance(cmd, (list, tuple)):
            parts = [str(c) for c in cmd]
        else:
            parts = str(cmd).split("|")[0].split()
        parts = [p for p in parts if "=" not in p.split("/")[-1][:1] or True]
        i = 0
        while i < len(parts) and ("=" in parts[i] and not parts[i].startswith("-")):
            i += 1                                   # skip VAR=value prefixes
        if i >= len(parts):
            return "?"
        base = os.path.basename(parts[i])
        if base.startswith("python") and "-m" in parts:
            j = parts.index("-m")
            if j + 1 < len(parts):
                return parts[j + 1]                  # e.g. warpt.MLIP_interface
        return base
    except Exception:
        return "?"


def record_cmd(cmd, dt: float) -> None:
    """Add one external command's wall time to the command table.

    Parameters
    ----------
    cmd : str or sequence of str
        Command as accepted by :func:`tool_name`.
    dt : float
        Wall time in seconds.

    Prints a ``[time]`` line when ``dt`` is at least ``WARPT_TIME_MIN_S``. No-op
    when ``WARPT_TIMING`` is ``"0"``; never raises.
    """
    if not _ENABLED:
        return
    try:
        name = tool_name(cmd)
        _add(_CMD, name, dt)
        if dt >= _MIN_PRINT_S:
            print(f"[time] {name:34s} {dt:9.1f} s", flush=True)
    except Exception:
        pass


class stage:
    """Time a named workflow stage, as a context manager or a decorator.

        with stage("lattice update"):
            ...

        @stage("sampling + forces")
        def do_sampling_and_run(...):
            ...

    Parameters
    ----------
    name : str
        Stage label used as the table key.

    On exit the elapsed wall time is added to the stage table and printed as a
    ``[time] stage`` line (unless ``WARPT_TIMING`` is ``"0"``). Exceptions raised
    in the body always propagate.
    """

    def __init__(self, name: str):
        """Store the stage label; the clock starts on entry, not here."""
        self.name = name

    def __enter__(self):
        """Start the clock and return the stage object."""
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        """Record and print the elapsed time; returns False, so exceptions propagate."""
        if not _ENABLED:
            return False
        try:
            dt = time.perf_counter() - self._t0
            _add(_STAGE, self.name, dt)
            print(f"[time] stage {self.name:28s} {dt:9.1f} s", flush=True)
        except Exception:
            pass
        return False                                  # never swallow exceptions

    def __call__(self, fn):
        """Use the stage as a decorator: every call of ``fn`` is timed under this label."""
        import functools

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            """Call the decorated function inside a fresh stage of the same label."""
            with stage(self.name):
                return fn(*a, **kw)
        return wrapper


def summary(title: str = "timing so far") -> None:
    """Print both tables to stdout, largest total first.

    Parameters
    ----------
    title : str
        Heading printed before each table (default ``"timing so far"``).

    Prints nothing when timing is disabled or no record exists. May be called
    repeatedly; the tables are not reset.
    """
    if not _ENABLED or (not _CMD and not _STAGE):
        return
    try:
        for label, table in (("external commands", _CMD), ("workflow stages", _STAGE)):
            if not table:
                continue
            tot = sum(v[0] for v in table.values())
            print(f"\n=== {title}: {label} ===")
            print(f"    {'':34s}{'calls':>7s}{'total':>12s}{'mean':>10s}{'share':>8s}")
            for name, (s, n) in sorted(table.items(), key=lambda kv: -kv[1][0]):
                print(f"    {name:34s}{n:>7d}{s / 3600:>10.2f} h{s / max(n, 1):>9.1f} s"
                      f"{100 * s / tot if tot else 0:>7.1f}%")
            print(f"    {'TOTAL':34s}{'':>7s}{tot / 3600:>10.2f} h")
        print(flush=True)
    except Exception:
        pass
