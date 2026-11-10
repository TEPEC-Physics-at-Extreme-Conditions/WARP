# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Resume planner for an existing run tree.

Scans ``<root>/iter_N/`` (``lattice_``, ``position_*``,
``preconditioning/geom_*``) with :func:`~warpt.validity.first_missing_stage` and
reports the newest iteration, the first directory that reports a missing stage and
that stage. A finished ``lattice_`` directory is reported at stage ``phonons``
because ``outfile.grid_dispersions.hdf5``, which ``valid_gruneisen`` requires, is
removed by :func:`~warpt.phonon_stress.internal_stress_voigt`; a finished
``position_*`` directory is reported at stage ``ifc_pressure`` because it is checked
with the default ``want_ifc=True`` and position steps never run ``ifc_pressure``.
The plan is printed only: :func:`~warpt.orchestrator.main` prints it and passes it to
:func:`~warpt.workflow_vc_relax.run_vc_relax`, which logs one line; the continuation
itself is the forward walk inside ``run_vc_relax``, which recognises a finished
lattice step by ``outfile.new_vc_ucposcar``.

Reads directory listings and stage output files under ``ctx.root``; writes
nothing. Changes the working directory while scanning and restores it.

Public functions: ``find_resume_plan``.
"""
import os
import glob

from .validity import first_missing_stage


def _iter_index(path: str) -> int:
    """Trailing integer of an ``iter_N`` directory name (``-1`` if unparseable)."""
    base = os.path.basename(path)
    try:
        return int(base.split("_")[-1])
    except Exception:
        return -1


def _pos_index(path: str) -> int:
    """Trailing integer of a ``position_N`` / ``geom_N`` directory name (``-1`` if none)."""
    base = os.path.basename(path)
    try:
        return int(base.split("_")[-1])
    except Exception:
        return -1


def find_resume_plan(ctx) -> dict:
    """Locate where an interrupted run should continue.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Run configuration; ``ctx.root`` and ``ctx.master_scf`` are used.

    Returns
    -------
    dict
        ``iter``: iteration to continue: 0 when no ``iter_*`` exists, the last
        index when a checked directory of the last iteration reports a missing
        stage, last index + 1 otherwise (on the normal path only when the last
        iteration has neither a ``lattice_`` nor a ``position_*`` directory, since
        both report a missing stage even when finished, see below); ``dir``:
        directory to continue in, or ``None``; ``stage``: its first missing stage,
        or ``None``; ``kind``: ``"lattice"``, ``"position"``, ``"preconditioning"``
        or ``"none"``. Checks, in order, ``lattice_`` with ``want_gruneisen`` and
        ``want_lattice`` set (a finished lattice step is reported at stage
        ``"phonons"`` because ``outfile.grid_dispersions.hdf5`` is removed by
        :func:`~warpt.phonon_stress.internal_stress_voigt`), ``position_*`` newest
        first with the default ``want_ifc=True`` (a finished position step is
        reported at stage ``"ifc_pressure"`` because position steps never run
        ``ifc_pressure``), then ``preconditioning/geom_*`` newest first (DOS stage
        required). The plan is printed only; ``run_vc_relax`` resumes by its own
        forward walk.

    Side effects
    ------------
    Changes into each inspected directory and restores the original working
    directory before returning.
    """
    root = ctx.root
    cwd0 = os.getcwd()

    try:
        iters = sorted(glob.glob(os.path.join(root, "iter_*")), key=_iter_index)
        if not iters:
            return {"iter": 0, "dir": None, "stage": None, "kind": "none"}

        last_iter = iters[-1]
        i = _iter_index(last_iter)

        # lattice_ first; a finished lattice step also reports "phonons" (grid file removed)
        lattice_dir = os.path.join(last_iter, "lattice_")
        if os.path.isdir(lattice_dir):
            os.chdir(lattice_dir)
            miss = first_missing_stage(
                ctx,
                want_dos=False,
                want_gruneisen=True,
                want_lattice=True,
            )
            if miss is not None:
                return {"iter": i, "dir": lattice_dir, "stage": miss, "kind": "lattice"}

        # Else the newest position_* that reports a missing stage (a finished one reports
        # "ifc_pressure", see the module docstring)
        positions = sorted(glob.glob(os.path.join(last_iter, "position_*")), key=_pos_index)
        for pdir in reversed(positions):
            os.chdir(pdir)
            miss = first_missing_stage(
                ctx,
                want_dos=False,
                want_gruneisen=False,
                want_lattice=False,
            )
            if miss is not None:
                return {"iter": i, "dir": pdir, "stage": miss, "kind": "position"}

        # Else the newest preconditioning geom_* that reports a missing stage
        pre = os.path.join(last_iter, "preconditioning")
        if os.path.isdir(pre):
            geoms = sorted(glob.glob(os.path.join(pre, "geom_*")), key=_pos_index)
            for gdir in reversed(geoms):
                os.chdir(gdir)
                miss = first_missing_stage(
                    ctx,
                    want_dos=True,
                    want_gruneisen=False,
                    want_lattice=False,
                )
                if miss is not None:
                    return {"iter": i, "dir": gdir, "stage": miss, "kind": "preconditioning"}

        # No checked directory of the last iteration reports a missing stage: continue with
        # the next iteration
        return {"iter": i + 1, "dir": None, "stage": None, "kind": "none"}

    finally:
        os.chdir(cwd0)