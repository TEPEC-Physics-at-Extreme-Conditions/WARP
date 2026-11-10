# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Stage-granular, idempotent completion of one work directory.

:func:`complete_dir` runs only the pipeline stages that
:func:`~warpt.validity.first_missing_stage` reports missing in the current
working directory, in order, and can be called on a fresh, partial or complete
directory. Together with the SCF-level resume in
:func:`~warpt.tdep_sampling.do_sampling_and_run`, an interrupted directory is
continued as follows:

    interrupted during        re-run on the next call
    ------------------------  ---------------------------------------------
    canonical_configuration   sampling (a partial contcar set is cleared and
                              the whole set regenerated), then later stages
    an SCF                    only the SCFs whose output lacks "JOB DONE.",
                              then later stages
    collect                   collect (and on_collect), then later stages
    extract_forceconstants    extract, then later stages
    ifc_pressure              ifc_pressure, then later stages
    phonon_dispersion         phonons, then the lattice update when required
    lattice update            the lattice update only

Reads and writes only through the stage functions it calls, in the current
working directory. Not used by the command-line workflows, which resume through
the forward walk in :func:`~warpt.workflow_vc_relax.run_vc_relax`.

Public functions: ``complete_dir``.
"""
import os

from .utilities import die, safe_run, omp_threads
from .tdep_sampling import do_sampling_and_run
from .tdep_collect import collect_stats
from .tdep_extract_fc import run_extract_fc
from .validity import first_missing_stage


def complete_dir(
    ctx,
    state,
    *,
    expected_ns,
    extract_mode,
    want_dos=False,
    want_gruneisen=False,
    want_lattice=False,
    want_ifc=True,
    on_collect=None,
    run_phonons=None,
    run_lattice=None,
):
    """Run only the missing stages of the current working directory.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Run configuration.
    state : context_state.WarpState
        Mutable run state, passed to the stage functions.
    expected_ns : int
        Number of configurations this step should have.
    extract_mode : str
        ``"second"``, ``"second_third"`` or ``"second_third_first"`` for
        :func:`~warpt.tdep_extract_fc.run_extract_fc`.
    want_dos, want_gruneisen, want_lattice, want_ifc : bool
        Forwarded to :func:`~warpt.validity.first_missing_stage` to define which
        optional stages this directory needs.
    on_collect : callable or None
        Run immediately after ``collect_stats`` so a resume that redoes collect
        also redoes the caller's sample combination.
    run_phonons : callable or None
        Produces the phonon output required by ``want_dos``
        (``outfile.dispersion_relations.hdf5``) or ``want_gruneisen``
        (``outfile.grid_dispersions.hdf5`` with Grüneisen data). The only producer of
        that grid file, :func:`~warpt.phonon_stress.internal_stress_voigt`, removes it
        after extracting ``sigma_ph``, so ``valid_gruneisen`` then fails and a
        ``want_gruneisen`` directory cannot complete: the 16-action guard exits via
        ``die``.
    run_lattice : callable or None
        Produces ``outfile.new_vc_ucposcar``.

    Returns
    -------
    int
        Number of stage actions executed; 0 when the directory was already complete.

    Side effects
    ------------
    Whatever the stage functions write in the working directory; ``ifc_pressure``
    is launched as a shell command; ``extract_forceconstants`` runs with
    ``OMP_NUM_THREADS=1``.

    Raises
    ------
    Calls :func:`~warpt.utilities.die` (exit 1) when the setup files are missing,
    a needed callback is absent, ``ifc_pressure`` fails, a stage does not advance
    within 16 actions, or an unknown stage name is returned. Errors raised by the
    stage functions propagate: ``SystemExit`` via ``die`` from sampling, collect
    and extraction, ``RuntimeError`` from
    :func:`~warpt.runner.run_jobs_parallel` when a job's output lacks
    ``JOB DONE.``.
    """
    actions = 0
    # Guard against an infinite loop if a stage silently fails to produce its
    # output: each stage should advance first_missing_stage exactly once.
    max_iters = 16
    while True:
        miss = first_missing_stage(
            ctx,
            want_dos=want_dos,
            want_gruneisen=want_gruneisen,
            want_lattice=want_lattice,
            want_ifc=want_ifc,
            expected_ns=expected_ns,
        )
        if miss is None:
            return actions

        actions += 1
        if actions > max_iters:
            die(f"[resume] stage '{miss}' did not advance in {os.getcwd()} "
                f"(its output is still missing after running it)")

        if miss == "setup":
            die(f"[resume] setup missing in {os.getcwd()} "
                f"(infiles must be prepared by the caller before complete_dir)")

        elif miss in ("sampling", "scf"):
            # SCF-level resume lives in do_sampling_and_run: a complete contcar
            # set is reused and only unfinished SCFs are queued.
            do_sampling_and_run(ctx, state, expected_ns)

        elif miss == "collect":
            collect_stats(ctx, state)
            if on_collect is not None:
                on_collect()

        elif miss == "extract":
            with omp_threads(1):
                run_extract_fc(ctx, state, mode=extract_mode)

        elif miss == "ifc_pressure":
            rc, _, _ = safe_run("ifc_pressure")
            if rc != 0:
                die("ifc_pressure failed")

        elif miss == "phonons":
            if run_phonons is None:
                die("[resume] phonons stage missing but no run_phonons callback provided")
            run_phonons()

        elif miss == "lattice":
            if run_lattice is None:
                die("[resume] lattice stage missing but no run_lattice callback provided")
            run_lattice()

        else:
            die(f"[resume] unknown stage: {miss}")
