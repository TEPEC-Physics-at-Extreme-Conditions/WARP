# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Force-constant preconditioning schedule run before the first variable-cell iteration.

Runs a fixed schedule of sampling geometries under ``<cwd>/preconditioning/geom_<k>``.
geom_0 samples from the bootstrap force constants (``state.use_fake_fc``); every later geom
samples from the force constants of the previous geom and is fitted on its own samples plus
the previous geom's ``local_samples/`` snapshot (sliding window of two geoms). From geom_1 on,
each geom is gated on two criteria: the minimum phonon frequency of the DOS dispersion must
exceed -1e-5 THz and the second-order cross-validation R^2 must be at least ``CV_R2_MIN``.
The first-order structure update is only requested from geom_2 on, and only when the
previous geom's minimum frequency exceeds ``ctx.firstorder_min_freq``.

Reads ``ctx.ucp``, ``ctx.sc_opts``, ``ctx.precond_schedule`` (default
``[8, 32, 32, 48, 48, 64, 64]``, CLI ``--precond_schedule``) and ``ctx.firstorder_min_freq``
(``--firstorder_min_freq``, THz). In the preconditioning root it writes ``infile.ucposcar``
and ``infile.ssposcar``; per geom, through the imported modules and the TDEP tools, the
TDEP infiles, ``contcar_conf*``, the calculator outputs, ``infile.{positions,forces,stat,meta}``,
``local_samples/``, ``extract_forceconstants.log``, ``outfile.forceconstant*``,
``infile.forceconstant*``, ``outfile.new_ucposcar``, ``outfile.stress_*`` (``ifc_pressure``)
and ``outfile.dispersion_relations*``. Launches ``generate_structure``,
``canonical_configuration``, the calculator, ``extract_forceconstants``, ``ifc_pressure`` and
``phonon_dispersion_relations``.

Public functions: ``parse_cv_r2``, ``save_local_samples_snapshot``, ``run_preconditioning``.
"""
import os
import shutil

from .utilities import make_and_enter, die, safe_run, omp_threads
from .tdep_sampling import do_sampling_and_run
from .tdep_collect import collect_stats, combine_samples
from .tdep_extract_fc import run_extract_fc
from .io_workflow import copy_previous, prepare_infiles_from_uc
from .phonon_stress import run_dos_phonons
from .validity import first_missing_stage
from .timing import stage

# Second-order cross-validation R^2 (from extract_forceconstants.log) below which a
# preconditioning fit fails the gate. In strongly anharmonic systems this R^2 measures the
# harmonic share of the forces rather than the quality of the fit, so the threshold is moderate.
CV_R2_MIN = 0.60


def parse_cv_r2(logpath="extract_forceconstants.log"):
    """Parse the cross-validation R^2 values from an ``extract_forceconstants`` log.

    Scans for the first line containing ``cross-validation`` and takes the third token of the
    first following ``second order:`` and ``third order:`` lines; once the second-order value
    has been read, the scan stops at the next line starting with ``rms`` or, in any letter
    case, ``forces``.

    Parameters
    ----------
    logpath : str
        Log file (default ``extract_forceconstants.log``).

    Returns
    -------
    tuple[float or None, float or None]
        ``(r2_second, r2_third)``; ``None`` for a value not found or when the file is missing.
    """
    if not os.path.exists(logpath):
        return None, None
    r2_2 = r2_3 = None
    in_cv = False
    with open(logpath) as f:
        for line in f:
            if "cross-validation" in line:
                in_cv = True
                continue
            if not in_cv:
                continue
            s = line.strip()
            # Only the first "second/third order:" after the cross-validation
            # header is the CV R^2; later blocks (rms, FORCES) reuse the label.
            if s.startswith("second order:") and r2_2 is None:
                try:
                    r2_2 = float(s.split()[2])
                except (IndexError, ValueError):
                    pass
            elif s.startswith("third order:") and r2_3 is None:
                try:
                    r2_3 = float(s.split()[2])
                except (IndexError, ValueError):
                    pass
            # CV block ends once the next section header appears.
            elif r2_2 is not None and (s.startswith("rms") or s.upper().startswith("FORCES")):
                break
    return r2_2, r2_3


def _read_min_freq_hdf5(path="outfile.dispersion_relations.hdf5"):
    """Minimum phonon frequency from a TDEP dispersion file, or None if unreadable."""
    if not os.path.exists(path):
        return None
    try:
        import h5py
        import numpy as np
        with h5py.File(path, "r") as f:
            return float(np.min(f["frequencies"][()]))
    except Exception:
        return None


def _precond_gate_passed(minw, r2_2, r2_3, disp_thr):
    """Print the gate diagnostics and return True iff ``minw > disp_thr`` (THz) and
    ``r2_2 >= CV_R2_MIN``; a ``None`` value fails its criterion.
    """
    def _fmt(x):
        """Format a float for the gate log, or ``n/a`` if ``None``."""
        return "n/a" if x is None else f"{x:.4f}"
    mtxt = "n/a" if minw is None else f"{minw:.6e}"
    print(f"[preconditioning] min(freq) = {mtxt}", flush=True)
    print(f"[preconditioning] cross-val R^2: 2nd={_fmt(r2_2)} 3rd={_fmt(r2_3)} "
          f"(gate: 2nd >= {CV_R2_MIN})", flush=True)
    if (r2_3 is not None) and (r2_3 < 0.0):
        print("[preconditioning][warn] third-order CV R^2 is negative "
              "(third-order fit likely overfit)", flush=True)
    stable = (minw is not None) and (minw > disp_thr)
    cv_ok = (r2_2 is not None) and (r2_2 >= CV_R2_MIN)
    if stable and cv_ok:
        print("[preconditioning] stable AND well cross-validated → proceed", flush=True)
        return True
    if stable and not cv_ok:
        print(f"[preconditioning] phonons stable but 2nd-order CV R^2 ({_fmt(r2_2)}) "
              f"< {CV_R2_MIN} → keep sampling (likely overfit)", flush=True)
    return False


def save_local_samples_snapshot(geom_dir):
    """Copy this geom's TDEP inputs into a ``local_samples/`` snapshot.

    Copies the geom's ``infile.{positions,forces,stat,meta}`` into ``local_samples/``.
    :func:`run_preconditioning` calls it before :func:`warpt.tdep_collect.combine_samples`
    appends the previous geom's rows to those files, so the next geom can reuse this geom's
    own samples.

    Parameters
    ----------
    geom_dir : str
        The preconditioning geometry directory.
    """
    snap = os.path.join(geom_dir, "local_samples")
    os.makedirs(snap, exist_ok=True)
    for name in ["positions", "forces", "stat", "meta"]:
        shutil.copy(
            os.path.join(geom_dir, f"infile.{name}"),
            os.path.join(snap, f"infile.{name}")
        )

@stage("preconditioning")
def run_preconditioning(ctx, state):
    """Run the preconditioning schedule under ``preconditioning/`` in the current directory.

    For each sample count ``ns`` in the schedule (``ctx.precond_schedule`` or the default
    ``[8, 32, 32, 48, 48, 64, 64]``) a directory ``geom_<k>`` is created and, unless it is
    already complete, sampled with ``ns`` configurations, combined with the previous geom's
    ``local_samples/`` (k >= 1), fitted with ``extract_forceconstants`` (FC2 + FC3; the
    first-order structure update is added from geom_2 on when the previous geom's minimum
    frequency exceeds ``ctx.firstorder_min_freq`` THz), followed by ``ifc_pressure`` and a DOS
    dispersion on the first valid q-grid of 22^3, 20^3, 18^3, 16^3, 14^3. From geom_1 on the
    geom is gated: minimum frequency > -1e-5 THz and second-order cross-validation
    R^2 >= ``CV_R2_MIN``; the first geom that passes ends the schedule. An already-complete
    geom (resume) is re-gated instead of recomputed.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Run configuration (``ucp``, ``sc_opts``, ``precond_schedule``, ``firstorder_min_freq``).
    state : context_state.WarpState
        Mutable run state; ``use_fake_fc`` is set to True for geom_0 and False for every
        later geom that is computed.

    Returns
    -------
    str
        Absolute path of the geom directory that passed the gate; the process is left in
        that directory. The one exception is a one-element schedule whose geom_0 is already
        complete: that directory is returned, but the process is left in
        ``preconditioning/``.

    Side effects
    ------------
    Creates ``preconditioning/`` and the ``geom_<k>`` directories and writes the files listed
    in the module docstring. Exits via :func:`warpt.utilities.die` when ``ifc_pressure``
    fails, when no q-grid yields a valid dispersion (geom_1 on), or when the last geom of the
    schedule fails the gate (geom_0 is never gated, so a one-element schedule always exits
    here on a fresh run).
    """
    make_and_enter("preconditioning")

    precond_root = os.getcwd()  # every geom_<k> directory is created under this root

    # infile.ucposcar and infile.ssposcar in the preconditioning root, built from ctx.ucp
    shutil.copy(ctx.ucp, "infile.ucposcar")
    prepare_infiles_from_uc(ctx.ucp, ctx.sc_opts)

    # geom_0 samples from the bootstrap (fake) force constants; ctx.precond_schedule
    # (--precond_schedule) overrides the default counts.
    schedule = list(ctx.precond_schedule) if getattr(ctx, "precond_schedule", None) else [8, 32, 32, 48, 48, 64, 64]
    disp_thr = -1e-5
    print(f"[preconditioning] sample schedule = {schedule}", flush=True)

    prev_geom_dir = None

    # min(freq) from the previous geom's dispersion, used to decide whether the
    # first-order structure update is safe to apply (see below). None until the
    # first gate check runs (geom_1).
    last_minw = None
    fo_thr = float(getattr(ctx, "firstorder_min_freq", -1.0))

    for g, ns in enumerate(schedule):
        geom_dir = os.path.join(precond_root, f"geom_{g}")
        make_and_enter(geom_dir)

        miss = first_missing_stage(ctx, want_dos=True, want_gruneisen=False, want_lattice=False, expected_ns=ns)
        if miss is None:
            # Already-complete geom (resume). Recreate the local_samples/ snapshot if it is
            # missing and re-apply the stability + cross-validation gate, so a geom that already
            # passes ends the schedule on resume.
            snap = os.path.join(geom_dir, "local_samples")
            if not os.path.exists(os.path.join(snap, "infile.positions")):
                save_local_samples_snapshot(geom_dir)

            if g >= 1:
                print(f"[preconditioning][resume] geom_{g} already complete; re-checking gate", flush=True)
                minw = _read_min_freq_hdf5("outfile.dispersion_relations.hdf5")
                # keep the first-order stability decision consistent on resume
                last_minw = minw
                r2_2, r2_3 = parse_cv_r2("extract_forceconstants.log")
                if _precond_gate_passed(minw, r2_2, r2_3, disp_thr):
                    prev_geom_dir = geom_dir
                    break
                if g == len(schedule) - 1:
                    die("Preconditioning failed: unstable phonons after full schedule")

            os.chdir(precond_root)
            prev_geom_dir = geom_dir
            continue

        # --- geometry hand-off: geom_k is not complete, so its infiles are (re)written here ---
        if g == 0:
            # geom_0 has no previous geom and no force constants yet: build the infiles from
            # the preconditioning root's infile.ucposcar.
            prepare_infiles_from_uc('../infile.ucposcar', ctx.sc_opts)
        else:
            copy_previous(prev_geom_dir, ctx.sc_opts)

        # bootstrap (fake) force constants are used for geom_0 only
        state.use_fake_fc = (g == 0)

        # --- sampling ---
        do_sampling_and_run(ctx, state, ns, force_regenerate=True)
        collect_stats(ctx, state)

        # snapshot this geom's own samples, before combination, for sliding-window reuse
        save_local_samples_snapshot(geom_dir)

        # sliding window: geom_k is fitted on its own samples plus geom_{k-1}'s local_samples
        if g > 0:
            combine_samples(os.path.join(prev_geom_dir, "local_samples"), geom_dir)
            print("[preconditioning] reusing previous geom local samples", flush=True)

        # The first-order structure update (mode "second_third_first") is the Newton step
        # du = -Phi^-1 F on the effective FC2 Hessian Phi. When Phi is indefinite (imaginary
        # modes) the step points uphill along those branches, so it is only requested once the
        # previous geom's minimum frequency exceeds ctx.firstorder_min_freq; until then FC2/FC3
        # are refined by sampling alone.
        if g <= 1:
            # geom_0 and geom_1: FC2 + FC3 only, no first-order structure update.
            fit_mode = "second_third"
        elif last_minw is None:
            fit_mode = "second_third"
            print("[preconditioning] no min(freq) yet -> deferring first-order "
                  "structure update", flush=True)
        elif last_minw <= fo_thr:
            fit_mode = "second_third"
            print(f"[preconditioning] min(freq) {last_minw:.4g} THz <= {fo_thr:g} "
                  f"-> FC2 indefinite, deferring first-order structure update "
                  f"(refining FC by sampling only)", flush=True)
        else:
            fit_mode = "second_third_first"
            print(f"[preconditioning] min(freq) {last_minw:.4g} THz > {fo_thr:g} "
                  f"-> applying first-order structure update", flush=True)

        with omp_threads(1):
            run_extract_fc(ctx, state, mode=fit_mode)

        # ifc_pressure writes outfile.stress_{dft,fc2,fc3,res} for the force constants just fitted
        rc, _, _ = safe_run("ifc_pressure")
        if rc != 0:
            die("ifc_pressure failed")

        # geom_0: dispersion and DOS are computed for the record only; the gate starts at geom_1.
        if g == 0:
            with omp_threads(16):
                success, minw = run_dos_phonons(
                grids=[(22, 22, 22), (20, 20, 20), (18, 18, 18), (16, 16, 16), (14, 14, 14)],
                label="preconditioning",
                return_minfreq=True,
                masses=getattr(ctx, "masses", None),
            )
        # geom_1 on: stability + cross-validation gate
        elif g >= 1:
            with omp_threads(16):
                success, minw = run_dos_phonons(
                grids=[(22, 22, 22), (20, 20, 20), (18, 18, 18), (16, 16, 16), (14, 14, 14)],
                label="preconditioning",
                return_minfreq=True,
                masses=getattr(ctx, "masses", None),
            )
            if not success:
                die("Preconditioning failed: phonon_dispersion_relations did not produce outfile.dispersion_relations.hdf5")

            # Feed this geom's stability into the next geom's decision about
            # whether the first-order structure update is safe to apply.
            last_minw = minw

            r2_2, r2_3 = parse_cv_r2("extract_forceconstants.log")
            if _precond_gate_passed(minw, r2_2, r2_3, disp_thr):
                prev_geom_dir = os.getcwd()
                break

        prev_geom_dir = os.getcwd()
        os.chdir(precond_root)

        if g == len(schedule) - 1:
            die("Preconditioning failed: unstable phonons after full schedule")

    return prev_geom_dir
