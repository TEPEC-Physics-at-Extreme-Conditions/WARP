# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Outer variable-cell relaxation loop of WARP.

Per iteration ``iter_<n>`` the loop relaxes the internal coordinates in ``position_<p>``
steps (sample, fit force constants with the first-order structure update, check the residual
force FC1) until both the mean and the maximum residual force are below ``ctx.fc1_mean_tol``
and ``ctx.fc1_max_tol``, then takes one lattice step in ``lattice_`` (sample, fit,
``ifc_pressure``, phonon stress ``sigma_ph``, symmetry-preserving lattice update). The run
stops when the stress residual, the change of the phonon stress since the previous lattice
step, and the previous lattice step's relative volume change (< 1 %) are all within
tolerance. Position steps are checked with :func:`warpt.validity.first_missing_stage` and
skipped when complete; a lattice step is skipped only when ``final_structure.vasp`` or
``outfile.new_vc_ucposcar`` already exists, otherwise it is rerun (finished configurations
and calculator outputs are reused by the sampling step). This is how a killed run resumes;
the plan from :mod:`warpt.restart_check` is only reported.

Reads ``ctx.root``, ``ctx.num_samples``, ``ctx.sc_opts``, ``ctx.metric``,
``ctx.reuse_threshold``, ``ctx.fc1_mean_tol``, ``ctx.fc1_max_tol``, ``ctx.sigma_mode``,
``ctx.sigma_ext``, ``ctx.press_conv_th``, ``ctx.press_conv_noise_k``, ``ctx.fc3_gate_thr``,
``ctx.fc3_gate_max_retries``, ``ctx.fc3_gate_nscale`` and ``ctx.ucp``. Writes, per lattice
step, ``outfile.new_vc_ucposcar``, ``outfile.sigma_ph`` and, with
the gate on, ``outfile.fc3_gate`` and, for every attempt that failed the gate and was not
restored, ``fc3_gate_attempt_<k>/``; at the run root ``vc_restart.ucposcar``; at convergence
``final_structure.vasp`` in the last ``lattice_`` directory. Launches ``ifc_pressure`` and, in ``_ensure_from_stage``, ``phonon_dispersion_relations`` directly
and the sampling, fitting and phonon tools through the imported modules.

Public functions: ``refine_new_lattice`` (not called; kept for reference), ``relative_volume_change``, ``sigma_ph_converged``,
``prev_update_small``, ``max_abs_fc1``, ``fc1_stats``, ``stress_converged``, ``run_vc_relax``.
"""

import os
import shutil
import numpy as np
from .utilities import make_and_enter, die, safe_run, omp_threads
from .tdep_sampling import do_sampling_and_run
from .tdep_collect import collect_stats, combine_samples
from .tdep_extract_fc import run_extract_fc
from .compliance_update import compliance_update, write_updated_structure
from .metrics import print_volumetric_expansion, compare_structures
from .io_workflow import copy_previous
from .validity import first_missing_stage, valid_phonons
from .phonon_stress import run_dos_phonons, internal_stress_voigt, fc3_gate_verdict
from .timing import stage, summary


def _symmetry_ops(path, symprec=1e-5):
    """Canonical set of symmetry operations of a POSCAR, expressed in its own cell basis.

    Each operation is ``(rotation, translation mod 1)``. Translations are reduced
    modulo the lattice and rounded, with the 0.99999 -> 0.0 wrap handled explicitly:
    without that, numerically identical operations compare as different and the
    caller sees a spurious symmetry "loss".

    Returns ``None`` if the structure cannot be read or spglib finds nothing, so
    callers can skip the check rather than crash.

    The returned set depends on both the basis and the origin, so it is only comparable
    between structures that share a cell and an origin; see the guard in
    :func:`refine_new_lattice`.
    """
    try:
        import spglib
        from .io_poscar import read_ucposcar_minimal_vasp5
        lattice, pos, types, _meta = read_ucposcar_minimal_vasp5(path)
        ds = spglib.get_symmetry_dataset((lattice, pos, types), symprec=symprec)
        if ds is None:
            return None
        rots = ds.rotations if hasattr(ds, "rotations") else ds["rotations"]
        trans = ds.translations if hasattr(ds, "translations") else ds["translations"]
    except Exception:
        return None

    out = set()
    for r, t in zip(rots, trans):
        tc = np.round(np.mod(np.asarray(t, float), 1.0), 4) % 1.0
        tc = np.where(np.isclose(tc, 1.0, atol=1e-3), 0.0, tc)
        out.add((tuple(np.asarray(r, int).flatten()), tuple(np.round(tc, 4))))
    return out


# ---------------------------------------------------------------------------
# FC3-gate archive helpers (ctx.fc3_gate_thr): a lattice-step attempt that fails the gate
# is moved in full (samples, calculator outputs, fits, stresses) into
# fc3_gate_attempt_<k>/ so the step restarts from an empty directory; the attempt with the
# smallest mismatch can be moved back.
# ---------------------------------------------------------------------------
_FC3_GATE_KEEP = ("fc3_gate_attempt_",)


def _fc3_gate_archive(attempt: int, cwd: str = ".") -> str:
    """Move the content of ``cwd`` into ``fc3_gate_attempt_<attempt + 1>/``.

    Every entry except existing ``fc3_gate_attempt_*`` directories is moved. Prints one
    ``[fc3-gate]`` line and returns the archive path.
    """
    dest = os.path.join(cwd, f"fc3_gate_attempt_{attempt + 1}")
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(cwd):
        if name.startswith(_FC3_GATE_KEEP):
            continue
        shutil.move(os.path.join(cwd, name), os.path.join(dest, name))
    print(f"[fc3-gate] archived attempt {attempt + 1} -> {dest}", flush=True)
    return dest


def _fc3_gate_restore(attempt: int, cwd: str = ".") -> None:
    """Replace the content of ``cwd`` with ``fc3_gate_attempt_<attempt + 1>/``.

    Deletes every entry of ``cwd`` except ``fc3_gate_attempt_*`` directories, moves the
    archive's entries back and removes the emptied archive directory. Exits via ``die`` if
    the archive is missing.
    """
    src = os.path.join(cwd, f"fc3_gate_attempt_{attempt + 1}")
    if not os.path.isdir(src):
        die(f"[fc3-gate] cannot restore attempt {attempt + 1}: {src} missing")
    for name in os.listdir(cwd):
        if name.startswith(_FC3_GATE_KEEP):
            continue
        p = os.path.join(cwd, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for name in os.listdir(src):
        shutil.move(os.path.join(src, name), os.path.join(cwd, name))
    os.rmdir(src)
    print(f"[fc3-gate] restored attempt {attempt + 1} from {src}", flush=True)

@stage("symmetrise lattice")
def refine_new_lattice(ctx, path="outfile.new_vc_ucposcar"):
    """Restore exact space-group symmetry to the lattice-updated structure in place.

    No longer called by the relaxation loop; the two call sites in this module are
    commented out and the function is kept for reference. The lattice update applies a
    strain projected onto the point group of the current cell
    (:func:`warpt.compliance_update.project_strain_onto_point_group`), which keeps the
    space group exactly, so this refinement left the updated cell unchanged (changes of
    order 1e-15 Å in a P2_1/m test cell). It also does not restore a symmetry that has
    been broken: on a cell whose symmetry was broken at the 1e-8 level it returned a cell
    of lower symmetry, which the guard below turns into a stop.

    Runs ``refine_structure -uc <path> -pf <ctx.ucp>`` (TDEP), which writes
    ``outfile.refined_cell`` using ``ctx.ucp`` as the symmetry prototype. If the tool exits 0
    and that file is non-empty, the symmetry operations of ``path`` and of
    ``outfile.refined_cell`` are compared (:func:`_symmetry_ops`, when both sets can be
    determined): the run is aborted via ``die`` if the original set is not a subset of the
    refined set, a ``[refine] SYMMETRY INCREASED`` line is printed if operations were
    gained, and ``outfile.refined_cell`` is copied over ``path``. Otherwise a
    ``[refine][warn]`` line is printed and ``path`` is left unchanged. Returns without
    action if ``path`` does not exist.

    Parameters
    ----------
    ctx : context_state.WarpContext
        ``ctx.ucp``, the initial structure used as the symmetry prototype.
    path : str
        Structure to refine in place (default ``outfile.new_vc_ucposcar``).
    """
    if not os.path.exists(path):
        return
    ops_before = _symmetry_ops(path)
    rc, _, _ = safe_run(f"refine_structure -uc {path} -pf {ctx.ucp}")
    if rc == 0 and os.path.exists("outfile.refined_cell") and os.path.getsize("outfile.refined_cell") > 0:
        ops_after = _symmetry_ops("outfile.refined_cell")

        # One-sided guard: the operations of the input must be a subset of those of the refined
        # cell, so symmetry may be gained (transition to a supergroup) but never lost. A count
        # comparison would miss a change into a different group of equal order. The subset test
        # requires the same cell and origin on both sides, which refine_structure preserves; an
        # origin shift changes the translation parts of the operations of centrosymmetric groups
        # and would make the test fail spuriously.
        if ops_before is not None and ops_after is not None:
            if not ops_before <= ops_after:
                die(f"[refine] refine_structure DESTROYED symmetry in {path}: "
                    f"{len(ops_before)} operations in, {len(ops_after)} out, and the "
                    f"original set is not a subset of the refined one. Refusing to "
                    f"continue with a structure of lower symmetry.")
            gained = len(ops_after) - len(ops_before)
            if gained > 0:
                print(f"[refine] SYMMETRY INCREASED: {len(ops_before)} -> {len(ops_after)} "
                      f"operations", flush=True)

        shutil.copy("outfile.refined_cell", path)
        print(f"[refine] symmetry-refined {path} (preserves c/a & V, restores exact space group)", flush=True)
    else:
        print(f"[refine][warn] refine_structure failed (rc={rc}); proceeding with the "
              f"unrefined {path} (next extract may hit a TDEP symmetry error)", flush=True)


def relative_volume_change(old_uc="infile.ucposcar", new_uc="outfile.new_vc_ucposcar"):
    """Relative volume change between two POSCAR files.

    Parameters
    ----------
    old_uc, new_uc : str
        POSCAR paths (default ``infile.ucposcar`` and ``outfile.new_vc_ucposcar``).

    Returns
    -------
    float
        ``|V_new - V_old| / V_old`` (dimensionless).

    Raises
    ------
    RuntimeError
        If ``V_old`` is not positive.
    """
    import numpy as np
    from .io_poscar import read_ucposcar_minimal_vasp5

    lat_old, _, _, _ = read_ucposcar_minimal_vasp5(old_uc)
    lat_new, _, _, _ = read_ucposcar_minimal_vasp5(new_uc)

    V_old = abs(np.dot(lat_old[0], np.cross(lat_old[1], lat_old[2])))
    V_new = abs(np.dot(lat_new[0], np.cross(lat_new[1], lat_new[2])))

    if V_old <= 0:
        raise RuntimeError("Non-positive old volume in relative_volume_change")

    return abs(V_new - V_old) / V_old

def sigma_ph_converged(ctx, sigma_ph, sigma_ph_prev):
    """Element-wise convergence check of the phonon stress between consecutive lattice steps.

    Parameters
    ----------
    ctx : context_state.WarpContext
        ``press_conv_th`` (GPa) and ``press_conv_noise_k``.
    sigma_ph, sigma_ph_prev : array-like, shape (6,) or None
        Phonon stress ``sigma_ph`` of this and of the previous lattice step, Voigt, GPa.

    Returns
    -------
    tuple[bool or None, str]
        ``(None, msg)`` if ``sigma_ph_prev`` is None. Otherwise ``ok`` is True when every
        component of ``|sigma_ph - sigma_ph_prev|`` is below its threshold: ``press_conv_th``,
        or, when ``press_conv_noise_k > 0`` and ``infile.stat`` is readable, per component
        ``max(press_conv_th, press_conv_noise_k * SE_i)`` with ``SE_i`` from
        :func:`warpt.noise_gain.stress_standard_error`. ``msg`` lists the component
        differences in GPa.
    """
    import numpy as np

    thr = ctx.press_conv_th
    # Same per-component noise floor as the stress gate (ctx.press_conv_noise_k,
    # --pct_noise_floor): threshold_i = max(press_conv_th, k * SE_i). Off when k is 0.
    thr_vec = None
    k_noise = float(getattr(ctx, "press_conv_noise_k", 0.0) or 0.0)
    if k_noise > 0.0:
        from .noise_gain import stress_standard_error
        se6, _n = stress_standard_error("infile.stat")
        if se6 is not None:
            thr_vec = np.maximum(thr, k_noise * se6)
    sigma_ph = np.asarray(sigma_ph, dtype=float)

    def _fmt(vec):
        """Format a vector as a bracketed, 4-decimal list for logging."""
        return "[" + ", ".join(f"{x:.4f}" for x in vec) + "]"

    if sigma_ph_prev is None:
        return None, "phonon elementwise |Δσ_ph| = N/A (no previous σ_ph)"

    sigma_ph_prev = np.asarray(sigma_ph_prev, dtype=float)
    diff_vec = np.abs(sigma_ph - sigma_ph_prev)
    diff = diff_vec.max()

    return (
        bool(np.all(diff_vec < thr_vec[:len(diff_vec)])) if thr_vec is not None else bool(diff < thr),
        f"phonon elementwise |Δσ_ph| = {_fmt(diff_vec)} GPa; max = {diff:.4f} GPa"
    )


def _ensure_from_stage(ctx, state, stage: str, kind: str):
    """Run the stages that :func:`warpt.validity.first_missing_stage` reports missing in the
    current directory until it reports none.

    ``stage`` is accepted and ignored. ``kind`` selects what "complete" means:
    ``"preconditioning"`` requires the DOS dispersion, ``"lattice"`` requires the Grüneisen
    grid and ``outfile.new_vc_ucposcar``, ``"position"`` requires neither; ``ifc_pressure``
    output is required for every kind. For ``"lattice"`` the phonons branch calls
    :func:`warpt.phonon_stress.internal_stress_voigt`, which deletes
    ``outfile.grid_dispersions.hdf5`` after use, so ``valid_gruneisen()`` stays False and the
    loop repeats that branch. Exits via ``die`` when setup files are missing, when a
    preconditioning directory still needs sampling, when at the extract stage the geom index
    cannot be read from a preconditioning directory's name, when ``ifc_pressure`` fails, when
    no q-grid from 30^3 down to 14^3 yields a valid DOS dispersion, when the phonons stage is
    requested for a kind other than ``"preconditioning"`` or ``"lattice"``, or when a stage
    name is unknown. Not called by :func:`run_vc_relax`, which resumes by its forward walk,
    nor by any other module.
    """
    _ = stage  # accepted and ignored

    def _geom_index_from_cwd():
        """Integer suffix of the current ``geom_N`` directory, or ``None``."""
        base = os.path.basename(os.getcwd())
        if base.startswith("geom_"):
            try:
                return int(base.split("_")[-1])
            except Exception:
                return None
        return None

    want_dos = (kind == "preconditioning")
    want_gruneisen = (kind == "lattice")
    want_lattice = (kind == "lattice")

    geom_idx = _geom_index_from_cwd() if kind == "preconditioning" else None

    while True:
        miss = first_missing_stage(
            ctx,
            want_dos=want_dos,
            want_gruneisen=want_gruneisen,
            want_lattice=want_lattice,
        )

        if miss is None:
            return

        if miss == "setup":
            die("setup missing: the infiles of this directory must be recreated (copy_previous/prepare_infiles_from_uc)")

        elif miss in ("sampling", "scf"):
            # Preconditioning geoms take their sample count from the schedule, which is not
            # available here; position and lattice steps use ctx.num_samples.
            if kind == "preconditioning":
                die("Preconditioning resume before extract is not supported; delete the incomplete geom directory and rerun")
            do_sampling_and_run(ctx, state, ctx.num_samples)

        elif miss == "collect":
            collect_stats(ctx, state)

        elif miss == "extract":
            if kind == "preconditioning":
                if geom_idx is None:
                    die("Could not determine geom index for preconditioning resume")

                if geom_idx == 0:
                    with omp_threads(1): 
                        run_extract_fc(ctx, state, mode="second_third")
                else:
                    with omp_threads(1): 
                        run_extract_fc(ctx, state, mode="second_third_first")
            else:
                with omp_threads(1): 
                   run_extract_fc(ctx, state, mode="second_third_first")

        elif miss == "ifc_pressure":
            rc, _, _ = safe_run("ifc_pressure")
            if rc != 0:
                die("ifc_pressure failed")

        elif miss == "phonons":
            if kind == "preconditioning":
                success = False
                grids = [
                    (30, 30, 30),
                    (26, 26, 26),
                    (22, 22, 22),
                    (20, 20, 20),
                    (18, 18, 18),
                    (16, 16, 16),
                    (14, 14, 14),
                ]

                for qx, qy, qz in grids:
                    # remove the previous grid's outputs so valid_phonons() only sees this
                    # grid's file
                    for f in [
                        "outfile.dispersion_relations.hdf5",
                        "outfile.dispersion_relations",
                        "outfile.dispersion_relations.gnuplot",
                    ]:
                        if os.path.exists(f):
                            os.remove(f)

                    cmd = f"phonon_dispersion_relations --qpoint_grid {qx} {qy} {qz} --dos"
                    print(f"[restart][preconditioning] trying phonons with grid {qx} {qy} {qz}")
                    with omp_threads(16): 
                        rc, _, _ =  safe_run(cmd)

                    if rc == 0 and valid_phonons():
                        success = True
                        break

                if not success:
                    die("Preconditioning failed: phonon_dispersion_relations did not produce a valid outfile.dispersion_relations.hdf5")

            elif kind == "lattice":
                # internal_stress_voigt runs phonon_dispersion_relations --dumpgrid --gruneisen,
                # evaluates sigma_ph and deletes outfile.grid_dispersions.hdf5 again, so
                # valid_gruneisen() stays False after this call.
                internal_stress_voigt(ctx, state)

            else:
                die(f"Unexpected phonons stage for kind={kind}")

        elif miss == "lattice":
            with omp_threads(16):
                sigma_res, sigma_ph, sigma_int = internal_stress_voigt(ctx, state)
            new_lattice, poscar_lines, p_int = compliance_update(
                ctx, state, sigma_int, ctx.sigma_ext
            )
            write_updated_structure(new_lattice, poscar_lines)
            # refine_new_lattice(ctx)  # disabled; see refine_new_lattice

        else:
            die(f"Unknown stage: {miss}")


def prev_update_small(ctx, prev_dir: str) -> tuple[float, bool]:
    """Size of the structure update recorded in ``prev_dir``.

    Compares ``prev_dir/infile.ucposcar`` with ``prev_dir/outfile.new_ucposcar`` using
    ``ctx.metric`` and ``ctx.reuse_threshold`` (:func:`warpt.metrics.compare_structures`).

    Returns
    -------
    tuple[float, bool]
        ``(distance, small)`` with ``small`` True when the distance is below
        ``ctx.reuse_threshold``; ``(0.0, True)`` if either file is missing.
    """
    in_uc = os.path.join(prev_dir, "infile.ucposcar")
    out_uc = os.path.join(prev_dir, "outfile.new_ucposcar")

    if (not os.path.exists(in_uc)) or (not os.path.exists(out_uc)):
        return 0.0, True

    rmsd, is_large = compare_structures(
        in_uc,
        out_uc,
        metric=ctx.metric,
        threshold=ctx.reuse_threshold,
    )
    return float(rmsd), (not bool(is_large))


def max_abs_fc1(path="outfile.forceconstant_firstorder"):
    """Largest absolute component of TDEP's first-order force constants (residual force, eV/Å).

    ``outfile.forceconstant_firstorder`` is one ``#`` header line followed by one
    ``fx fy fz`` row per atom. Non-numeric lines are skipped.

    Parameters
    ----------
    path : str
        File to read (default ``outfile.forceconstant_firstorder``).

    Returns
    -------
    float
        ``max |component|`` in eV/Å; 0.0 if the file does not exist.
    """
    if not os.path.exists(path):
        return 0.0

    maxv = 0.0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(x) for x in s.split()]
            except ValueError:
                continue
            for v in vals:
                if abs(v) > maxv:
                    maxv = abs(v)

    return maxv



def fc1_stats(path="outfile.forceconstant_firstorder"):
    """Mean and maximum per-atom residual force from TDEP's first-order force constants.

    Each atom's residual is the magnitude of its ``(fx, fy, fz)`` row in
    ``outfile.forceconstant_firstorder``; rows without exactly three numbers are skipped. The
    mean is the position loop's binding criterion (``ctx.fc1_mean_tol``); the maximum guards
    against a single stuck atom (``ctx.fc1_max_tol``).

    Parameters
    ----------
    path : str
        File to read (default ``outfile.forceconstant_firstorder``).

    Returns
    -------
    tuple[float, float]
        ``(mean, max)`` in eV/Å; ``(0.0, 0.0)`` if the file is absent or has no data rows.
    """
    if not os.path.exists(path):
        return 0.0, 0.0

    mags = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(x) for x in s.split()]
            except ValueError:
                continue
            if len(vals) == 3:
                mags.append((vals[0] ** 2 + vals[1] ** 2 + vals[2] ** 2) ** 0.5)

    if not mags:
        return 0.0, 0.0
    return float(sum(mags) / len(mags)), float(max(mags))


def stress_converged(ctx, sigma_int, p_int=None):
    """Element-wise convergence check of the internal stress against ``ctx.sigma_ext``.

    The target is ``sigma_ext`` in ``scalar`` mode (diagonal ``sigma_ext[0]``, shears
    ``sigma_ext[3:6]`` or zero) and in ``full6`` mode, and ``sigma_ext[:3]`` against
    ``sigma_int[:3]`` in ``diag3`` mode. Component ``i`` passes when
    ``|sigma_int_i - target_i|`` is below ``ctx.press_conv_th`` (GPa) or, when
    ``ctx.press_conv_noise_k > 0`` and ``infile.stat`` is readable, below
    ``max(press_conv_th, press_conv_noise_k * SE_i)`` with ``SE_i`` from
    :func:`warpt.noise_gain.stress_standard_error` (the per-component gate is printed as a
    ``[convergence]`` line).

    Parameters
    ----------
    ctx : context_state.WarpContext
        ``sigma_mode``, ``sigma_ext``, ``press_conv_th``, ``press_conv_noise_k``.
    sigma_int : array-like, shape (6,)
        Internal stress, Voigt, tension-positive, GPa.
    p_int : float or None
        Accepted and ignored.

    Returns
    -------
    tuple[bool, str]
        ``(all compared components pass, message listing the per-component differences in
        GPa)``.
    """

    thr = ctx.press_conv_th
    # Per-component noise floor (ctx.press_conv_noise_k, --pct_noise_floor; warpt/noise_gain.py).
    # Component i passes when |dsigma_i| < max(press_conv_th, k * SE_i). The three shears share
    # their largest SE; the diagonal components keep their own SE, so shear noise does not
    # loosen the diagonal gate.
    thr_vec = None
    k_noise = float(getattr(ctx, "press_conv_noise_k", 0.0) or 0.0)
    if k_noise > 0.0:
        from .noise_gain import stress_standard_error
        se6, nstat = stress_standard_error("infile.stat")
        if se6 is not None:
            thr_vec = np.maximum(thr, k_noise * se6)
            print(f"[convergence] pressure gate per component = max(pct {thr:.4f}, {k_noise:g} x SE) = "
                  f"[{', '.join(f'{x:.4f}' for x in thr_vec)}] GPa (SE from {nstat} samples)")

    def _pass(diff_vec):
        """All components below their threshold (press_conv_th, or the per-component noise floor)."""
        if thr_vec is None:
            return bool(np.max(diff_vec) < thr)
        return bool(np.all(np.asarray(diff_vec) < thr_vec[:len(diff_vec)]))
    sigma_int = np.asarray(sigma_int, dtype=float)

    def _fmt(vec):
        """Format a vector as a bracketed, 4-decimal list for logging."""
        return "[" + ", ".join(f"{x:.4f}" for x in vec) + "]"

    if ctx.sigma_mode == "scalar":
        # Hydrostatic target sigma_ext[0] on the diagonal; shear targets from sigma_ext[3:6]
        # when present, otherwise zero.
        sigma_target = np.zeros(6, dtype=float)
        sigma_target[:3] = float(ctx.sigma_ext[0])
        if len(ctx.sigma_ext) >= 6:
            sigma_target[3:] = np.asarray(ctx.sigma_ext[3:6], dtype=float)

        diff_vec = np.abs(sigma_int - sigma_target)
        diff = diff_vec.max()

        return (
            _pass(diff_vec),
            f"scalar elementwise |Δσ| = {_fmt(diff_vec)} GPa; max = {diff:.4f} GPa"
        )

    elif ctx.sigma_mode == "diag3":
        sigma_target = np.asarray(ctx.sigma_ext[:3], dtype=float)
        diff_vec = np.abs(sigma_int[:3] - sigma_target)
        diff = diff_vec.max()

        return (
            _pass(diff_vec),
            f"diag elementwise |Δσ| = {_fmt(diff_vec)} GPa; max = {diff:.4f} GPa"
        )

    else:
        sigma_target = np.asarray(ctx.sigma_ext, dtype=float)
        diff_vec = np.abs(sigma_int - sigma_target)
        diff = diff_vec.max()

        return (
            _pass(diff_vec),
            f"full elementwise |Δσ| = {_fmt(diff_vec)} GPa; max = {diff:.4f} GPa"
        )

def _write_trajectory(ctx) -> None:
    """Run :func:`warpt.get_frames.collect` with the current working directory as root.

    ``collect`` scans ``iter_0/preconditioning/geom_*``, ``iter_*/position_*`` and
    ``iter_*/lattice_*`` below that root and writes ``<root>/trajectory.xyz`` when at least
    one frame is found, after which one ``[postproc]`` line is printed; with no step
    directory below the root nothing is written and nothing is printed. At the call site in
    :func:`run_vc_relax` the working directory is the converged ``lattice_`` directory, which
    holds no such subdirectory. Every exception is caught and reported as
    ``[postproc][warn]``. ``ctx`` is unused. The standalone tool,
    ``python3 -m warpt.get_frames -r <run root>``, must be run on the run root (``ctx.root``).
    """
    try:
        from pathlib import Path
        from .get_frames import collect
        root = Path(os.getcwd())
        n = collect(root, root / "trajectory.xyz", copy_frames=False, quiet=True)
        if n:
            print(f"[postproc] trajectory.xyz written with {n} frames", flush=True)
    except Exception as exc:                      # noqa: BLE001 - never fatal
        print(f"[postproc][warn] trajectory.xyz not written: {exc}", flush=True)


def run_vc_relax(ctx, state, resume_plan=None):
    """Drive the variable-cell relaxation for one structure from ``ctx.root``.

    Iteration 0 runs :func:`warpt.workflow_preconditioning.run_preconditioning`; every
    iteration then runs position steps ``iter_<n>/position_<p>`` (sample ``ctx.num_samples``
    configurations, combine with the previous step's samples when its update was below
    ``ctx.reuse_threshold`` and the lattice did not just change, fit FC2/FC3 with the
    first-order structure update, run the DOS dispersion) until the mean and max FC1 are
    below ``ctx.fc1_mean_tol`` / ``ctx.fc1_max_tol``, then the lattice step
    ``iter_<n>/lattice_`` (sample, combine with the last position step's samples when its
    update was small and the lattice did not just change, fit, ``ifc_pressure``, phonon
    stress ``sigma_ph``, optional FC3 gate, convergence report, symmetry-preserving lattice
    update). Converged when the stress gate, the change of ``sigma_ph`` since the
    previous lattice step, and the previous lattice step's ``|dV/V| < 0.01`` all pass; then
    ``final_structure.vasp`` is written and the function returns.

    Position directories are checked with :func:`warpt.validity.first_missing_stage` and
    skipped when complete; a lattice directory is skipped only when ``final_structure.vasp``
    (converged run, the function returns) or ``outfile.new_vc_ucposcar`` exists (its
    ``|dV/V|`` and ``outfile.sigma_ph`` are re-read), otherwise it is recomputed. A restarted
    run therefore walks forward from iteration 0 over the existing tree.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Run configuration.
    state : context_state.WarpState
        Mutable run state (``lattice_changed``, ``last_update_significant``,
        ``fc3_gate_events``; ``last_lattice_rel_change`` and ``prev_sigma_ph`` are added
        here).
    resume_plan : dict or None
        Output of :func:`warpt.restart_check.find_resume_plan`; only reported in a
        ``[restart]`` line.

    Side effects
    ------------
    Creates the ``iter_<n>`` tree; writes ``outfile.sigma_ph``, ``outfile.new_vc_ucposcar``,
    ``outfile.fc3_gate`` (gate on),
    ``fc3_gate_attempt_<k>/`` (on a failed FC3-gate attempt) and ``final_structure.vasp`` in
    lattice directories and ``vc_restart.ucposcar`` in ``ctx.root``; prints the timing
    summary at the start of every iteration after the first. Exits via ``die`` when
    ``ifc_pressure`` fails or when the FC3-gate
    attempt to be restored is missing.
    """
    # Restart = idempotent forward execution: every geom, position step and lattice step
    # checks whether it is already complete and skips the expensive work (SCF-level resume
    # inside a step keeps finished calculator outputs). The walk always starts at iteration 0;
    # finished iterations reduce to file-existence checks.
    if resume_plan and resume_plan.get("dir"):
        print(f"[restart] incomplete {resume_plan.get('kind')} detected at "
              f"{resume_plan.get('dir')}; resuming via idempotent forward walk", flush=True)

    iteration = 0

    # Cross-iteration attributes that WarpState does not declare.
    if not hasattr(state, "last_lattice_rel_change"):
        state.last_lattice_rel_change = None

    if not hasattr(state, "prev_sigma_ph"):
        state.prev_sigma_ph = None

    while True:
        print(f"\n=== TDEP_vc-relax Iteration {iteration} ===")
        # timing summary at the start of every iteration after the first, so a run killed by
        # the scheduler still leaves one
        if iteration:
            summary(f"through iteration {iteration - 1}")

        iter_root = os.path.join(ctx.root, f"iter_{iteration}")
        make_and_enter(iter_root)
        state.current_dir = iter_root

        # ==================================
        # PRECONDITIONING (ONLY iteration 0) -- run_preconditioning is idempotent
        # ==================================
        if iteration == 0:
            print("\n=== Preconditioning ===")
            from .workflow_preconditioning import run_preconditioning
            last_geom = run_preconditioning(ctx, state)
        else:
            prev_root = os.path.join(ctx.root, f"iter_{iteration - 1}")
            last_geom = os.path.join(prev_root, "lattice_")

        # =========================
        # POSITION LOOP
        # =========================
        position = 0

        while True:
            print(f"\n{'=' * 30} Position update {position} {'=' * 30}")
            position_dir = os.path.join(iter_root, f"position_{position}")
            os.makedirs(position_dir, exist_ok=True)
            os.chdir(position_dir)
            state.current_dir = position_dir

            if first_missing_stage(ctx, want_dos=False, want_gruneisen=False,
                                   want_lattice=False, want_ifc=False) is None:
                # Already complete (resume): sampling, SCF and extract are not repeated.
                # Position steps never run ifc_pressure, so want_ifc=False.
                fcmax = max_abs_fc1()
                print(f"[position][resume] position_{position} already complete; max(|FC1|) = {fcmax:.6f}")
            else:
                copy_previous(last_geom, ctx.sc_opts)

                do_sampling_and_run(ctx, state, ctx.num_samples)
                collect_stats(ctx, state)

                # reuse decision based on previous step (last_geom)
                rmsd, small = prev_update_small(ctx, last_geom)

                # fresh samples exactly once after a lattice update
                if state.lattice_changed:
                    reuse_prev = False
                    state.lattice_changed = False
                    print(f"[samples] prev RMSD={rmsd:.6g} thr={ctx.reuse_threshold:.6g} -> fresh (lattice changed)")
                else:
                    reuse_prev = small
                    print(f"[samples] prev RMSD={rmsd:.6g} thr={ctx.reuse_threshold:.6g} -> {'reuse' if reuse_prev else 'fresh'}")

                if reuse_prev:
                    combine_samples(last_geom, position_dir)

                # fit on (possibly) combined dataset
                with omp_threads(1): run_extract_fc(ctx, state, mode="second_third_first")

                # record whether this step's own update was significant (step-local)
                rmsd_step, step_small = prev_update_small(ctx, position_dir)
                state.last_update_significant = (not step_small)

                with omp_threads(16):
                    ok_ph = run_dos_phonons(label="phonons", stop_on_valid=True,
                                            masses=getattr(ctx, "masses", None))
                if not ok_ph:
                    print("[position][warn] phonon check failed, continuing based on FC1 only")

                fcmax = max_abs_fc1()
                print(f"[position] max(|FC1|) = {fcmax:.6f}")

            # Position-loop exit: both the mean and the max per-atom residual force |FC1| must
            # be below ctx.fc1_mean_tol / ctx.fc1_max_tol (eV/A). The residual cannot be
            # converged below the force error of the calculator, so the tolerances are chosen
            # relative to it (see --fc1_mean_tol).
            fc1_mean, fc1_max = fc1_stats()
            print(f"[position] FC1 residual: mean = {fc1_mean * 1000:.2f} meV/A "
                  f"(tol {ctx.fc1_mean_tol * 1000:.2f}), "
                  f"max = {fc1_max * 1000:.2f} meV/A "
                  f"(tol {ctx.fc1_max_tol * 1000:.2f})")
            if fc1_mean < ctx.fc1_mean_tol and fc1_max < ctx.fc1_max_tol:
                print(f"[position] converged: mean and max FC1 below tolerance "
                      f"after {position + 1} step(s)")
                break

            last_geom = position_dir
            position += 1

        # =========================
        # LATTICE STEP
        # =========================
        print(f"\n{'=' * 30} lattice update {'=' * 30}")

        vc_dir = os.path.join(iter_root, "lattice_")
        os.makedirs(vc_dir, exist_ok=True)
        os.chdir(vc_dir)
        state.current_dir = vc_dir

        # --- resume: a finished lattice step leaves a marker file ---
        if os.path.exists("final_structure.vasp"):
            print("[lattice][resume] iteration already converged here.")
            return
        if os.path.exists("outfile.new_vc_ucposcar"):
            # The lattice update was already produced (resume). Reconstruct the
            # cheap cross-iteration state from files and advance without redoing
            # the SCFs / FC fit / stress.
            state.last_lattice_rel_change = relative_volume_change(
                "infile.ucposcar", "outfile.new_vc_ucposcar")
            shutil.copy("outfile.new_vc_ucposcar",
                        os.path.join(ctx.root, "vc_restart.ucposcar"))
            state.lattice_changed = True
            # recover this iteration's σ_ph so the next iteration's Δσ_ph check is exact
            if os.path.exists("outfile.sigma_ph"):
                try:
                    state.prev_sigma_ph = np.loadtxt("outfile.sigma_ph")
                except Exception:
                    pass
            print(f"[lattice][resume] lattice update already present; "
                  f"|ΔV/V| = {state.last_lattice_rel_change:.6%}; advancing to next iteration")
            iteration += 1
            continue

        copy_previous(position_dir, ctx.sc_opts)

        reuse_prev_vc = (not state.last_update_significant) and (not state.lattice_changed)

        # Lattice-step sampling + fit + stress, wrapped in the optional FC3/sigma_ph
        # consistency gate (ctx.fc3_gate_thr; off when None, then max_retries = 0 and the loop
        # runs once). Attempt k samples round(num_samples * nscale**k) configurations. A failed
        # attempt is moved to fc3_gate_attempt_<k+1>/ and the step is rebuilt from the position
        # step's structure and force constants; after the last retry the attempt with the smallest
        # mismatch is restored (its stress recomputed) or, if that is the current one, kept. On a
        # restart the attempt counter starts at 0 and existing archives are ignored. The gate is
        # not applied in preconditioning.
        gate_thr = getattr(ctx, "fc3_gate_thr", None)
        max_retries = int(getattr(ctx, "fc3_gate_max_retries", 2)) if gate_thr is not None else 0
        nscale = float(getattr(ctx, "fc3_gate_nscale", 1.0))
        attempts = []          # (attempt index, n_samples, mismatch)
        attempt = 0
        while True:
            ns_k = max(1, int(round(ctx.num_samples * (nscale ** attempt))))
            if attempt > 0:
                print(f"[fc3-gate] resampling the lattice stage: attempt {attempt + 1}/{max_retries + 1}, "
                      f"n = {ns_k}", flush=True)
            do_sampling_and_run(ctx, state, ns_k)
            collect_stats(ctx, state)

            print(f"[samples] position to lattice-step reuse -> {'yes' if reuse_prev_vc else 'no'}")
            if reuse_prev_vc:
                combine_samples(position_dir, vc_dir)

            with omp_threads(1): run_extract_fc(ctx, state, mode="second_third_first")

            rc, _, _ = safe_run("ifc_pressure")
            if rc != 0:
                die("ifc_pressure failed")

            with omp_threads(16):
                sigma_res, sigma_ph, sigma_int = internal_stress_voigt(ctx, state)

            gate_ok, mism = fc3_gate_verdict(ctx, sigma_ph, attempt, max_retries)
            attempts.append((attempt, ns_k, float(mism)))
            state.fc3_gate_events.append((iteration, attempt, ns_k, float(mism)))
            if gate_ok:
                break
            if attempt >= max_retries:
                best = min(attempts, key=lambda t: t[2])
                if best[0] != attempt:
                    print(f"[fc3-gate] retries exhausted; restoring attempt {best[0] + 1} "
                          f"(mismatch {best[2]:.3f} GPa) as the least-bad estimate", flush=True)
                    _fc3_gate_archive(attempt)
                    _fc3_gate_restore(best[0])
                    with omp_threads(16):
                        sigma_res, sigma_ph, sigma_int = internal_stress_voigt(ctx, state)
                else:
                    print(f"[fc3-gate] retries exhausted; proceeding with the last attempt "
                          f"(mismatch {mism:.3f} GPa)", flush=True)
                print(f"[fc3-gate][WARNING] lattice step taken with |σ_FC3+σ_ph| above the threshold "
                      f"({gate_thr} GPa)", flush=True)
                break
            _fc3_gate_archive(attempt)
            copy_previous(position_dir, ctx.sc_opts)
            attempt += 1
        if gate_thr is not None:
            with open("outfile.fc3_gate", "w") as fh:
                fh.write("# attempt  n_samples  max|sigma_FC3+sigma_ph| (GPa)\n")
                for a, n, m in attempts:
                    fh.write(f"{a + 1:4d} {n:8d} {m:12.4f}\n")

        # 1) pressure/stress convergence (always evaluated)
        p_ok, p_msg = stress_converged(ctx, sigma_int)

        # 2) phonon-stress delta (always evaluated/reported if previous exists)
        ph_ok_raw, ph_msg = sigma_ph_converged(ctx, sigma_ph, state.prev_sigma_ph)

        # ph_sigma acts as a convergence gate only once the pressure criterion holds
        ph_gate_ok = p_ok and (ph_ok_raw is not None) and bool(ph_ok_raw)

        # 3) volume guard only matters at the end:
        # only evaluate it as a convergence gate if pressure + ph_sigma are both OK
        if ph_gate_ok:
            last_rel = state.last_lattice_rel_change
            vol_ok = (last_rel is not None) and (last_rel < 0.01)
            vol_msg = f"last actual lattice step |ΔV/V| = {last_rel:.6%}"
        else:
            vol_ok = False
            vol_msg = "volume guard skipped (requires pressure + ph_sigma convergence)"

        # unified convergence report
        print("[convergence]")
        print(f"  pressure : {p_msg} -> {'OK' if p_ok else 'NOT OK'}")

        if ph_ok_raw is None:
            print(f"  ph_sigma : {ph_msg} -> N/A")
        else:
            print(f"  ph_sigma : {ph_msg} -> {'OK' if ph_ok_raw else 'NOT OK'}")

        if ph_gate_ok:
            print(f"  volume   : {vol_msg} -> {'OK' if vol_ok else 'NOT OK'}")
        else:
            print(f"  volume   : {vol_msg}")

        # store the current sigma_ph for the comparison in the next lattice iteration,
        # and persist it so a cold restart can reconstruct the Δσ_ph check.
        state.prev_sigma_ph = np.array(sigma_ph, copy=True)
        np.savetxt("outfile.sigma_ph", np.asarray(sigma_ph, dtype=float))

        if p_ok and ph_gate_ok and vol_ok:
            # Current structure is the converged one:
            # stress was evaluated on infile.ucposcar
            shutil.copy("infile.ucposcar", "final_structure.vasp")
            print("Converged.")
            _write_trajectory(ctx)
            return

        # Not converged -> propose a new lattice update
        new_lattice, poscar_lines, p_int = compliance_update(
            ctx,
            state,
            sigma_int,
            ctx.sigma_ext,
        )

        write_updated_structure(new_lattice, poscar_lines)
        # refine_new_lattice(ctx)  # disabled; see refine_new_lattice

        print_volumetric_expansion("infile.ucposcar", "outfile.new_vc_ucposcar")

        # Record the size of this lattice step for the next convergence check
        state.last_lattice_rel_change = relative_volume_change(
            "infile.ucposcar",
            "outfile.new_vc_ucposcar",
        )
        print(
            f"[volume-guard] recorded last lattice step |ΔV/V| = "
            f"{state.last_lattice_rel_change:.6%}"
        )

        root_handoff = os.path.join(ctx.root, "vc_restart.ucposcar")
        shutil.copy("outfile.new_vc_ucposcar", root_handoff)

        print(f"[lattice] Restart prepared: {root_handoff}")
        state.lattice_changed = True
        iteration += 1
