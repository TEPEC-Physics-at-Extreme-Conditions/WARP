# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Phonon (Grüneisen) stress ``sigma_ph``, the total internal stress, and the dispersion runner.

``internal_stress_voigt`` reads the stresses written by ``ifc_pressure``
(``outfile.stress_dft``, ``outfile.stress_fc2``, ``outfile.stress_fc3``; compression-positive),
negates them, runs ``phonon_dispersion_relations --dumpgrid --gruneisen`` and adds the
phonon stress ``sigma_ph`` computed from ``outfile.grid_dispersions.hdf5``:

    sigma_int = -(sigma_DFT + sigma_FC2 + sigma_FC3)_file + sigma_ph     (tension-positive)

``sigma_ph`` is anisotropic only if the HDF5 carries ``gruneisen_tensors``; with the scalar
``gruneisen_parameters`` it is hydrostatic. ``run_dos_phonons`` wraps
``phonon_dispersion_relations --dos`` on a list of q-grids for the stability check and reads
the minimum frequency from ``outfile.dispersion_relations.hdf5``. ``fc3_ph_mismatch`` and
``fc3_gate_verdict`` implement the optional FC3 / ``sigma_ph`` consistency gate
(``ctx.fc3_gate_thr``).

Reads ``ctx.temp`` (K), ``ctx.kB_eV``, ``ctx.hbar_eVs``, ``ctx.EV_TO_GPA_A3``,
``ctx.sigma_ext``, ``ctx.fc3_gate_thr`` and ``ctx.masses`` (non-natural masses: the
dispersion runs read ``infile.isotopes`` through ``--readiso``). Launches ``phonon_dispersion_relations``, which
writes ``outfile.dispersion_relations*`` and ``outfile.grid_dispersions.hdf5``; the latter is
deleted once ``sigma_ph`` is extracted.

Public functions: ``min_frequency``, ``readiso_flag``, ``run_dos_phonons``,
``sigma_ph_from_dispersion_h5``,
``internal_stress_voigt``, ``fc3_ph_mismatch``, ``fc3_gate_verdict``.
"""
import os
import numpy as np
import h5py

from .utilities import die, safe_run
from .io_workflow import write_isotopes
from .tensors import to_voigt
from .validity import valid_phonons

def min_frequency(path):
    """Return the minimum phonon frequency in an HDF5 dispersion file.

    A negative value indicates an imaginary (unstable) mode.

    Parameters
    ----------
    path : str
        Path to a ``*dispersion*.hdf5`` file with a ``frequencies`` dataset.

    Returns
    -------
    float
        The smallest frequency over all q-points and branches, in the file's unit (THz for
        the files written by :func:`run_dos_phonons`, TDEP's default ``--unit thz``).
    """
    import h5py, numpy as np
    with h5py.File(path, "r") as f:
        freqs = f["frequencies"][()]
    return float(np.min(freqs))

def readiso_flag(masses):
    """``" --readiso"`` for ``phonon_dispersion_relations`` when ``masses`` (the ``--masses``
    dict) is set, after making sure ``infile.isotopes`` exists in the current directory
    (written from ``infile.ucposcar`` by :func:`warpt.io_workflow.write_isotopes`); the empty
    string otherwise."""
    if not masses:
        return ""
    if not os.path.exists("infile.isotopes"):
        write_isotopes(masses)
    return " --readiso"


def run_dos_phonons(
    grids=None,
    label="phonons",
    stop_on_valid=False,
    return_minfreq=False,
    masses=None,
):
    """Run ``phonon_dispersion_relations --qpoint_grid qx qy qz --dos`` over a list of q-grids.

    For each grid, in order, the stale ``outfile.dispersion_relations``,
    ``outfile.dispersion_relations.hdf5`` and ``outfile.dispersion_relations.gnuplot`` are
    removed and the tool is run; a grid counts as valid when it exits 0 and the HDF5 has a
    ``frequencies`` dataset (:func:`warpt.validity.valid_phonons`).

    Parameters
    ----------
    grids : list[tuple[int, int, int]] or None
        q-grids to try; ``None`` means ``[(22, 22, 22), (20, 20, 20), (18, 18, 18),
        (16, 16, 16), (14, 14, 14)]``.
    label : str
        Tag of the ``[label] trying phonons with grid ...`` log line.
    stop_on_valid : bool
        Return ``True`` at the first valid grid.
    return_minfreq : bool
        Return ``(True, min_frequency)`` (THz, from the HDF5) at the first valid grid and
        ``(False, None)`` if no grid is valid.
    masses : dict or None
        Non-natural masses (``ctx.masses``); adds ``--readiso`` (see :func:`readiso_flag`).

    Returns
    -------
    bool or tuple[bool, float or None]
        With ``return_minfreq`` the tuple above; otherwise ``True`` only when
        ``stop_on_valid`` is set and a grid is valid. With neither flag set every grid is run
        and ``False`` is returned.
    """
    if grids is None:
        grids = [(22, 22, 22), (20, 20, 20), (18, 18, 18), (16, 16, 16), (14, 14, 14)]

    for qx, qy, qz in grids:
        for f in [
            "outfile.dispersion_relations.hdf5",
            "outfile.dispersion_relations",
            "outfile.dispersion_relations.gnuplot",
        ]:
            if os.path.exists(f):
                os.remove(f)

        cmd = f"phonon_dispersion_relations --qpoint_grid {qx} {qy} {qz} --dos{readiso_flag(masses)}"
        print(f"[{label}] trying phonons with grid {qx} {qy} {qz}")
        rc, _, _ = safe_run(cmd)

        if rc == 0 and valid_phonons():
            if return_minfreq:
                return True, min_frequency("outfile.dispersion_relations.hdf5")
            if stop_on_valid:
                return True

    if return_minfreq:
        return False, None
    return False


def _read_voigt_file(filename):
    """Read the first 6-component Voigt row from a TDEP stress file.

    Returns a zero vector if the file is missing, so a stage that produced no
    stress contributes nothing rather than erroring.
    """
    if not os.path.exists(filename):
        return np.zeros(6)

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) == 6:
                    return np.array([float(x) for x in parts])

    return np.zeros(6)


def _run_phonon_grid_with_gruneisen(ctx):
    """Run ``phonon_dispersion_relations --dumpgrid --gruneisen`` until the grid HDF5 exists.

    Tries q-grids 16^3, 14^3, 12^3 and 10^3 in that order and returns as soon as the tool
    exits 0 and ``outfile.grid_dispersions.hdf5`` is present. ``ctx.masses`` adds
    ``--readiso``. Raises ``RuntimeError`` if no grid succeeds.
    """
    target_file = "outfile.grid_dispersions.hdf5"

    grids = [(16, 16, 16), (14, 14, 14), (12, 12, 12), (10, 10, 10)]

    for qx, qy, qz in grids:
        cmd = (f"phonon_dispersion_relations --dumpgrid --gruneisen --qpoint_grid {qx} {qy} {qz}"
               f"{readiso_flag(getattr(ctx, 'masses', None))}")
        print(f"[σ_ph] Running: {cmd}")

        rc, _, _ = safe_run(cmd)

        if rc == 0 and os.path.exists(target_file):
            return

    raise RuntimeError("Failed to generate grid_dispersions.hdf5")


def sigma_ph_from_dispersion_h5(ctx, path="outfile.grid_dispersions.hdf5"):
    """Phonon stress tensor ``sigma_ph`` (3x3, GPa) from a TDEP grid-dispersion HDF5.

    ``frequencies`` is taken as angular frequency in rad/s (TDEP's grid dump, unit
    'Hz (angular)'), so ``E = ctx.hbar_eVs * frequencies`` is in eV. For every q-point and
    mode with ``E > 1e-8`` eV the mode-Grüneisen parameter (tensor if ``gruneisen_tensors``
    is present, else scalar ``gruneisen_parameters`` times the identity) is weighted by
    ``E * (1/2 + n_BE(E, ctx.temp))``; the sum is divided by the number of q-points and by
    the cell volume from ``lattice_vectors`` and converted with ``ctx.EV_TO_GPA_A3``. The
    result is in the file's compression-positive sense; the caller negates it.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Provides ``temp`` (K), ``kB_eV``, ``hbar_eVs`` and ``EV_TO_GPA_A3``.
    path : str
        Grid-dispersion HDF5 (default ``outfile.grid_dispersions.hdf5``).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Phonon stress in GPa.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        freqs = np.asarray(f["frequencies"][()])

        if "gruneisen_tensors" in f:
            gamma = np.asarray(f["gruneisen_tensors"][()])
            gamma_is_tensor = True
        else:
            gamma = np.asarray(f["gruneisen_parameters"][()])
            gamma_is_tensor = False

        A = np.asarray(f["lattice_vectors"][()])

    # Keep q-point structure: (nq, nmodes)
    if freqs.ndim == 1:
        freqs = freqs.reshape(1, -1)
    else:
        freqs = freqs.reshape(freqs.shape[0], -1)

    nq, nm = freqs.shape

    if gamma_is_tensor:
        gamma = gamma.reshape(nq, nm, 3, 3)
    else:
        gamma = gamma.reshape(nq, nm)

    V = abs(np.dot(A[0], np.cross(A[1], A[2])))

    E = ctx.hbar_eVs * freqs
    good = (E > 1e-8)

    pref = np.zeros_like(E)
    x = E[good] / (ctx.kB_eV * ctx.temp)
    n = 1.0 / np.expm1(x)
    pref[good] = E[good] * (0.5 + n)

    if gamma_is_tensor:
        pV = np.zeros((3, 3))
        for iq in range(nq):
            for im in range(nm):
                if good[iq, im]:
                    pV += gamma[iq, im] * pref[iq, im]
    else:
        p_scalar = 0.0
        for iq in range(nq):
            for im in range(nm):
                if good[iq, im]:
                    p_scalar += gamma[iq, im] * pref[iq, im]
        pV = np.eye(3) * p_scalar

    # mean over q-points; the sum over modes is kept
    pV /= nq

    sigma = (pV * ctx.EV_TO_GPA_A3) / V
    return sigma


def internal_stress_voigt(ctx, state):
    """Assemble the total internal stress of the current directory for the lattice update.

    Reads ``outfile.stress_dft``, ``outfile.stress_fc2`` and ``outfile.stress_fc3`` written by
    ``ifc_pressure`` (compression-positive; a missing file reads as zeros), negates each and
    sums them into ``sigma_res``. If both a second- and a third-order force constant file
    are present (``infile.forceconstant``/``outfile.forceconstant`` and the ``_thirdorder``
    pair) it runs ``phonon_dispersion_relations --dumpgrid --gruneisen`` on q-grids 16^3,
    14^3, 12^3, 10^3 until ``outfile.grid_dispersions.hdf5`` appears, computes ``sigma_ph``
    from it with :func:`sigma_ph_from_dispersion_h5`, negates it to tension-positive and
    deletes the HDF5; otherwise ``sigma_ph`` is zero. Prints a stress diagnostic block.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Temperature, constants and ``sigma_ext`` (printed).
    state : context_state.WarpState
        Unused.

    Returns
    -------
    sigma_res : numpy.ndarray, shape (6,)
        ``-(sigma_DFT + sigma_FC2 + sigma_FC3)`` in file sign, i.e. tension-positive, Voigt,
        GPa.
    sigma_ph : numpy.ndarray, shape (6,)
        Phonon stress, tension-positive, Voigt, GPa.
    sigma_int : numpy.ndarray, shape (6,)
        ``sigma_res + sigma_ph``.

    Raises
    ------
    RuntimeError
        If no q-grid produces ``outfile.grid_dispersions.hdf5``; any exception from the
        phonon-stress evaluation is re-raised after a ``[WARN]`` line.
    """
    print("[debug] cwd =", os.getcwd())

    # sigma_int = sigma_DFT + sigma_FC2 + sigma_FC3 + sigma_ph in file (compression-positive)
    # values, every term with a plus sign:
    #   sigma_FC3 and sigma_ph are the same Psi<uu>r contraction evaluated in real space and
    #   on the q grid (Ladygin, PRM 9, 125001 (2025)); they cancel up to q-point interpolation.
    #   sigma_FC2 = 2<V2>/3V equals the ionic kinetic pressure that the Born-Oppenheimer
    #   stress lacks (virial theorem; the same term as in the SSCHA pressure, Monacelli et
    #   al., PRB 98, 024106 (2018)).
    # WARP works tension-positive, hence the negation on read. A missing file reads as zeros,
    # so without third-order force constants the result is sigma_DFT + sigma_FC2.
    sigma_dft = -_read_voigt_file("outfile.stress_dft")
    sigma_fc2 = -_read_voigt_file("outfile.stress_fc2")
    sigma_fc3 = -_read_voigt_file("outfile.stress_fc3")
    sigma_res = sigma_dft + sigma_fc2 + sigma_fc3

    have_fc2 = os.path.exists("infile.forceconstant") or os.path.exists("outfile.forceconstant")
    have_fc3 = os.path.exists("infile.forceconstant_thirdorder") or os.path.exists("outfile.forceconstant_thirdorder")

    if have_fc2 and have_fc3:
        try:
            _run_phonon_grid_with_gruneisen(ctx)
            sigma_ph_tensor = sigma_ph_from_dispersion_h5(ctx)

            # outfile.grid_dispersions.hdf5 is GB-scale and only needed for sigma_ph; it is
            # regenerated when the lattice step is rerun.
            try:
                if os.path.exists("outfile.grid_dispersions.hdf5"):
                    os.remove("outfile.grid_dispersions.hdf5")
                    print("[cleanup] removed outfile.grid_dispersions.hdf5 (sigma_ph extracted)", flush=True)
            except OSError:
                pass

        except Exception as exc:
            print(f"[WARN] σ_ph failed: {exc}")
            raise

    else:
        print("[info] No FC3 → σ_ph = 0")
        sigma_ph_tensor = np.zeros((3, 3))

    sigma_ph = -to_voigt(sigma_ph_tensor)
    sigma_int = sigma_res + sigma_ph

    # diagnostics
    print("\n=== STRESS DIAGNOSTIC ===")
    print("σ_DFT:", sigma_dft)
    print("σ_FC2:", sigma_fc2, " (+, nuclear kinetic / momenta term)")
    print("σ_FC3:", sigma_fc3, " (+, real-space Grüneisen; ~cancels σ_ph)")
    print("σ_0  :", sigma_res, " (= σ_DFT + σ_FC2 + σ_FC3, all-plus)")
    print("σ_ph :", sigma_ph)
    print("σ_int:", sigma_int)
    print("σ_ext:", ctx.sigma_ext)
    print("=========================\n")

    return sigma_res, sigma_ph, sigma_int


# ---------------------------------------------------------------------------
# Optional FC3 / sigma_ph consistency gate (ctx.fc3_gate_thr, --fc3_gate_thr; off when
# None). The resampling it drives is in workflow_vc_relax.run_vc_relax.
# ---------------------------------------------------------------------------
def fc3_ph_mismatch(sigma_ph):
    """Mismatch between the real-space and q-space third-order stress.

    ``sigma_FC3`` (``outfile.stress_fc3``, negated to tension-positive) and ``sigma_ph`` are
    the same contraction evaluated in real space and on the q grid, so their sum vanishes up
    to q-point interpolation; a large sum indicates an ill-conditioned third-order fit.

    Parameters
    ----------
    sigma_ph : array-like, shape (6,)
        Phonon stress, Voigt, tension-positive, GPa.

    Returns
    -------
    tuple[float, numpy.ndarray]
        ``max_i |sigma_FC3,i + sigma_ph,i|`` in GPa and the six-component sum.
    """
    sigma_fc3 = -_read_voigt_file("outfile.stress_fc3")
    vec = np.asarray(sigma_fc3, dtype=float) + np.asarray(sigma_ph, dtype=float)
    return float(np.max(np.abs(vec))), vec


def fc3_gate_verdict(ctx, sigma_ph, attempt, max_retries):
    """Evaluate the optional FC3 / sigma_ph gate for one lattice-step attempt.

    Parameters
    ----------
    ctx : context_state.WarpContext
        ``fc3_gate_thr`` (GPa, or None for off).
    sigma_ph : array-like, shape (6,)
        Phonon stress, Voigt, tension-positive, GPa.
    attempt, max_retries : int
        Zero-based attempt index and the retry limit; used only in the log line.

    Returns
    -------
    tuple[bool, float]
        ``(ok, mismatch)``: ``ok`` is True when ``fc3_gate_thr`` is None or the mismatch is
        <= the threshold; ``mismatch`` is ``max_i |sigma_FC3,i + sigma_ph,i|`` in GPa. One
        ``[fc3-gate]`` line is printed in either case.
    """
    thr = getattr(ctx, "fc3_gate_thr", None)
    mism, vec = fc3_ph_mismatch(sigma_ph)
    if thr is None:
        print(f"[fc3-gate] |σ_FC3+σ_ph| max = {mism:.3f} GPa (gate off)", flush=True)
        return True, mism
    ok = mism <= float(thr)
    print(f"[fc3-gate] |σ_FC3+σ_ph| max = {mism:.3f} GPa (thr {float(thr):.3f}) "
          f"components {np.round(vec, 3)} -> "
          f"{'OK' if ok else f'RESAMPLE (attempt {attempt + 1}/{max_retries + 1} failed)'}",
          flush=True)
    return ok, mism
