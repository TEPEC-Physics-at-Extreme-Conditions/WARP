# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Noise-referenced lattice damping gain and stress noise estimate.

The per-configuration stresses of a sampled ensemble are the rows of ``infile.stat``; the
standard error of their mean,

    SE_i = sd_i / sqrt(n)          (i = xx, yy, zz and the three shears; GPa)

is the resolution of the stress estimate at that step. The residual expressed in units of
SE,

    z = max_i |sigma_int,i - sigma_ext,i| / SE_i ,

is used in two places:

1. Lattice damping gain (``--lattice_noise_gain Z0``, ``ctx.lattice_noise_z0``):
   :func:`warpt.compliance_update.compliance_update` multiplies ``lambda_l`` by
   ``z**2 / (z**2 + Z0**2)``, so the full elastic step is taken far from convergence and the
   step shrinks once the residual is within a few SE.

2. Stress-gate floor (``--pct_noise_floor K``, ``ctx.press_conv_noise_k``):
   :func:`warpt.workflow_vc_relax.stress_converged` and
   :func:`warpt.workflow_vc_relax.sigma_ph_converged` raise the gate of component ``i`` to
   ``max(pct, K * SE_i)``.

Both are off when the corresponding option is 0 (the default). The three shear standard
errors are folded into one (their maximum), so the result does not depend on the shear
order of ``infile.stat`` relative to WARP's Voigt vector. No gain above 1 is produced.

Reads ``infile.stat`` (columns 8-13). Writes nothing.

Public functions: ``stress_standard_error``, ``noise_z``, ``noise_gain``, ``noise_floor``.
"""
import os
import numpy as np


def stress_standard_error(stat_path="infile.stat"):
    """Standard error of the sample-mean stress from a TDEP ``infile.stat``.

    Columns 8-13 (1-based) hold the per-configuration stress in GPa in the order xx, yy, zz,
    then three shears (as written by :func:`warpt.tdep_collect.collect_stats`).

    Parameters
    ----------
    stat_path : str
        File to read (default ``infile.stat``).

    Returns
    -------
    tuple[numpy.ndarray or None, int]
        ``(se6, n)`` with ``se6 = [SE_xx, SE_yy, SE_zz, s, s, s]`` (GPa, ``ddof=1``), ``s``
        the largest shear standard error, and ``n`` the row count. ``(None, 0)`` when the
        file is missing, empty or unreadable; ``(None, rows)`` when it has fewer than 3 rows
        or fewer than 13 columns.
    """
    try:
        if not os.path.exists(stat_path) or os.path.getsize(stat_path) == 0:
            return None, 0
        A = np.loadtxt(stat_path, ndmin=2)
    except Exception:
        return None, 0
    if A.ndim != 2 or A.shape[0] < 3 or A.shape[1] < 13:
        return None, int(A.shape[0]) if A.ndim == 2 else 0
    n = A.shape[0]
    st = A[:, 7:13]
    se = st.std(axis=0, ddof=1) / np.sqrt(n)
    s_shear = float(np.max(se[3:6]))
    se6 = np.array([se[0], se[1], se[2], s_shear, s_shear, s_shear], dtype=float)
    return se6, int(n)


def noise_z(residual_voigt, se6, floor_gpa=1e-6):
    """Largest residual-to-noise ratio over the six Voigt components.

    Components whose standard error is below ``floor_gpa`` (constant columns,
    symmetry-zero shears) use the floor, so a genuinely zero residual gives z = 0
    and a finite residual on a noiseless component gives a large z (full step).

    Parameters
    ----------
    residual_voigt : array-like, shape (6,)
        Stress residual ``sigma_int - sigma_ext``, Voigt order, GPa; only the first six
        components are used.
    se6 : array-like, shape (6,)
        Standard errors of the stress components, GPa (:func:`stress_standard_error`).
    floor_gpa : float
        Lower bound applied to every standard error, GPa (default 1e-6).

    Returns
    -------
    tuple[float, int]
        ``(z, i_max)``: the largest ``|residual_i| / max(se6_i, floor_gpa)`` and the index of
        the component that sets it.
    """
    r = np.abs(np.asarray(residual_voigt, dtype=float)[:6])
    se = np.maximum(np.asarray(se6, dtype=float)[:6], floor_gpa)
    zc = r / se
    i = int(np.argmax(zc))
    return float(zc[i]), i


def noise_gain(residual_voigt, se6, z0):
    """Gain ``z**2 / (z**2 + z0**2)`` in [0, 1]; 1.0 when ``z0 <= 0``.

    Parameters
    ----------
    residual_voigt : array-like, shape (6,)
        Stress residual ``sigma_int - sigma_ext``, Voigt order, GPa.
    se6 : array-like, shape (6,)
        Standard errors of the stress components, GPa (:func:`stress_standard_error`).
    z0 : float
        Residual-to-noise ratio at which the gain is 1/2 (dimensionless;
        ``ctx.lattice_noise_z0``).

    Returns
    -------
    tuple[float, float, int]
        ``(gain, z, i_max)`` with the gain in [0, 1] and ``z``, ``i_max`` from :func:`noise_z`
        (called with its default floor).
    """
    z, i = noise_z(residual_voigt, se6)
    z0 = float(z0)
    if z0 <= 0.0:
        return 1.0, z, i
    g = z * z / (z * z + z0 * z0)
    return float(g), z, i


def noise_floor(pct, se6, k):
    """Convergence threshold raised to ``k * max_i SE_i`` when that exceeds ``pct``.

    No pipeline stage calls this function; the per-component floor of the stress gate is
    formed in :func:`warpt.workflow_vc_relax.stress_converged` and
    :func:`warpt.workflow_vc_relax.sigma_ph_converged`.

    Parameters
    ----------
    pct : float
        Base convergence threshold, GPa.
    se6 : array-like, shape (6,)
        Standard errors of the stress components, GPa; only the first six are used.
    k : float
        Multiplier of the largest standard error (dimensionless).

    Returns
    -------
    tuple[float, float]
        ``(max(pct, floor), floor)`` in GPa, with ``floor = k * max_i se6_i``.
    """
    floor = float(k) * float(np.max(np.asarray(se6, dtype=float)[:6]))
    return (max(float(pct), floor), floor)
