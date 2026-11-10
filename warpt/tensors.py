# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Voigt <-> 3x3 tensor conversions for the stress/strain path.

The Voigt ordering used throughout WARP (and matching TDEP's
``outfile.stress_*`` and ``outfile.compliance_tensor``) is

    index 0..5  =  xx, yy, zz, yz, xz, xy.

Two conventions are used:

* **Stress** is stored with *un-doubled* shear components — :func:`to_voigt`
  symmetrises but does not double the off-diagonals (``sigma_6 = sigma_xy``).
* **Strain** in Voigt form carries *engineering* shear (``eps_4 = 2*eps_yz``);
  :func:`voigt_to_strain` therefore halves components 4-6 to recover the
  symmetric tensor strain. This pairs with TDEP's compliance, which maps
  un-doubled stress to engineering strain.

Public functions: ``to_voigt``, ``voigt_to_strain``, ``voigt_trace``.
"""
import numpy as np


def to_voigt(mat: np.ndarray) -> np.ndarray:
    """Contract a symmetric 3x3 tensor to its 6-component Voigt vector.

    Off-diagonals are symmetrised but **not** doubled, i.e. the result is the
    stress-style Voigt vector ``[xx, yy, zz, yz, xz, xy]``.

    Parameters
    ----------
    mat : numpy.ndarray
        Array with trailing shape ``(3, 3)`` (broadcasting over leading axes
        is supported).

    Returns
    -------
    numpy.ndarray
        Array with the trailing ``(3, 3)`` replaced by a length-6 Voigt axis.
    """
    t = np.asarray(mat)
    return np.stack([
        t[..., 0, 0],
        t[..., 1, 1],
        t[..., 2, 2],
        0.5 * (t[..., 1, 2] + t[..., 2, 1]),
        0.5 * (t[..., 0, 2] + t[..., 2, 0]),
        0.5 * (t[..., 0, 1] + t[..., 1, 0]),
    ], axis=-1)


def voigt_to_strain(voigt: np.ndarray) -> np.ndarray:
    """Expand a Voigt strain vector to the symmetric 3x3 strain tensor.

    The input is interpreted as *engineering* strain (components 4-6 are
    ``2*eps_ij``), so the shear entries are halved on the way back to the
    tensor form.

    Parameters
    ----------
    voigt : numpy.ndarray, shape (6,)
        Engineering strain in Voigt order ``[xx, yy, zz, yz, xz, xy]``.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        The symmetric tensor strain.
    """
    v = np.asarray(voigt)
    return np.array([
        [v[0], v[5]/2, v[4]/2],
        [v[5]/2, v[1], v[3]/2],
        [v[4]/2, v[3]/2, v[2]],
    ], dtype=float)


def voigt_trace(sig6: np.ndarray) -> float:
    """Tensor trace of a Voigt vector, i.e. ``xx + yy + zz``.

    Parameters
    ----------
    sig6 : numpy.ndarray, shape (6,)
        A Voigt vector (stress or strain).

    Returns
    -------
    float
        The sum of the three normal components.
    """
    s = np.asarray(sig6)
    return float(s[0] + s[1] + s[2])