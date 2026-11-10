# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Structure-comparison metrics between two POSCAR files.

Distance measures between two POSCARs (fractional/Cartesian RMSD, maximum displacement,
pairwise-distance-spectrum difference), each returning ``(value, value >= threshold)``, plus
two volume-change helpers. :func:`compare_structures` dispatches on a metric name; WARP calls
it from ``workflow_vc_relax.prev_update_small`` with ``--metric`` and ``--reuse_threshold`` to
decide whether a step's first-order structure update was small enough to reuse the previous
samples.

Reads POSCARs via :mod:`warpt.io_poscar`; writes nothing.

Public functions: ``fractional_rmsd``, ``max_component_frac``, ``cartesian_rmsd``,
``max_atom_disp``, ``pairwise_spectrum_diff_frac``, ``compare_structures``,
``volumetric_expansion_ratio``, ``print_volumetric_expansion``.
"""
import numpy as np

from .io_poscar import (
    read_poscar_frac,
    wrap_to_pm_half,
    read_poscar,
    compute_volume,
)


def fractional_rmsd(file1: str, file2: str, threshold_frac: float = 0.01):
    """Minimal-image RMSD of fractional coordinates (dimensionless).

    ``file1``, ``file2``: POSCAR paths with the same atom order. Returns
    ``(rmsd, rmsd >= threshold_frac)``.
    """
    f1, _, _ = read_poscar_frac(file1)
    f2, _, _ = read_poscar_frac(file2)
    d = wrap_to_pm_half(f2 - f1)
    rmsd = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    return rmsd, (rmsd >= threshold_frac)


def max_component_frac(file1: str, file2: str, threshold_frac: float = 0.02):
    """Largest absolute minimal-image fractional displacement component over all atoms
    (dimensionless).

    Returns ``(value, value >= threshold_frac)``.
    """
    f1, _, _ = read_poscar_frac(file1)
    f2, _, _ = read_poscar_frac(file2)
    d = wrap_to_pm_half(f2 - f1)
    m = float(np.max(np.abs(d)))
    return m, (m >= threshold_frac)


def cartesian_rmsd(file1: str, file2: str, threshold_A: float = 0.01):
    """Minimal-image Cartesian RMSD in Å, using the lattice of ``file1``.

    Returns ``(rmsd, rmsd >= threshold_A)``.
    """
    f1, A, _ = read_poscar_frac(file1)
    f2, _, _ = read_poscar_frac(file2)
    d_cart = wrap_to_pm_half(f2 - f1) @ A
    rmsd = float(np.sqrt((d_cart ** 2).sum(axis=1).mean()))
    return rmsd, (rmsd >= threshold_A)


def max_atom_disp(file1: str, file2: str, threshold_A: float = 0.05):
    """Largest minimal-image per-atom displacement in Å, using the lattice of ``file1``.

    Returns ``(value, value >= threshold_A)``.
    """
    f1, A, _ = read_poscar_frac(file1)
    f2, _, _ = read_poscar_frac(file2)
    d_cart = wrap_to_pm_half(f2 - f1) @ A
    maxd = float(np.linalg.norm(d_cart, axis=1).max())
    return maxd, (maxd >= threshold_A)


def _pairwise_frac_distances(frac: np.ndarray) -> np.ndarray:
    """Sorted spectrum of pairwise minimal-image distances in fractional L2 (dimensionless)."""
    n = frac.shape[0]
    ds = []
    for i in range(n):
        df = wrap_to_pm_half(frac[i + 1:] - frac[i])
        ds.append(np.linalg.norm(df, axis=1))
    return np.sort(np.concatenate(ds)) if ds else np.array([])


def pairwise_spectrum_diff_frac(file1: str, file2: str, threshold_frac: float = 1e-3):
    """RMS difference between the sorted pairwise minimal-image fractional distance spectra.

    Returns ``(diff, diff >= threshold_frac)``. Raises ValueError when the two files have
    different atom counts.
    """
    f1, _, _ = read_poscar_frac(file1)
    f2, _, _ = read_poscar_frac(file2)
    d1 = _pairwise_frac_distances(f1)
    d2 = _pairwise_frac_distances(f2)
    if d1.shape != d2.shape:
        raise ValueError("Mismatch in pairwise spectrum size.")
    diff = float(np.sqrt(np.mean((d1 - d2) ** 2)))
    return diff, (diff >= threshold_frac)


def compare_structures(file1: str, file2: str, metric: str = "frac_rmsd", threshold: float = 0.01):
    """Dispatch to one of the five metric functions of this module by name.

    ``metric`` is one of ``frac_rmsd``, ``max_comp_frac``, ``pair_spectrum`` (``threshold``
    dimensionless) or ``cart_rmsd``, ``max_atom`` (``threshold`` in Å). Returns that metric's
    ``(value, value >= threshold)``. Raises ValueError for an unknown name.
    """
    if metric == "frac_rmsd":
        return fractional_rmsd(file1, file2, threshold_frac=threshold)
    if metric == "max_comp_frac":
        return max_component_frac(file1, file2, threshold_frac=threshold)
    if metric == "pair_spectrum":
        return pairwise_spectrum_diff_frac(file1, file2, threshold_frac=threshold)
    if metric == "cart_rmsd":
        return cartesian_rmsd(file1, file2, threshold_A=threshold)
    if metric == "max_atom":
        return max_atom_disp(file1, file2, threshold_A=threshold)
    raise ValueError(f"Unknown metric: {metric}")


def volumetric_expansion_ratio(poscar_initial: str, poscar_final: str) -> float:
    """Relative volume change ``(V_final - V_initial) / V_initial`` (dimensionless) between the
    lattices of two POSCAR files (volumes in Å³, POSCAR scale applied).
    """
    lat_i, _ = read_poscar(poscar_initial)
    lat_f, _ = read_poscar(poscar_final)
    v_i = compute_volume(lat_i)
    v_f = compute_volume(lat_f)
    return float((v_f - v_i) / v_i)


def print_volumetric_expansion(poscar_initial: str, poscar_final: str) -> float:
    """Print both cell volumes (Å³) and their relative change, and return the relative change
    ``(V_final - V_initial) / V_initial`` (dimensionless).
    """
    lat_i, _ = read_poscar(poscar_initial)
    lat_f, _ = read_poscar(poscar_final)
    v_i = compute_volume(lat_i)
    v_f = compute_volume(lat_f)
    alpha_v = float((v_f - v_i) / v_i)

    print(f"Initial volume: {v_i:.6f} Å³")
    print(f"Final volume:   {v_f:.6f} Å³")
    print(f"Volumetric expansion ratio: {alpha_v:.6%}")

    return alpha_v