# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Elastic-compliance lattice update inside the lattice step of the variable-cell relaxation.

Given the internal stress ``sigma_int`` and the target external stress ``sigma_ext`` (Voigt
order xx, yy, zz, yz, xz, xy; tension-positive; GPa) the step is

    eps   = -lambda_l * S (sigma_int - sigma_ext)
    A_new = A (I + eps)          (rows of A are the lattice vectors)

with ``S`` the compliance tensor and ``lambda_l`` the damping: ``ctx.lattice_damping``
(default 1.0, the full elastic step), multiplied by the factor of an optional ladder
(``ctx.lattice_damping_ladder``) and by an optional noise-referenced gain
(``ctx.lattice_noise_z0`` > 0, :mod:`warpt.noise_gain`). Before it is applied, ``eps`` is
projected onto the strains invariant under the point group of the current cell.

``S`` is, by default, ``outfile.compliance_tensor`` (1/GPa), the inverse of the clamped-ion
elastic constants TDEP derives from the fitted second-order force constants. With
``ctx.elastic_constants`` (``--elastic_constants FILE``) it is instead the inverse of the 6x6
stiffness read from that file (GPa, Voigt order xx, yy, zz, yz, xz, xy, in the Cartesian
frame of ``infile.ucposcar``; ``#`` comment lines allowed), the same in every lattice step.

Reads ``infile.ucposcar``, ``outfile.compliance_tensor`` (or the elastic-constants file) and,
when the noise gain is on, ``infile.stat``; writes ``outfile.new_vc_ucposcar`` (through
``write_updated_structure``).

Public functions: ``read_elastic_constants``, ``project_strain_onto_point_group``,
``compliance_update``, ``write_updated_structure``.
"""

import numpy as np

from .tensors import voigt_to_strain, voigt_trace
from .io_poscar import read_poscar, write_poscar_with_new_lattice
from .utilities import die
from .timing import stage


def _read_compliance_tensor(filename):
    """Read a 6x6 Voigt compliance matrix (1/GPa), skipping ``#`` comment lines."""
    with open(filename, 'r') as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]
    return np.array([[float(x) for x in line.split()] for line in lines])


def read_elastic_constants(filename, sym_tol=1e-3):
    """Read and check a user-supplied 6x6 stiffness tensor (GPa) for the lattice update.

    The file holds six rows of six numbers, Voigt order xx, yy, zz, yz, xz, xy, in the
    Cartesian frame of ``infile.ucposcar``; blank lines and lines starting with ``#`` are
    skipped. The matrix must be 6x6, symmetric (largest ``|C - C^T|`` relative to the largest
    ``|C|`` below ``sym_tol``; the symmetrised half-sum is returned) and positive definite,
    so that its inverse is a usable compliance. Any violation raises ValueError with the
    reason; the orchestrator turns that into a fatal error before the run starts.

    Parameters
    ----------
    filename : str
        Path of the elastic-constants file.
    sym_tol : float
        Relative asymmetry allowed (default 1e-3).

    Returns
    -------
    C : numpy.ndarray, shape (6, 6)
        Symmetrised stiffness in GPa.
    """
    try:
        C = _read_compliance_tensor(filename)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{filename}: cannot read a numeric 6x6 table ({exc})") from exc
    if C.shape != (6, 6):
        raise ValueError(f"{filename}: expected a 6x6 matrix, got {C.shape[0]}x{C.shape[1] if C.ndim == 2 else '?'}")
    scale = float(np.max(np.abs(C)))
    if not np.isfinite(C).all() or scale == 0.0:
        raise ValueError(f"{filename}: the matrix is zero or not finite")
    asym = float(np.max(np.abs(C - C.T))) / scale
    if asym > sym_tol:
        raise ValueError(f"{filename}: the matrix is not symmetric (relative asymmetry {asym:.2e} > {sym_tol:g})")
    C = 0.5 * (C + C.T)
    eig = np.linalg.eigvalsh(C)
    if eig.min() <= 0.0:
        raise ValueError(f"{filename}: the matrix is not positive definite (smallest eigenvalue {eig.min():.4g} GPa)")
    return C


def project_strain_onto_point_group(strain_tensor, ucposcar="infile.ucposcar", symprec=1e-5):
    """Project a Cartesian strain tensor onto the totally symmetric representation of the point
    group of a unit cell.

    The strain a crystal can undergo without lowering its symmetry is the subspace invariant
    under every point-group operation R (Neumann's principle): ``eps_sym = (1/|G|) sum_R R
    eps R^T``, the group average. Applied before the lattice update it keeps, to rounding, the
    space group of the current cell; ``||eps - eps_sym||`` is the symmetry-forbidden part,
    i.e. the sampling noise of the stress residual. Rows of ``lattice`` are the lattice
    vectors, so a Cartesian column vector is ``x_c = A^T x_f``; spglib's rotations act on
    fractional vectors, hence ``R_c = A^T R_f (A^T)^-1``. A supergroup's invariant subspace
    lies inside the subgroup's, so symmetry raising is not blocked.

    Parameters
    ----------
    strain_tensor : array-like, shape (3, 3)
        Cartesian strain (dimensionless).
    ucposcar : str
        POSCAR whose point group is used (default ``infile.ucposcar``).
    symprec : float
        spglib symmetry tolerance in Å (default 1e-5).

    Returns
    -------
    eps_sym : numpy.ndarray, shape (3, 3)
        Symmetrised projected strain; the unprojected input if spglib is unavailable or fails.
    forbidden : float
        Frobenius norm of the projected-out part; ``nan`` when no projection was done.
    nops : int
        Number of spglib symmetry operations averaged (rotation parts of the space-group
        operations of ``ucposcar``, repeated once per centring translation of a centred
        cell); 0 when no projection was done (a ``[lattice][warn]`` line is printed).
    """
    try:
        import spglib
        from .io_poscar import read_ucposcar_minimal_vasp5
        lattice, frac, types, _meta = read_ucposcar_minimal_vasp5(ucposcar)
        _map = {t: i + 1 for i, t in enumerate(dict.fromkeys(types))}   # species -> small integer label
        numbers = [_map[t] for t in types]
        sym = spglib.get_symmetry((np.asarray(lattice, float), np.asarray(frac, float), numbers), symprec=symprec)
        if sym is None:
            raise RuntimeError("spglib returned no symmetry")
        rots = np.asarray(sym["rotations"], float)
    except Exception as exc:  # a failed projection must not stop the run: fall back to the unprojected strain
        print(f"[lattice][warn] strain projection skipped ({exc}); using the unprojected strain")
        return np.asarray(strain_tensor, float), float("nan"), 0
    A_T = np.asarray(lattice, float).T
    A_T_inv = np.linalg.inv(A_T)
    eps = np.asarray(strain_tensor, float)
    acc = np.zeros((3, 3))
    for R_f in rots:
        R_c = A_T @ R_f @ A_T_inv
        acc += R_c @ eps @ R_c.T
    eps_sym = acc / len(rots)
    eps_sym = 0.5 * (eps_sym + eps_sym.T)
    forbidden = float(np.linalg.norm(eps - eps_sym))
    return eps_sym, forbidden, int(len(rots))


@stage("lattice update")
def compliance_update(ctx, state, sigma_int_voigt, sigma_ext_voigt):
    """Compute the strained lattice from the compliance tensor and the stress residual.

    Reads ``infile.ucposcar`` and the compliance ``S`` (6x6 Voigt, 1/GPa): the inverse of the
    stiffness in ``ctx.elastic_constants`` when that is set (:func:`read_elastic_constants`),
    otherwise ``outfile.compliance_tensor``. Forms ``residual = sigma_int - sigma_ext`` and the
    damping ``lambda_l``:

    * ``lambda_l = ctx.lattice_damping`` (default 1.0);
    * if ``ctx.lattice_damping_ladder`` (``'thr:f,thr:f'``, GPa -> factor) is set,
      ``lambda_l`` becomes ``f * ctx.lattice_damping`` for the smallest ``thr`` exceeding
      ``|dsigma|``, where ``|dsigma|`` is ``max_i |residual_i|`` over all six components
      (``scalar``/``full6``) or the three diagonal ones (``diag3``);
    * if ``ctx.lattice_noise_z0 > 0``, ``lambda_l`` is multiplied by the gain
      ``z**2/(z**2 + z0**2)`` from :func:`warpt.noise_gain.noise_gain`, with the standard
      errors read from ``infile.stat`` (skipped with a message when that file is missing or
      unreadable, or has fewer than 3 rows or 13 columns).

    The engineering Voigt strain ``-lambda_l * S @ residual`` is converted to a tensor
    (shears halved), projected onto the point-group-invariant strains of ``infile.ucposcar``
    (:func:`project_strain_onto_point_group`) and applied as ``A_new = A (I + eps)``;
    components of ``A_new`` below 1e-6 Å in magnitude are set to zero. Prints the damping,
    the projection summary and the old and new lattice.

    Parameters
    ----------
    ctx : context_state.WarpContext
        ``sigma_mode``, ``lattice_damping``, ``lattice_damping_ladder``, ``lattice_noise_z0``,
        ``elastic_constants``.
    state : context_state.WarpState
        Unused.
    sigma_int_voigt, sigma_ext_voigt : numpy.ndarray, shape (6,)
        Internal and external stress, Voigt order, tension-positive, GPa.

    Returns
    -------
    new_lattice : numpy.ndarray, shape (3, 3)
        Updated lattice vectors (rows), Å.
    poscar_lines : list[str]
        The lines of ``infile.ucposcar`` for :func:`write_updated_structure`.
    p_int : float
        ``-(xx + yy + zz)/3`` of ``sigma_int_voigt`` (GPa, compression-positive).
    """

    src_poscar = "infile.ucposcar"
    lattice, poscar_lines = read_poscar(src_poscar)

    # The compliance S: a user-supplied stiffness (--elastic_constants, inverted here) or the
    # clamped-ion one TDEP's ifc_pressure derives from the fitted FC2.
    ec_file = getattr(ctx, "elastic_constants", None)
    if ec_file:
        try:
            C_user = read_elastic_constants(ec_file)
        except ValueError as exc:
            die(f"--elastic_constants: {exc}")
        compliance = np.linalg.inv(C_user)
        diag = " ".join(f"{c:.1f}" for c in np.diag(C_user))
        print(f"[lattice] compliance from --elastic_constants {ec_file} (diag C11..C66 = {diag} GPa), "
              f"outfile.compliance_tensor not used")
    else:
        compliance = _read_compliance_tensor("outfile.compliance_tensor")

    # stress residual driving the step (Voigt, tension-positive, GPa)
    stress_voigt = sigma_int_voigt - sigma_ext_voigt

    # hydrostatic pressure (GPa, compression positive)
    p_int = -voigt_trace(sigma_int_voigt) / 3.0

    # --- |dsigma|: largest compared residual component (damping ladder input, logged) ---
    if ctx.sigma_mode == "scalar":
        # sigma_ext holds the full hydrostatic target [-P, -P, -P, 0, 0, 0], so the element-wise
        # residual includes the deviatoric part, not only the mean pressure.
        diff = float(np.max(np.abs(sigma_int_voigt - sigma_ext_voigt)))

    elif ctx.sigma_mode == "diag3":
        diff = float(np.max(np.abs(sigma_int_voigt[:3] - sigma_ext_voigt[:3])))

    else:  # full6
        diff = float(np.max(np.abs(sigma_int_voigt - sigma_ext_voigt)))

    # lambda_l = ctx.lattice_damping (--lattice_damping, default 1.0 = the full elastic step
    # -S (sigma_int - sigma_ext); values above 1 over-relax). It is multiplied only by the
    # factor of an explicit --lattice_damping_ladder and by the noise gain below.
    base = float(getattr(ctx, "lattice_damping", 1.0))
    damping = base
    ladder = getattr(ctx, "lattice_damping_ladder", None)
    if ladder:
        # ladder 'thr:f,thr:f' (GPa -> factor), checked from the largest threshold down, so the
        # smallest threshold above |dsigma| sets the factor
        rungs = sorted(((float(t), float(f)) for t, f in (r.split(":") for r in str(ladder).split(",") if r.strip())), reverse=True)
        for thr, fac in rungs:
            if diff < thr:
                damping = fac * base
        print(f"[lattice] damping ladder {ladder} -> damping={damping:.3f} (|Δσ|={diff:.4f} GPa)")
    print(f"[lattice] damping lambda_l={damping:.3f} (|Δσ|={diff:.4f} GPa){'' if ladder else ' (no ladder: full step scaled by lambda_l)'}")

    # --- noise-referenced gain (warpt/noise_gain.py); off unless ctx.lattice_noise_z0 > 0
    # (--lattice_noise_gain) ---
    z0 = float(getattr(ctx, "lattice_noise_z0", 0.0) or 0.0)
    if z0 > 0.0:
        from .noise_gain import stress_standard_error, noise_gain
        se6, nstat = stress_standard_error("infile.stat")
        if se6 is None:
            print(f"[lattice] noise gain requested (z0={z0}) but infile.stat is unreadable here ({nstat} rows) -> damping kept at {damping:.3f}")
        else:
            gain, z, imax = noise_gain(stress_voigt, se6, z0)
            damping *= gain
            comp = ("xx", "yy", "zz", "shear", "shear", "shear")[imax]
            print(f"[lattice] noise gain: z = |Δσ|/SE = {z:.2f} on {comp} (|Δσ|={abs(float(stress_voigt[imax])):.4f} GPa, "
                  f"SE={se6[imax]:.4f} GPa from {nstat} samples, z0={z0:g}) -> gain {gain:.3f}, lambda_l={damping:.3f}")

    strain_voigt = (-1.0 * compliance @ stress_voigt) * damping
    strain_tensor = voigt_to_strain(strain_voigt)

    # Project onto the point-group-invariant strains before moving the cell (see
    # project_strain_onto_point_group): the cell cannot lose symmetry to stress noise, and the
    # removed component is a direct measure of that noise.
    strain_tensor, forbidden, nops = project_strain_onto_point_group(strain_tensor, src_poscar)
    if nops:
        print(f"[lattice] strain projected onto the invariant subspace of {nops} point-group operations; "
              f"symmetry-forbidden component removed = {forbidden:.3e}")

    # Rows of `lattice` are the lattice vectors; a symmetric Cartesian strain acts on each row
    # as a_i -> a_i (I + eps), i.e. A_new = A (I + eps). (I + eps) @ A would instead mix the
    # lattice vectors; the two agree only when A and eps commute.
    new_lattice = lattice @ (np.eye(3) + strain_tensor)
    new_lattice[np.abs(new_lattice) < 1e-6] = 0.0

    print("\n=== LATTICE UPDATE ===")
    print("OLD lattice:")
    print(lattice)

    print("NEW lattice:")
    print(new_lattice)
    print("======================\n")

    return new_lattice, poscar_lines, p_int


def write_updated_structure(new_lattice, poscar_lines, out_path="outfile.new_vc_ucposcar"):
    """Write the strained lattice into a POSCAR, keeping the species/coords block.

    Parameters
    ----------
    new_lattice : numpy.ndarray, shape (3, 3)
        Updated lattice vectors (rows), Å.
    poscar_lines : list[str]
        Original POSCAR lines (from :func:`compliance_update`).
    out_path : str
        Output path (default ``outfile.new_vc_ucposcar``).

    Returns
    -------
    str
        ``out_path``.
    """
    write_poscar_with_new_lattice(out_path, poscar_lines, new_lattice)
    return out_path
