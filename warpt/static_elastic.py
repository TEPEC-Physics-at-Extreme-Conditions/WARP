#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Static elastic constants of a unit cell from a machine-learned potential, for ``--elastic_constants``.

WARP's lattice step is a preconditioned stress update, ``eps = -lambda_l S (sigma_int - sigma_ext)``,
whose compliance ``S`` only sets the size of each step: the converged structure, where
``sigma_int = sigma_ext``, does not depend on it. TDEP's ``outfile.compliance_tensor`` is the
clamped-ion compliance of the fitted force constants (the atoms keep their fractional coordinates
under the strain), which is far too stiff wherever the atoms would relax inside the strained cell,
for example along $c$ in layered crystals. This tool computes the elastic tensor of the potential
itself by stress-strain finite differences on the unit cell, with the internal coordinates relaxed
at every strain (relaxed-ion; ``--clamped`` keeps them fixed), symmetrises it over the point group
of the cell and writes it in the format ``--elastic_constants`` reads.

Procedure: the internal coordinates are first relaxed at the given cell (symmetry kept, so the
reference is the potential's own minimum in that cell; ``--no-relax-start`` skips this); then for
each of the six Voigt strains ``j`` the cell is strained by ``+h`` and ``-h`` (engineering shear),
the atoms are relaxed at fixed cell (BFGS to ``--fmax``), and column ``j`` of ``C`` is the central
difference of the stress. ``C`` is symmetrised, averaged over the point-group rotations of the
reference cell (spglib) and checked to be positive definite.

Command line::

    python3 -m warpt.static_elastic -ucp start.vasp -mle upet -mlp model.ckpt [-mlv HEAD] [-mlc]
        [-mld cpu] [--strain 0.005] [--fmax 2e-4] [--clamped] [--no-relax-start]
        [--no-symmetrize] [-o C_static.dat]

The force-engine options are those of the orchestrator (``-mle``, ``-mlp``, ``-mlv``, ``-mlc``,
``-mld``). Requires ASE and numpy, spglib for the symmetrisation, and the potential's package.

Public functions: ``voigt_strain_tensor``, ``symmetrize_point_group``, ``static_elastic_tensor``,
``write_elastic_constants``, ``main``.
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np

from .constants import EV_TO_GPA_A3

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_strain_tensor(j: int, e: float) -> np.ndarray:
    """The 3x3 strain tensor of engineering Voigt strain ``e`` in component ``j`` (shears halved)."""
    E = np.zeros((3, 3))
    p, q = VOIGT[j]
    if p == q:
        E[p, p] = e
    else:
        E[p, q] = E[q, p] = 0.5 * e
    return E


def _voigt_to_tensor4(C6):
    C = np.zeros((3, 3, 3, 3))
    for i, (a, b) in enumerate(VOIGT):
        for j, (g, l) in enumerate(VOIGT):
            C[a, b, g, l] = C[b, a, g, l] = C[a, b, l, g] = C[b, a, l, g] = C6[i, j]
    return C


def _tensor4_to_voigt(C):
    return np.array([[C[a, b, g, l] for (g, l) in VOIGT] for (a, b) in VOIGT])


def symmetrize_point_group(C6, atoms, symprec=1e-5):
    """Average a 6x6 Voigt stiffness over the point-group rotations of ``atoms``.

    The rotations are spglib's for the cell (fractional), taken to the Cartesian frame as in
    :func:`warpt.compliance_update.project_strain_onto_point_group`. Returns ``(C_sym, nops)``;
    ``nops`` is 0 and ``C6`` is returned unchanged when spglib is missing or fails.
    """
    try:
        import spglib
        sym = spglib.get_symmetry((atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers), symprec=symprec)
        rots = np.asarray(sym["rotations"], float)
    except Exception as exc:  # noqa: BLE001
        print(f"[static_elastic][warn] point-group symmetrisation skipped ({exc})")
        return np.asarray(C6, float), 0
    A_T = np.asarray(atoms.cell[:], float).T
    A_T_inv = np.linalg.inv(A_T)
    C = _voigt_to_tensor4(np.asarray(C6, float))
    acc = np.zeros_like(C)
    for R_f in rots:
        R = A_T @ R_f @ A_T_inv
        acc += np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, C)
    return _tensor4_to_voigt(acc / len(rots)), int(len(rots))


def static_elastic_tensor(atoms, calc, strain=0.005, fmax=2e-4, relaxed=True, relax_start=True,
                          steps=500, verbose=True):
    """Stress-strain elastic tensor (GPa, Voigt xx yy zz yz xz xy) of ``atoms`` with ``calc``.

    Returns ``(C, reference)``: the unsymmetrised 6x6 stiffness (already made symmetric) and the
    reference structure (ASE Atoms, internal coordinates relaxed when ``relax_start``).
    """
    from ase.optimize import BFGS
    ref = atoms.copy()
    ref.calc = calc
    if relax_start:
        from ase.constraints import FixSymmetry
        ref.set_constraint(FixSymmetry(ref))
        BFGS(ref, logfile=None).run(fmax=fmax, steps=steps)
        ref.set_constraint()
    s0 = ref.get_stress(voigt=True) * EV_TO_GPA_A3
    if verbose:
        print("[static_elastic] reference stress (tension-positive, GPa): "
              + " ".join(f"{x:+.3f}" for x in s0))

    def stress_at(j, e):
        b = ref.copy()
        b.calc = calc
        b.set_constraint()
        b.set_cell(ref.cell[:] @ (np.eye(3) + voigt_strain_tensor(j, e)), scale_atoms=True)
        if relaxed:
            BFGS(b, logfile=None).run(fmax=fmax, steps=steps)
        return b.get_stress(voigt=True) * EV_TO_GPA_A3

    C = np.zeros((6, 6))
    for j in range(6):
        C[:, j] = (stress_at(j, strain) - stress_at(j, -strain)) / (2.0 * strain)
        if verbose:
            print(f"[static_elastic] strain {j + 1}/6 done: C_{j + 1}{j + 1} = {C[j, j]:.1f} GPa")
    return 0.5 * (C + C.T), ref


def write_elastic_constants(path, C6, header_lines):
    """Write ``C6`` (GPa) in the ``--elastic_constants`` format with ``#`` header lines."""
    with open(path, "w") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        for row in C6:
            fh.write(" ".join(f"{x:14.6f}" for x in row) + "\n")
    return path


def main(argv=None):
    """Command-line entry point; returns 0, or 1 when the tensor is not positive definite."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-ucp", "--unit_cell", dest="ucp", required=True, help="unit cell (VASP POSCAR)")
    p.add_argument("-mle", "--ml_engine", dest="engine", choices=["upet", "mace"], default="upet")
    p.add_argument("-mlp", "--ml_params", dest="ml_params", required=True, help="model file (.ckpt or .model)")
    p.add_argument("-mlv", "--ml_variant", dest="variant", default=None, help="UPET energy head")
    p.add_argument("-mlc", "--ml_conservative", dest="conservative", action="store_true",
                   help="forces and stress as energy gradients (UPET)")
    p.add_argument("-mld", "--ml_device", dest="device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--strain", type=float, default=0.005, help="finite strain h (default 0.005)")
    p.add_argument("--fmax", type=float, default=2e-4, help="BFGS force tolerance, eV/Å (default 2e-4)")
    p.add_argument("--clamped", action="store_true", help="clamped-ion tensor: no internal relaxation under strain")
    p.add_argument("--no-relax-start", dest="relax_start", action="store_false",
                   help="do not relax the internal coordinates of the given cell first")
    p.add_argument("--no-symmetrize", dest="symmetrize", action="store_false",
                   help="skip the point-group average")
    p.add_argument("-o", "--output", default="C_static.dat", help="file to write (default C_static.dat)")
    a = p.parse_args(argv)

    from ase.io import read
    from .MLIP_interface import make_calc
    from .compliance_update import read_elastic_constants

    calc = make_calc(SimpleNamespace(engine=a.engine, ml_params=a.ml_params, device=a.device,
                                     conservative=a.conservative, rot_avg=0, rot_avg_batch=8,
                                     variant=a.variant, model="pet-mad-s", version=None,
                                     dispersion=False, dtype="float64"))
    atoms = read(a.ucp, format="vasp")
    C, ref = static_elastic_tensor(atoms, calc, strain=a.strain, fmax=a.fmax, relaxed=not a.clamped,
                                   relax_start=a.relax_start)
    nops = 0
    if a.symmetrize:
        C, nops = symmetrize_point_group(C, ref)
    kind = "clamped-ion" if a.clamped else "relaxed-ion"
    np.set_printoptions(precision=1, suppress=True, linewidth=120)
    print(f"[static_elastic] {kind} elastic constants (GPa), Voigt xx yy zz yz xz xy"
          + (f", averaged over {nops} point-group operations" if nops else "") + ":")
    print(C)
    eig = np.linalg.eigvalsh(C)
    cp = ref.cell.cellpar()
    header = [f"{kind} elastic constants (GPa) of {a.ucp} with {a.engine} model {a.ml_params}"
              + (f" head {a.variant}" if a.variant else ""),
              f"stress-strain central differences, h = {a.strain:g}, fmax = {a.fmax:g} eV/A"
              + (", internal coordinates relaxed at the start" if a.relax_start else "")
              + (f", point-group average over {nops} operations" if nops else ""),
              f"cell a b c = {cp[0]:.4f} {cp[1]:.4f} {cp[2]:.4f} A, alpha beta gamma = {cp[3]:.3f} {cp[4]:.3f} {cp[5]:.3f}",
              "Voigt xx yy zz yz xz xy, Cartesian frame of that cell; for warpt --elastic_constants"]
    write_elastic_constants(a.output, C, header)
    if eig.min() <= 0.0:
        print(f"[static_elastic] WARNING: not positive definite (smallest eigenvalue {eig.min():.2f} GPa); "
              f"{a.output} written but --elastic_constants will refuse it", file=sys.stderr)
        return 1
    read_elastic_constants(a.output)          # the file must pass WARP's own check
    print(f"[static_elastic] wrote {a.output} (smallest eigenvalue {eig.min():.1f} GPa); "
          f"use: python3 -m warpt ... --elastic_constants {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
