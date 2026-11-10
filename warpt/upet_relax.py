#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Symmetry-constrained static vc-relax onto the MLIP's own minimum at a target pressure.

A WARP run with an MLIP should start from the geometry where that MLIP itself gives
vanishing classical forces and an isotropic stress equal to the target pressure, so
that the residuals WARP measures are not the DFT/MLIP mismatch (the protocol of
Errea et al., Nature 578, 66 (2020)). This tool produces that geometry and checks it
against an acceptance gate.

Reads: the structure named by ``--structure`` (any format ``ase.io.read`` accepts)
and the UPET checkpoint named by ``--ml_params``.
Writes: the relaxed structure as a VASP5 Direct POSCAR (``--out``, default
``<stem>_upet_static_<P>GPa.vasp``) and, with ``--report``, a JSON report of the
start and final states and the gate results.

Usage::

    python3 -m warpt.upet_relax --structure LaH10_150GPa.vasp \\
        --press 150 --ml_params pet-omat-l-v1.0.0.ckpt --variant none

``--variant none`` evaluates the checkpoint's base head; for a fine-tuned checkpoint,
pass the name of its fine-tuned head instead.

Exit codes: 0 = relaxed and passed the gate (or ``--eval-only``), 3 = gate failed
(the structure is still written), 1 = error.

Public functions: ``state``, ``spacegroup``, ``fmt``, ``main``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from .constants import EV_TO_GPA_A3

GPA = EV_TO_GPA_A3   # eV/A^3 -> GPa

# Checkpoint filename pattern, used only for a warning: the warning is printed for any
# checkpoint not named ``pet-omat-<s|m|l|xl>-v<version>.ckpt`` (version: digits and
# dots). UPET itself (0.2.1) accepts ``pet-<model>-<xs|s|m|l|xl>-[v]<x.y.z>.ckpt`` and
# infers model, size and version from that name; any other name yields none of them.
CKPT_NAME = re.compile(r"^pet-omat-(s|m|l|xl)-v[\d.]+\.ckpt$")


def _fix_symmetry(atoms, symprec):
    """Return a FixSymmetry constraint (cell and positions adjusted) at tolerance ``symprec`` (Å).

    Imported from ``ase.constraints``, or from ``ase.spacegroup.symmetrize`` on ASE
    releases before 3.23.
    """
    try:
        from ase.constraints import FixSymmetry
    except ImportError:  # ASE < 3.23
        from ase.spacegroup.symmetrize import FixSymmetry
    return FixSymmetry(atoms, symprec=symprec, adjust_cell=True, adjust_positions=True)


def _read_structure(path, informat=None):
    """Read a geometry with ``ase.io.read``, tolerating ``@`` in the filename.

    ``ase.io.read`` parses ``name@index`` as slice syntax, so a path containing ``@``
    is opened and passed as a file handle; the format then cannot be inferred from
    the name and defaults to ``vasp`` unless ``informat`` is given.
    """
    from ase.io import read
    if "@" in str(path):
        with open(path) as fh:
            return read(fh, format=informat or "vasp")
    return read(path, format=informat)


def state(atoms):
    """Pressure, stress anisotropy, residual forces and cell parameters of ``atoms``.

    Stress is taken from ASE (tension-positive) and converted to GPa, so
    ``P = -tr(sigma)/3`` and the deviatoric part is ``sigma + P*I``.

    Parameters
    ----------
    atoms : ase.Atoms
        With a calculator attached; ``get_stress`` and ``get_forces`` are called (forces
        once with ``apply_constraint=False``, once with the constraint applied).

    Returns
    -------
    dict
        ``pressure_GPa``; ``aniso_GPa`` (largest absolute deviatoric component);
        ``s_xx_minus_zz_GPa``; ``fmax_meV_per_A`` (largest force component with the
        constraint not applied); ``fmax_sym_meV_per_A`` (same after the symmetry
        constraint has projected the forces; this is what the gate tests);
        ``volume_A3``; ``vol_per_atom_A3``; ``a``, ``b``, ``c`` (Å); ``alpha``, ``beta``,
        ``gamma`` (degrees); ``c_over_a``.
    """
    s = atoms.get_stress(voigt=False) * GPA
    p = float(-np.trace(s) / 3.0)
    dev = s + p * np.eye(3)
    f = atoms.get_forces(apply_constraint=False)
    f_sym = atoms.get_forces()
    cell = atoms.cell.cellpar()
    return dict(
        pressure_GPa=p,
        aniso_GPa=float(np.abs(dev).max()),
        s_xx_minus_zz_GPa=float(s[0, 0] - s[2, 2]),
        fmax_meV_per_A=float(np.abs(f).max() * 1000.0),
        fmax_sym_meV_per_A=float(np.abs(f_sym).max() * 1000.0),
        volume_A3=float(atoms.get_volume()),
        vol_per_atom_A3=float(atoms.get_volume() / len(atoms)),
        a=float(cell[0]), b=float(cell[1]), c=float(cell[2]),
        alpha=float(cell[3]), beta=float(cell[4]), gamma=float(cell[5]),
        c_over_a=float(cell[2] / cell[0]),
    )


def spacegroup(atoms, symprec):
    """Space-group number of ``atoms`` from spglib at tolerance ``symprec`` (Å).

    Returns None when spglib is not installed or fails.
    """
    try:
        import spglib
        ds = spglib.get_symmetry_dataset(
            (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
            symprec=symprec)
        return None if ds is None else int(ds.number)
    except Exception:  # noqa: BLE001 - a missing or failing spglib must not abort the relaxation
        return None


def fmt(tag, st, spg):
    """Format one report line from ``st`` (a :func:`state` dictionary) with label ``tag`` and
    space-group number ``spg``."""
    return (f"  {tag:9s} P = {st['pressure_GPa']:9.4f} GPa | aniso {st['aniso_GPa']:7.4f} "
            f"| xx-zz {st['s_xx_minus_zz_GPa']:+8.4f} "
            f"| max|F| {st['fmax_meV_per_A']:7.3f} (sym {st['fmax_sym_meV_per_A']:6.3f}) meV/A "
            f"| V/atom {st['vol_per_atom_A3']:.4f} | c/a {st['c_over_a']:.4f} | spg {spg}")


def main():
    """Command-line entry point; returns the process exit code (0, 1 or 3).

    Parses the options, loads the checkpoint, reports the starting state, relaxes the
    cell and positions with ``FrechetCellFilter`` under ``FixSymmetry`` (unless
    ``--no-fixsym``), evaluates and prints the gate (optimizer converged, |P - target|,
    anisotropy, symmetrised residual force, space group unchanged), then writes the
    POSCAR and the optional JSON report (which records the gate result) and returns 3
    when the gate failed.

    Options without a help text: ``--device`` (torch device, ``cpu`` (default) or
    ``cuda``), ``--steps`` (maximum number of optimizer steps, default 500),
    ``--optimizer`` (``lbfgs`` (default) or ``fire``) and ``--symprec`` (symmetry
    tolerance in Å for spglib and FixSymmetry, default 1e-5).
    """
    p = argparse.ArgumentParser(
        description="Symmetry-constrained static vc-relax onto the MLIP's own minimum "
                    "at a target pressure. Produces the -ucp structure for WARP.")
    p.add_argument("--structure", required=True, help="input geometry (POSCAR/vasp, cif, ...)")
    p.add_argument("--informat", default=None,
                   help="ASE format name of the input, e.g. vasp for outfile.new_ucposcar. "
                        "Default: inferred from the filename (vasp when the name contains '@').")
    p.add_argument("--press", type=float, required=True, help="target pressure in GPa")
    p.add_argument("--out", default=None,
                   help="output POSCAR (default: <stem>_upet_static_<P>GPa.vasp)")
    p.add_argument("--report", default=None,
                   help="path of a JSON report with start/final state and gate results "
                        "(default: none)")

    p.add_argument("--ml_params", required=True, help="UPET checkpoint (.ckpt)")
    p.add_argument("--variant", required=True,
                   help="UPET energy head to evaluate, e.g. 'lah' for a fine-tuned head; pass "
                        "'none' for the checkpoint's base head. Required: a fine-tuned checkpoint "
                        "evaluated without its head returns the base-model energies.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--non-conservative", action="store_true",
                   help="use the direct UPET force and stress heads instead of energy gradients "
                        "(faster; default: off, energy gradients)")

    p.add_argument("--fmax", type=float, default=0.001,
                   help="optimizer convergence threshold in eV/A on the forces and cell gradients "
                        "of the FrechetCellFilter (default 0.001)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "fire"])
    p.add_argument("--symprec", type=float, default=1e-5)
    p.add_argument("--no-fixsym", action="store_true",
                   help="relax without the FixSymmetry constraint (default: constraint on); the "
                        "space-group check of the gate still applies")
    p.add_argument("--eval-only", action="store_true",
                   help="print the starting state and exit 0 without relaxing or writing any file")

    p.add_argument("--gate-press", type=float, default=0.05,
                   help="max |P - target| in GPa to pass (default 0.05)")
    p.add_argument("--gate-aniso", type=float, default=0.10,
                   help="max deviatoric stress in GPa to pass (default 0.10)")
    p.add_argument("--gate-fmax", type=float, default=2.0,
                   help="max symmetrised residual force in meV/A to pass (default 2.0). The "
                        "symmetry-projected force is used because the raw residual includes the "
                        "model's equivariance error, which no symmetry-preserving move can reduce.")
    p.add_argument("--warn-fmax-raw", type=float, default=10.0,
                   help="print a diagnostic if the RAW residual force exceeds this "
                        "(meV/A). Not a gate: it measures the model, not the geometry.")
    args = p.parse_args()

    from ase.io import read, write
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE, LBFGS
    from upet.calculator import UPETCalculator

    given = Path(args.ml_params).expanduser()
    ck = given.resolve()
    if not ck.exists():
        print(f"[upet_relax] checkpoint not found: {ck}", file=sys.stderr)
        return 1
    # Test the name as given, not the resolved path: a correctly named symlink may
    # point at a file with another name.
    if not CKPT_NAME.match(given.name):
        print(f"[upet_relax][warn] checkpoint is named '{given.name}'. UPET infers the "
              f"architecture from the FILENAME; rename it to e.g. "
              f"pet-omat-l-v1.0.0.ckpt if the load fails or the numbers look wrong.",
              flush=True)

    atoms = _read_structure(args.structure, args.informat)
    spg0 = spacegroup(atoms, args.symprec)

    variants = None if args.variant.lower() == "none" else {"energy": args.variant}
    atoms.calc = UPETCalculator(
        checkpoint_path=str(ck), device=args.device,
        non_conservative=args.non_conservative,
        rotational_average_order=0, rotational_average_batch_size=8,
        variants=variants,
    )
    if not args.no_fixsym:
        atoms.set_constraint(_fix_symmetry(atoms, args.symprec))

    print(f"[upet_relax] {args.structure}  ({len(atoms)} atoms, {atoms.get_chemical_formula()})")
    print(f"[upet_relax] target {args.press:g} GPa | variant {args.variant} | "
          f"{'no ' if args.no_fixsym else ''}FixSymmetry(symprec={args.symprec:g})")
    st0 = state(atoms)
    print(fmt("start", st0, spg0), flush=True)

    if args.eval_only:
        print("[upet_relax] --eval-only: not relaxing")
        return 0

    t0 = time.time()
    # scalar_pressure is positive for compression and lives in eV/A^3.
    filt = FrechetCellFilter(atoms, scalar_pressure=args.press / GPA)
    Opt = LBFGS if args.optimizer == "lbfgs" else FIRE
    opt = Opt(filt, logfile="-")
    converged = opt.run(fmax=args.fmax, steps=args.steps)
    wall = time.time() - t0

    st1 = state(atoms)
    spg1 = spacegroup(atoms, args.symprec)
    print(fmt("final", st1, spg1))
    print(f"[upet_relax] {opt.get_number_of_steps()} steps, {wall / 60:.1f} min, "
          f"optimizer converged = {bool(converged)}")

    dp = abs(st1["pressure_GPa"] - args.press)
    checks = [
        ("optimizer converged", bool(converged), "", ""),
        ("|P - target|", dp <= args.gate_press, f"{dp:.4f}", f"<= {args.gate_press} GPa"),
        ("anisotropy", st1["aniso_GPa"] <= args.gate_aniso,
         f"{st1['aniso_GPa']:.4f}", f"<= {args.gate_aniso} GPa"),
        ("max|F| (sym)", st1["fmax_sym_meV_per_A"] <= args.gate_fmax,
         f"{st1['fmax_sym_meV_per_A']:.3f}", f"<= {args.gate_fmax} meV/A"),
        ("space group kept", spg0 == spg1, f"{spg0} -> {spg1}", ""),
    ]
    # The raw residual is reported, not gated: for atoms on special positions the
    # symmetrised force is exactly zero while the raw force is the model's
    # equivariance error, which grows as the symmetry falls.
    if st1["fmax_meV_per_A"] > args.warn_fmax_raw:
        print(f"[upet_relax][note] raw residual force {st1['fmax_meV_per_A']:.3f} meV/A "
              f"vs symmetrised {st1['fmax_sym_meV_per_A']:.3f}. The difference is the "
              f"model's symmetry-breaking (equivariance) error at this geometry, not a "
              f"structural residual. Rotational averaging in the sampling backend "
              f"(MLIP_interface --rot-avg) reduces it; for scale, the default "
              f"position-loop tolerances --fc1_mean_tol and --fc1_max_tol are 30 and "
              f"75 meV/A.")
    print("[upet_relax] acceptance gate:")
    for name, ok, got, want in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {name:20s} {got:>12s} {want}")
    passed = all(c[1] for c in checks)

    out = Path(args.out) if args.out else Path(
        f"{Path(args.structure).stem}_upet_static_{args.press:g}GPa.vasp")
    # Written without the constraint: with FixSymmetry attached ASE writes a
    # "Selective dynamics" block with per-atom T/F flags, which TDEP's
    # generate_structure does not parse. The geometry is unchanged.
    final = atoms.copy()
    final.set_constraint()
    write(str(out), final, format="vasp", direct=True, vasp5=True, sort=False)
    print(f"[upet_relax] wrote {out}")

    if args.report:
        Path(args.report).write_text(json.dumps(dict(
            structure=str(args.structure), target_press_GPa=args.press,
            checkpoint=str(ck), variant=args.variant,
            fixsym=not args.no_fixsym, symprec=args.symprec,
            optimizer=args.optimizer, fmax_target=args.fmax,
            n_steps=int(opt.get_number_of_steps()), wall_s=wall,
            optimizer_converged=bool(converged),
            spacegroup_start=spg0, spacegroup_final=spg1,
            start=st0, final=st1, gate_passed=passed,
            gate=dict(press=args.gate_press, aniso=args.gate_aniso, fmax=args.gate_fmax),
            output=str(out),
        ), indent=2))
        print(f"[upet_relax] wrote {args.report}")

    if not passed:
        print("[upet_relax] GATE FAILED -- inspect the structure before using it as the "
              "-ucp for WARP; an automated scan should skip this point.", file=sys.stderr)
        return 3
    print("[upet_relax] gate passed; this is a valid -ucp for WARP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
