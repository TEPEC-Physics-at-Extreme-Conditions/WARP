#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Machine-learned potential backend that mimics Quantum ESPRESSO output.

Evaluates energy, forces and stress for a structure with an MLIP (UPET or MACE
through ASE) and prints them in a QE-like format so the collector
(:mod:`warpt.tdep_collect`, :func:`warpt.io_qe.read_qe_stress_gpa`) handles both
calculators. Forces are converted to Ry/bohr (sign preserved) and stress to
Ry/bohr³ with the sign flipped to QE's compression-positive convention.

Reads: the QE ``pw.x`` input named by ``-input``, or the ``--batch`` list file of
``<input> <output>`` pairs, and the model file named by ``--ml_params``.
Writes: the QE-like block to stdout (single mode) or to each listed output file
(batch mode), ending in ``JOB DONE.`` on success or ``JOB FAILED. <error>`` on
failure. Run as ``python3 -m warpt.MLIP_interface`` (the package-relative import
needs the module form); ``main`` lists the options. When no CUDA device is
available, ``torch.load`` and ``torch.jit.load`` are wrapped at import time to
deserialise on the CPU.

Public functions: ``read_qe_input_file``, ``print_qe_like``, ``make_upet_calc``,
``make_mace_calc``, ``make_calc``, ``run_once``, ``run_batch``, ``main``.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"e3nn\.o3\._wigner")
warnings.filterwarnings("ignore", message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", message=r".*Creating a tensor from a list of numpy\.ndarrays.*")

import argparse
import numpy as np
import sys
from pathlib import Path

from ase.units import Rydberg, Bohr

from upet.calculator import UPETCalculator
from .constants import KBAR_TO_GPA, RYBOHR3_TO_GPA
# Defined in io_qe (shared with VASP_interface) and re-exported as part of this
# module's public interface.
from .io_qe import read_qe_input_file, print_qe_like  # noqa: F401

# Force CPU deserialisation when no CUDA device is present. A CUDA-trained
# checkpoint stores CUDA tensors, and e3nn's CodeGenMixin.__setstate__ calls
# torch.jit.load without map_location, so both torch.jit.load and torch.load are
# wrapped with map_location="cpu". No-op when CUDA is available.
import functools as _ft
import torch as _torch

if not _torch.cuda.is_available():
    if not getattr(_torch.jit.load, "_warpt_cpu_patched", False):
        _j = _ft.partial(_torch.jit.load, map_location="cpu")
        _j._warpt_cpu_patched = True
        _torch.jit.load = _j
    if not getattr(_torch.load, "_warpt_cpu_patched", False):
        _l = _ft.partial(_torch.load, map_location="cpu")
        _l._warpt_cpu_patched = True
        _torch.load = _l

# UPET model name -> default version
MODELS = {
    "pet-mad-xs": "1.5.0",
    "pet-mad-s": "1.5.0",
    "pet-omat-s": "1.0.0",
    "pet-omat-m": "1.0.0",
    "pet-omat-l": "1.0.0",
    "pet-omatpes-l": "0.1.0",
}


KBAR_PER_RY_BOHR3 = RYBOHR3_TO_GPA / KBAR_TO_GPA  # kbar per Ry/bohr^3


def _rot_avg_order(args):
    """Return None for ``--rot-avg 0``, otherwise the integer order.

    ``UPETCalculator`` (upet 0.2.1) wraps the model in ``SymmetrizedCalculator``
    whenever ``rotational_average_order`` is not None, including 0, and its
    ``calculate()`` aliases ``self.results`` to the inner calculator's dictionary. With
    that wrapper a calculator reused across structures can return the previous
    structure's forces when only the energy is requested. Passing None keeps the
    wrapper out.
    """
    order = getattr(args, "rot_avg", 0)
    return None if not order else int(order)


def make_upet_calc(args):
    """Construct a UPET ASE calculator from a checkpoint (``--ml_params``) or model+version.

    Parameters
    ----------
    args : argparse.Namespace
        Fields used: ``ml_params``, ``device``, ``conservative``, ``rot_avg`` (through
        :func:`_rot_avg_order`), ``rot_avg_batch``, ``variant``, and ``model`` and
        ``version`` when ``ml_params`` is empty (version default: the ``MODELS`` entry
        for ``model``, else ``latest``).

    Returns
    -------
    upet.calculator.UPETCalculator
        Built from the checkpoint file when ``ml_params`` is given, otherwise from the
        model name and version, which UPET fetches from the Hugging Face Hub when not
        cached locally.

    Raises
    ------
    FileNotFoundError
        When ``ml_params`` names a file that does not exist.
    """
    if args.ml_params:
        ckpt = Path(args.ml_params).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"UPET checkpoint not found: {ckpt}")
        return UPETCalculator(
            checkpoint_path=str(ckpt),
            device=args.device,
            non_conservative=(not args.conservative),
            rotational_average_order=_rot_avg_order(args),
            rotational_average_batch_size=args.rot_avg_batch,
            variants=({"energy": args.variant} if args.variant else None),
        )

    version = args.version or MODELS.get(args.model, "latest")
    return UPETCalculator(
        model=args.model,
        version=version,
        device=args.device,
        non_conservative=(not args.conservative),
        rotational_average_order=_rot_avg_order(args),
        rotational_average_batch_size=args.rot_avg_batch,
            variants=({"energy": args.variant} if args.variant else None),
    )


def make_mace_calc(args):
    """Construct a MACE ASE calculator from the ``--ml_params`` model file.

    ``mace.calculators.mace_mp`` is imported here, not at module level, so the UPET
    engine runs without mace installed.

    Parameters
    ----------
    args : argparse.Namespace
        Fields used: ``ml_params``, ``dispersion``, ``device``, ``dtype``.

    Returns
    -------
    ase.calculators.calculator.Calculator
        The calculator returned by ``mace_mp`` for the model file.

    Raises
    ------
    ImportError
        When mace is not installed (raised before the arguments are checked).
    ValueError
        When ``ml_params`` is empty.
    FileNotFoundError
        When ``ml_params`` names a file that does not exist.
    """
    # Imported here so that the UPET engine runs without mace installed.
    from mace.calculators import mace_mp 

    if not args.ml_params:
        raise ValueError("MACE requires --ml_params pointing to a .model file")

    model_path = Path(args.ml_params).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MACE model not found: {model_path}")

    return mace_mp(
        model=str(model_path),
        dispersion=args.dispersion,
        device=args.device,
        default_dtype=args.dtype,
    )


def make_calc(args):
    """Dispatch to the UPET or MACE calculator factory based on ``args.engine``.

    Parameters
    ----------
    args : argparse.Namespace
        ``engine`` selects the factory; the other fields are those of
        :func:`make_upet_calc` or :func:`make_mace_calc`.

    Returns
    -------
    ase.calculators.calculator.Calculator
        The calculator built by the selected factory.

    Raises
    ------
    RuntimeError
        When ``engine`` is neither ``upet`` nor ``mace``. The factories' own errors
        propagate.
    """
    if args.engine == "upet":
        return make_upet_calc(args)
    if args.engine == "mace":
        return make_mace_calc(args)
    raise RuntimeError(f"Unknown engine: {args.engine}")


def run_once(args, calc=None, input_path=None):
    """Evaluate one structure with the MLIP and print the QE-like block to stdout.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options of :func:`main`; ``args.input`` is used when ``input_path`` is
        None and ``args.allow_no_stress`` selects zero stress when none is available.
    calc : ase.calculators.calculator.Calculator or None
        Calculator to reuse; built with :func:`make_calc` when None.
    input_path : str or None
        QE input file to read instead of ``args.input``.

    Side effects
    ------------
    Prints the QE-like block to stdout (:func:`~warpt.io_qe.print_qe_like`): the
    energy in Ry, the forces in Ry/bohr (sign preserved) and the stress in Ry/bohr³
    with the sign flipped to QE's compression-positive convention. Returns None.

    Raises
    ------
    RuntimeError
        When the calculator provides no stress and ``--allow_no_stress`` is not set.
    """
    atoms = read_qe_input_file(input_path or args.input)
    atoms.calc = calc if calc is not None else make_calc(args)

    # Forces are requested before the energy. UPETCalculator.calculate() ends with
    # ``self.results = self.calculator.results``, aliasing the inner dictionary,
    # which ASE's per-call reset does not clear; an energy-only request would then
    # leave a stale 'forces' entry from a previous structure. Requesting forces
    # first makes one calculate() call produce energy, forces and stress. The
    # clear below covers a calculator passed in with results already present.
    if getattr(atoms.calc, "results", None):
        try:
            atoms.calc.reset()
            inner = getattr(atoms.calc, "calculator", None)
            if inner is not None and hasattr(inner, "results"):
                inner.results.clear()
        except Exception:
            pass

    # ASE -> QE units
    forces_ry_bohr = atoms.get_forces() / Rydberg * Bohr
    energy_ry = atoms.get_potential_energy() / Rydberg

    # Stress: ASE gives eV/Å^3, tension-positive; QE wants Ry/bohr^3, compression-positive.
    try:
        stress_ry_bohr3 = atoms.get_stress(voigt=False) * (-(1.0 / Rydberg) * (Bohr ** 3))
    except Exception:
        if args.allow_no_stress:
            stress_ry_bohr3 = np.zeros((3, 3), float)
        else:
            raise RuntimeError("Calculator did not provide stress; required (use --allow_no_stress to force zeros).")

    print_qe_like(atoms, energy_ry, forces_ry_bohr, stress_ry_bohr3)


def run_batch(args):
    """Evaluate every structure of ``args.batch`` in this process with one calculator.

    The batch file lists ``<input> <output>`` pairs, one per line; blank lines and
    lines starting with ``#`` are ignored. The calculator is built once and reused,
    so the freshness of the results relies on :func:`_rot_avg_order` (no
    ``SymmetrizedCalculator`` wrapper) and on :func:`run_once` requesting forces before
    the energy.

    Side effects
    ------------
    For each pair, stdout is redirected to the output file, which receives the
    QE-like block followed by ``JOB DONE.``, or ``JOB FAILED. <error>`` when the
    evaluation raises (the error is also printed to stderr). Finished outputs stay
    valid if a later structure fails, so the runner's resume logic can reuse them.
    Returns None; calls ``sys.exit(1)`` after the loop if any structure failed.
    """
    pairs = []
    with open(args.batch) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            src, dst = line.split(None, 1)
            pairs.append((src, dst.strip()))
    if not pairs:
        return

    calc = make_calc(args)                      # built once, reused for every structure
    out0 = sys.stdout
    failures = 0
    for src, dst in pairs:
        try:
            with open(dst, "w") as fh:
                sys.stdout = fh
                try:
                    run_once(args, calc=calc, input_path=src)
                    print("JOB DONE.", flush=True)
                except Exception as exc:
                    print(f"JOB FAILED. {exc}", flush=True)
                    sys.stdout = out0
                    print(f"[MLIP_interface] Error on {src}: {exc}",
                          file=sys.stderr, flush=True)
                    failures += 1
        finally:
            sys.stdout = out0
    if failures:
        sys.exit(1)


def main():
    """Command-line entry point.

    Options: ``-input`` (QE input; required unless ``--batch``), ``--batch`` (list of
    ``<input> <output>`` pairs), ``--engine`` (``upet``, the default, or ``mace``),
    ``--device`` (``cpu``, the default, or ``cuda``), ``--ml_params`` (model file),
    ``--model`` (UPET model name, default ``pet-mad-s``, one of the ``MODELS`` keys),
    ``--version`` (UPET model version; default: the ``MODELS`` entry for ``--model``,
    else ``latest``), ``--conservative``, ``--variant``, ``--rot-avg``,
    ``--rot-avg-batch``, ``--dispersion``, ``--dtype`` and ``--allow_no_stress``.

    Single mode prints the QE-like block and ``JOB DONE.`` to stdout, or
    ``JOB FAILED. <error>`` and exits 1. Batch mode writes one output file per
    structure and prints ``JOB DONE.`` to its own stdout when every structure
    succeeded, else exits 1.
    """
    p = argparse.ArgumentParser(description="QE-like runner for WARP: evaluates structures with a "
                                            "machine-learned potential (UPET or MACE); each output "
                                            "ends in 'JOB DONE.' on success or 'JOB FAILED. <error>' "
                                            "on failure.")

    # runner.build_cmd appends '-input <file>' to the calculator command.
    p.add_argument("-input", required=False,
                   help="QE input file (pw.x style). Required unless --batch is given.")

    # Engine selection
    p.add_argument("--engine", choices=["upet", "mace"], default="upet")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # Shared: either UPET checkpoint (.ckpt) or MACE model (.model), depending on engine
    p.add_argument("--ml_params", default=None,
                   help="Model file: a UPET .ckpt (engine upet; overrides --model/--version) or "
                        "a MACE .model (engine mace, required). Default: none, so UPET downloads "
                        "--model/--version.")

    # UPET options (--model and --version are ignored when --ml_params names a checkpoint)
    p.add_argument("--model", default="pet-mad-s", choices=sorted(MODELS.keys()))
    p.add_argument("--version", default=None)
    p.add_argument("--conservative", action="store_true",
                   help="Compute UPET forces and stress as energy gradients (autograd) instead of "
                        "the direct force/stress heads; uses more memory. Default: off (direct "
                        "heads).")
    p.add_argument("--variant", default=None,
                   help="UPET energy head to evaluate, e.g. 'lah' for a checkpoint fine-tuned on "
                        "the head 'energy/lah'. Default: none, which uses the checkpoint's base "
                        "head; a fine-tuned checkpoint evaluated without its head returns the "
                        "base-model energies.")
    p.add_argument("--batch", default=None,
                   help="File of '<input> <output>' pairs, one per line (blank and '#' lines "
                        "ignored); all are evaluated in this process with one calculator and each "
                        "output ends in 'JOB DONE.' or 'JOB FAILED.'. Default: none (single "
                        "-input mode).")
    p.add_argument("--rot-avg", type=int, default=0,
                   help="Lebedev-Laikov order for UPET rotational averaging; 0 (default) disables "
                        "the averaging wrapper.")
    p.add_argument("--rot-avg-batch", type=int, default=8,
                   help="Number of rotated copies evaluated per batch when --rot-avg > 0 "
                        "(default 8).")

    # MACE options
    p.add_argument("--dispersion", action="store_true",
                   help="Enable MACE's dispersion correction (default: off).")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                   help="Floating-point precision of the MACE model: float32 or float64 "
                        "(default float64).")

    # Stress policy
    p.add_argument("--allow_no_stress", action="store_true",
                   help="Print a zero stress tensor when the calculator provides none instead of "
                        "failing (default: off; not recommended for vc-relax).")

    args = p.parse_args()

    if getattr(args, "batch", None):
        # run_batch writes a marker into each per-structure output. The batch
        # process prints one to its own stdout as well, because runner.job_done()
        # checks the output file of the job it launched; without it a batch in
        # which every structure succeeded would be reported as failed.
        run_batch(args)
        print("JOB DONE.", flush=True)
        return
    if not args.input:
        p.error("either -input or --batch is required")

    # JOB DONE. is the completion marker tested by runner.job_done(),
    # tdep_collect._file_ready() and validity.valid_scf(); it is printed only after
    # a successful evaluation. On failure JOB FAILED. <error> is printed instead,
    # which tdep_collect._file_ready() turns into an immediate abort.
    try:
        run_once(args)
    except Exception as exc:
        print(f"[MLIP_interface] Error: {exc}", file=sys.stderr, flush=True)
        print(f"JOB FAILED. {exc}", flush=True)
        sys.exit(1)
    print("JOB DONE.", flush=True)


if __name__ == "__main__":
    main()
