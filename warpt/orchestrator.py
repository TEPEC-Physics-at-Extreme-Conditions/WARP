#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Command-line entry point and run setup for WARP.

Parses the command line, builds the frozen :class:`~warpt.context_state.WarpContext`
and the mutable :class:`~warpt.context_state.WarpState`, selects the calculator
command and dispatches to :func:`~warpt.workflow_vc_relax.run_vc_relax`
(``-mode vc-relax``, after printing the plan from
:func:`~warpt.restart_check.find_resume_plan`) or to
:func:`~warpt.workflow_preconditioning.run_preconditioning`
(``-mode preconditioning``). Invoked as ``python -m warpt`` or as the ``warpt``
console script.

Resolves the master QE input deck (``-m``) and the unit-cell POSCAR (``-ucp``) against
the working directory, which becomes the run root, and checks that the master deck
exists and, with ``--calculator VASP``, that the VASP template matches the elements of
the unit cell. Writes nothing itself; the ``iter_<n>`` tree and its step directories are
created by the workflows.

Public functions: ``parse_arguments``, ``build_ctx_state``, ``main``.
"""
import argparse
import os
import shutil
import numpy as np

from . import __version__
from .context_state import WarpContext, WarpState
from .utilities import die, parse_supercell, parse_masses
from .constants import kB_eV, hbar_eVs, EV_TO_GPA_A3
from .restart_check import find_resume_plan
from .workflow_vc_relax import run_vc_relax
from .workflow_preconditioning import run_preconditioning


# To the right of the letters: an Alcubierre warp bubble in profile. Space expands
# behind the ship (York time theta > 0) and contracts in front of it (theta < 0).
BANNER = rf"""
   ██     ██   █████   ██████   ██████         ▄▄▄▄
   ██     ██  ██░░░██  ██░░░██  ██░░░██      ▄▀░░░░▀▄
   ██  █  ██  ███████  ██████░  ██████░   ━━▀        ▀━ ══▐█▌ ━▄        ▄━━
   ██ ███ ██  ██░░░██  ██░░██   ██░░░░                         ▀▄░░░░▄▀
    ███ ███   ██░  ██  ██░ ██   ██░                              ▀▀▀▀
     ░   ░     ░    ░   ░   ░    ░          θ > 0                θ < 0

   WARP v{__version__}
   Wyckoff-constrained Anharmonic Relaxation
   at Pressure and temperature
   ------------------------------------------------------------------
   TDEP-based quantum / anharmonic vc-relax  |  QE / VASP / MLIP
   Interface to Olle Hellman's TDEP

   H. R. A. ten Eikelder   <henricus.ten.eikelder@liu.se>
   F. Trybel               <florian.trybel@liu.se>
   Cite WARP: https://arxiv.org/abs/2611.12736
   GPL-3.0-or-later, without any warranty; see LICENSE and NOTICE
"""


def parse_arguments():
    """Define and parse the WARP command-line interface.

    The options that select the run mode and the force engine:

    * ``-mode`` (required; ``vc-relax`` or ``preconditioning``): ``vc-relax`` runs the
      preconditioning schedule in ``iter_0`` and then position and lattice steps per
      ``iter_<n>``; ``preconditioning`` runs the force-constant bootstrap only, in
      ``preconditioning/`` under the working directory.
    * ``-c`` / ``--calculator`` (``QE``, ``MLIP`` or ``VASP``, default ``QE``): ``QE``
      runs the ``--pwx`` command per sample; ``MLIP`` runs
      ``python3 -m warpt.MLIP_interface --engine <-mle> --device <-mld> --ml_params <-mlp>``
      (+ ``--variant <-mlv>``, ``--conservative`` with ``-mlc``); ``VASP`` runs
      ``python3 -m warpt.VASP_interface --template <--vasp_template>`` with the
      command given by ``--vasp_cmd``.
    * ``-mle`` / ``--ml_engine`` (``upet`` or ``mace``, default ``upet``): forwarded as
      ``--engine`` to ``warpt.MLIP_interface``; used only with ``--calculator MLIP``.
    * ``-mld`` / ``--ml_device`` (``cpu`` or ``cuda``, default ``cpu``): forwarded as
      ``--device`` to ``warpt.MLIP_interface``; used only with ``--calculator MLIP``.

    Returns
    -------
    argparse.Namespace
        The parsed CLI options (calculator, structure/master deck, temperature,
        FC cutoffs, supercell, target stress, schedule, etc.).

    Raises
    ------
    SystemExit
        With code 2 from argparse on an invalid or missing required option. ``--help``
        prints the option list and exits with code 0 (``main`` has printed ``BANNER``
        before the arguments are parsed, so the banner appears once).
    """
    p = argparse.ArgumentParser(
        prog="warpt",
        description="Relax a crystal structure at finite temperature and pressure with "
                    "TDEP sampling and Quantum ESPRESSO, VASP or machine-learned forces.",
    )

    # --- required ---
    p.add_argument("-m", "--masterscf", required=True,
                   help="Quantum ESPRESSO input deck (relative to the working directory) used as "
                        "the template for every sampled configuration. Its basename is copied "
                        "into each step directory and prefixes the <master>.samplesN inputs and "
                        "their .out files.")
    p.add_argument("-sc", "--supercell", nargs="+", required=True,
                   help="Supercell passed to generate_structure: 1 integer = target number of "
                        "atoms (--desired_na), 3 integers = diagonal repetitions (--dimensions), "
                        "9 integers = row-major nondiagonal matrix (--nondiagonal_dimensions).")
    # vc-relax: the full loop (preconditioning in iter_0, then position and lattice steps in
    # iter_N). preconditioning: only the force-constant bootstrap, in preconditioning/ directly
    # under the working directory.
    p.add_argument("-mode", "--mode", dest="mode",
                   choices=["vc-relax", "preconditioning"], required=True,
                   help="vc-relax: run the preconditioning stage as iteration 0, then the "
                        "position and lattice loop, resuming an incomplete run found in the "
                        "working directory. preconditioning: run only the preconditioning "
                        "stage; geoms already complete are skipped.")

    # --- structure / run controls ---
    p.add_argument("-ucp", "--ucp", "--unit-cell-poscar", dest="ucp", required=True,
                   help="Starting unit cell in POSCAR format (relative to the working directory). "
                        "Copied to infile.ucposcar in the preconditioning directory "
                        "(iter_0/preconditioning/ in a vc-relax run).")
    p.add_argument("-n", "--n", "--num-samples", "--num_samples", dest="n",
                   type=int, default=64,
                   help="Configurations sampled per position step and per lattice step "
                        "(default 64). Preconditioning uses --precond_schedule instead; "
                        "FC3-gate retries scale this by --fc3_gate_nscale.")
    p.add_argument("-t", "--t", "--temp", "--temperature", dest="temp",
                   type=int, default=300,
                   help="Temperature in K (integer, default 300): canonical_configuration -t and "
                        "the Bose-Einstein factor of the phonon stress.")

    # --- stress / convergence ---
    p.add_argument("-pe", "--press_ext", "--press-ext", dest="press_ext",
                   nargs="+", type=float, required=True,
                   help="Target external pressure/stress in GPa, compression positive: 1 value = "
                        "hydrostatic with zero shear target, 3 values = diagonal xx yy zz, "
                        "6 values = Voigt xx yy zz yz xz xy. Stored internally with the opposite "
                        "sign (tension positive).")
    p.add_argument("-pct", "--press_conv_th", "--press-conv-th", "--pressure_convergence_threshold",
                   dest="press_conv_th", type=float, default=0.005,
                   help="Stress convergence gate in GPa (default 0.005): every compared component "
                        "of |sigma_int - sigma_ext| must be below it (all six in scalar and full6 "
                        "mode, the three diagonals in diag3 mode). The same value gates the change "
                        "of the phonon stress between lattice steps.")

    # --- cutoffs ---
    p.add_argument("-rc2", "--rc2", dest="rc2", type=float, default=5.0,
                   help="Second-order force-constant cutoff in Å (default 5.0), passed as "
                        "extract_forceconstants -rc2.")
    p.add_argument("-rc3", "--rc3", dest="rc3", type=float, default=None,
                   help="Third-order force-constant cutoff in Å, passed as extract_forceconstants "
                        "-rc3. Default: none, i.e. no third-order fit.")

    # --- statistics / reuse ---
    p.add_argument("-q", "--quantum", dest="quantum", action="store_true",
                   help="Sample configurations with Bose-Einstein (quantum) statistics: adds "
                        "--quantum to canonical_configuration. Default: classical Boltzmann "
                        "sampling.")
    p.add_argument("--masses", dest="masses", default=None,
                   help="Atomic masses for the thermal sampling and the phonons, as "
                        "ELEMENT=MASS[,ELEMENT=MASS...] in atomic mass units, e.g. H=2.01410 "
                        "puts deuterium on every H site. Elements not listed keep their "
                        "standard atomic weight. WARP writes infile.isotopes in every step "
                        "directory and runs canonical_configuration and "
                        "phonon_dispersion_relations with --readiso. Default: natural masses.")
    p.add_argument("-met", "--metric", dest="metric", default="frac_rmsd",
                   choices=["frac_rmsd", "max_comp_frac", "pair_spectrum", "cart_rmsd", "max_atom"],
                   help="Metric comparing a step's input and updated unit cell (default "
                        "frac_rmsd); with --reuse_threshold it decides whether the previous "
                        "step's samples are merged into the current fit. frac_rmsd, max_comp_frac "
                        "and pair_spectrum are dimensionless (fractional coordinates), cart_rmsd "
                        "and max_atom are in Å.")
    p.add_argument("--fc1_mean_tol", "--fc1-mean-tol", dest="fc1_mean_tol",
                   type=float, default=0.030,
                   help="Position-loop exit: mean per-atom residual force |FC1| in eV/Å "
                        "(default 0.030). The loop ends when both this and --fc1_max_tol hold. "
                        "Cannot usefully be tighter than the calculator's own force error; "
                        "0.0026 corresponds to forc_conv_thr = 1e-4 Ry/bohr in Quantum ESPRESSO.")
    p.add_argument("--fc1_max_tol", "--fc1-max-tol", dest="fc1_max_tol",
                   type=float, default=0.075,
                   help="Position-loop exit: maximum per-atom residual force |FC1| in eV/Å "
                        "(default 0.075). Guards against a single stuck atom; keep it about "
                        "2.5 x --fc1_mean_tol so the mean stays the binding test.")
    p.add_argument("-rt", "--reuse_threshold", "--reuse-threshold", dest="reuse_threshold",
                   type=float, default=0.002,
                   help="Threshold on --metric (default 0.002; fractional or Å depending on the "
                        "metric) below which a step's structure update counts as small and its "
                        "samples are merged into the next fit. Not applied to the first position "
                        "step after a lattice update, which always samples fresh.")

    # --- bootstrap scale (used only while state.use_fake_fc is True) ---
    p.add_argument("-mf", "--max_freq", "--maximum_frequency", dest="max_freq",
                   type=int, default=None,
                   help="Maximum phonon frequency in THz (integer) for the geom_0 bootstrap, "
                        "passed as canonical_configuration --maximum_frequency while sampling "
                        "from the bootstrap (Debye-model) force constants. Default: not passed.")
    p.add_argument("-Td", "--Tdebye", "--debye_temperature", dest="Tdebye",
                   type=int, default=None,
                   help="Debye temperature in K (integer) for the geom_0 bootstrap, passed as "
                        "canonical_configuration --debye_temperature while sampling from the "
                        "bootstrap (Debye-model) force constants. Default: not passed.")

    # --- experimental; suppressed from --help ---
    p.add_argument("--wz_supercell", "--wz-supercell", dest="wz_supercell",
                   default=None, help=argparse.SUPPRESS)

    # --- preconditioning schedule (optional override) ---
    p.add_argument("-ps", "--precond_schedule", "--precond-schedule", dest="precond_schedule",
                   default=None,
                   help="Comma-separated sample counts for the preconditioning geoms, e.g. "
                        "'20,40,40,60,60' (default 8,32,32,48,48,64,64). Multiples of the "
                        "MAX_CONC / MAX_CONC_MLIP concurrency (environment variables) avoid idle "
                        "processes.")

    # --- first-order structure update (Wyckoff-projected) stability controls ---
    p.add_argument("-wd", "--wyckoff_damping", "--wyckoff-damping", dest="wyckoff_damping",
                   type=float, default=0.8,
                   help="Fraction of the TDEP first-order displacement applied per structure "
                        "update (default 0.8). Lower it to 0.1-0.3 if the structure runs away.")
    p.add_argument("-ld", "--lattice_damping", "--lattice-damping", dest="lattice_damping",
                   type=float, default=1.0,
                   help="Damping factor lambda_l of the lattice update, eps = -lambda_l * S "
                        "(sigma_int - sigma_ext); default 1.0 = the full elastic step. "
                        "Multiplied further only by --lattice_damping_ladder and "
                        "--lattice_noise_gain when those are given.")
    p.add_argument("-ec", "--elastic_constants", "--elastic-constants", dest="elastic_constants",
                   default=None,
                   help="File with a 6x6 elastic-constant (stiffness) tensor in GPa, Voigt order "
                        "xx yy zz yz xz xy, in the Cartesian frame of the unit cell (-ucp); '#' "
                        "lines are comments. Its inverse is the compliance S of every lattice "
                        "update, in place of TDEP's outfile.compliance_tensor, which is the "
                        "clamped-ion compliance of the fitted force constants (too stiff where "
                        "internal coordinates relax under strain, e.g. along c in layered "
                        "crystals). The file is checked (6x6, symmetric, positive definite) "
                        "before the run starts. Default: outfile.compliance_tensor.")
    p.add_argument("--lattice_damping_ladder", "--lattice-damping-ladder", dest="lattice_damping_ladder",
                   default=None,
                   help="Optional adaptive ladder 'thr1:f1,thr2:f2,...' with thresholds in GPa: "
                        "when the largest compared component of |sigma_int - sigma_ext| is below "
                        "thr, the step is lambda_l * f; the rung with the smallest satisfied "
                        "threshold wins. Default: no ladder, the damping is always lambda_l.")
    p.add_argument("--lattice_noise_gain", "--lattice-noise-gain", dest="lattice_noise_z0", type=float, default=0.0,
                   help="Noise-referenced lattice damping: lambda_l is multiplied by "
                        "z^2/(z^2+Z0^2) with z = max_i |sigma_int,i - sigma_ext,i| / SE_i and "
                        "SE_i the standard error of the sample-mean stress from infile.stat "
                        "(warpt/noise_gain.py). The step stays full far from convergence and "
                        "shrinks smoothly once the residual is within a few SE. 0 = off (default). "
                        "Recommended 3.0.")
    p.add_argument("--pct_noise_floor", "--pct-noise-floor", dest="press_conv_noise_k", type=float, default=0.0,
                   help="Per-component noise floor on the stress gate: component i must satisfy "
                        "|sigma_int,i - sigma_ext,i| < max(--press_conv_th, K x SE_i), with SE_i "
                        "the standard error of the sample-mean stress from infile.stat (the three "
                        "shears share their largest SE). Also applied to the phonon-stress change "
                        "gate. 0 = off (default). Recommended 2.0.")
    p.add_argument("--fc3_gate_thr", "--fc3-gate-thr", dest="fc3_gate_thr", type=float, default=None,
                   help="Experimental consistency gate on the lattice step, in GPa (default: "
                        "off). The real-space third-order stress sigma_FC3 must cancel the q-space "
                        "sigma_ph (healthy: a few tenths of a GPa). If max|sigma_FC3 + sigma_ph| "
                        "exceeds this threshold the third-order fit is treated as ill-conditioned, "
                        "the step's files are archived in fc3_gate_attempt_<k>/ and the step is "
                        "resampled (see --fc3_gate_max_retries, --fc3_gate_nscale). Each retry "
                        "costs one extra sampling of the lattice step.")
    p.add_argument("--fc3_gate_max_retries", "--fc3-gate-max-retries", dest="fc3_gate_max_retries",
                   type=int, default=2,
                   help="Maximum number of resamples per lattice step when --fc3_gate_thr trips "
                        "(default 2). After the last retry the attempt with the smallest mismatch "
                        "is used.")
    p.add_argument("--fc3_gate_nscale", "--fc3-gate-nscale", dest="fc3_gate_nscale", type=float, default=1.0,
                   help="Sample-count multiplier per FC3-gate retry: retry k samples "
                        "round(n * nscale**k) configurations (default 1.0 = same n; 2.0 doubles "
                        "the set each retry to reduce noise).")
    p.add_argument("-fomf", "--firstorder_min_freq", "--firstorder-min-freq",
                   dest="firstorder_min_freq", type=float, default=-1.0,
                   help="Preconditioning only: from geom_2 on, the first-order structure update "
                        "(extract_forceconstants --firstorder) is fitted only when min(freq) of "
                        "the previous geom's dispersion exceeds this value in THz (default -1.0); "
                        "otherwise the geom refits FC2/FC3 by sampling alone, because a Newton "
                        "step on an indefinite FC2 Hessian points uphill along imaginary "
                        "branches. Position and lattice steps always fit with --firstorder.")

    # --- calculator selection ---
    # Force/stress calculator (default QE). QE runs --pwx (default "mpirun -n 6 pw.x"); MLIP runs
    # "python3 -m warpt.MLIP_interface --engine <ml_engine> --device <ml_device>
    # --ml_params <ml_params>"; VASP runs "python3 -m warpt.VASP_interface
    # --template <vasp_template>".
    p.add_argument("-c", "--calculator", dest="calculator",
                   choices=["QE", "MLIP", "VASP"], default="QE",
                   help="Force engine for the sampled configurations (default QE). QE runs "
                        "--pwx; MLIP runs python3 -m warpt.MLIP_interface with --ml_engine, "
                        "--ml_params, --ml_device, --ml_variant and --ml_conservative; VASP runs "
                        "python3 -m warpt.VASP_interface with --vasp_template and --vasp_cmd.")
    p.add_argument("-pwx", "--pwx", dest="pwx", default=None,
                   help='Command that runs one SCF when --calculator QE (default '
                        '"mpirun -n 6 pw.x"); "-input <file>" is appended. Ignored with '
                        '--calculator MLIP or VASP. A "python3 -m warpt.MLIP_interface ..." '
                        'command may be given here to pass MLIP_interface options that -c MLIP '
                        'does not expose.')

    # --- MLIP options (used only with --calculator MLIP) ---
    # MLIP engine forwarded as --engine to warpt.MLIP_interface (default upet).
    p.add_argument("-mle", "--ml_engine", dest="ml_engine",
                   choices=["upet", "mace"], default="upet",
                   help="MLIP engine, forwarded as --engine to warpt.MLIP_interface "
                        "(default upet). Used only with --calculator MLIP.")
    p.add_argument("-mlp", "--ml_params", dest="ml_params", default=None,
                   help="Path to the MLIP parameters (UPET .ckpt or MACE .model), forwarded as "
                        "--ml_params to warpt.MLIP_interface. Required with --calculator MLIP.")
    # Torch device forwarded as --device to warpt.MLIP_interface (default cpu).
    p.add_argument("-mld", "--ml_device", dest="ml_device",
                   choices=["cpu", "cuda"], default="cpu",
                   help="Device, forwarded as --device to warpt.MLIP_interface (default cpu). "
                        "Used only with --calculator MLIP.")
    # Fine-tuned UPET heads: without the variant, UPETCalculator silently evaluates the
    # untuned base head of the checkpoint.
    p.add_argument("-mlv", "--ml_variant", dest="ml_variant", default=None,
                   help="UPET energy head, forwarded as --variant to warpt.MLIP_interface, e.g. "
                        "'lah' for a checkpoint fine-tuned on the head 'energy/lah' (default: none, "
                        "the checkpoint's base head). Used only with --calculator MLIP.")
    p.add_argument("-mlc", "--ml_conservative", dest="ml_conservative", action="store_true",
                   help="Forwarded as --conservative to warpt.MLIP_interface: UPET forces and "
                        "stress as energy gradients instead of the direct heads (default: off). "
                        "Used only with --calculator MLIP.")

    # --- VASP options (required with --calculator VASP) ---
    p.add_argument("--vasp_template", dest="vasp_template", default=None,
                   help="Directory with INCAR (single point: NSW = 0, ISIF >= 2), KPOINTS for the "
                        "SUPERCELL (unless INCAR sets KSPACING) and one POTCAR_<element> per "
                        "element. Required with --calculator VASP.")
    p.add_argument("--vasp_cmd", dest="vasp_cmd", default=None,
                   help="Command that runs VASP in a sample directory, e.g. 'srun vasp_std' or "
                        "'mpirun -np 64 vasp_std'. Required with --calculator VASP. Samples run "
                        "MAX_CONC at a time (default 1 for VASP).")

    return p.parse_args()


def _select_pw_command(args) -> str:
    """Return the calculator command string stored as ``ctx.pwscf_run``.

    ``QE``: ``args.pwx`` or ``"mpirun -n 6 pw.x"``. ``MLIP``:
    ``python3 -m warpt.MLIP_interface --engine E --device D --ml_params P``, plus
    ``--variant V`` when ``--ml_variant`` is given and ``--conservative`` when
    ``--ml_conservative`` is set. ``P`` is made absolute because the samples run in
    per-geometry subdirectories. ``VASP``: ``python3 -m warpt.VASP_interface
    --template T`` with ``T`` made absolute likewise, the VASP command exported as
    ``WARPT_VASP_CMD`` and ``MAX_CONC`` defaulting to 1. Calls
    :func:`~warpt.utilities.die` (exit 1) when ``--ml_params`` is missing for ``MLIP``
    or ``--vasp_template`` or ``--vasp_cmd`` is missing for ``VASP``.
    """
    if args.calculator == "QE":
        return args.pwx or "mpirun -n 6 pw.x"

    if args.calculator == "VASP":
        if not args.vasp_template or not args.vasp_cmd:
            die("--vasp_template and --vasp_cmd are required when --calculator VASP")
        # The runner splits the calculator command on whitespace, so the VASP command
        # (which has spaces) travels in the environment the sample processes inherit.
        os.environ["WARPT_VASP_CMD"] = args.vasp_cmd
        # One VASP run at a time unless MAX_CONC is set: each run is already parallel.
        os.environ.setdefault("MAX_CONC", "1")
        return f"python3 -m warpt.VASP_interface --template {os.path.abspath(args.vasp_template)}"

    if not args.ml_params:
        die("--ml_params is required when --calculator MLIP")

    # Invoke as a module so the package-relative imports in MLIP_interface
    # resolve (requires warpt to be importable, i.e. installed / on PYTHONPATH).
    cmd = (
        f"python3 -m warpt.MLIP_interface "
        f"--engine {args.ml_engine} "
        f"--device {args.ml_device} "
        f"--ml_params {os.path.abspath(args.ml_params)}"
    )
    if args.ml_variant:
        cmd += f" --variant {args.ml_variant}"
    if args.ml_conservative:
        cmd += " --conservative"
    return cmd


def _parse_sigma_ext(press_ext):
    """Convert ``--press_ext`` into a tension-positive Voigt external stress.

    Input values are compression-positive pressures in GPa. One value gives the
    hydrostatic target ``[-P, -P, -P, 0, 0, 0]`` (mode ``"scalar"``); three give
    the negated diagonal (``"diag3"``); six give the negated Voigt vector
    (``"full6"``).

    Parameters
    ----------
    press_ext : list[float]
        One, three or six values in GPa.

    Returns
    -------
    tuple[str, numpy.ndarray]
        ``(sigma_mode, sigma_ext)``; ``sigma_ext`` is a length-6 Voigt vector in GPa.

    Raises
    ------
    Calls :func:`~warpt.utilities.die` (exit 1) when ``press_ext`` is None or has a number
    of values other than 1, 3 or 6.
    """
    if press_ext is None:
        die("vc-relax requires --press_ext")

    if not isinstance(press_ext, (list, tuple)):
        press_ext = [press_ext]

    if len(press_ext) == 1:
        sigma_mode = "scalar"
        v = float(press_ext[0])
        sigma_ext = np.array([-v, -v, -v, 0.0, 0.0, 0.0], float)
    elif len(press_ext) == 3:
        sigma_mode = "diag3"
        sigma_ext = np.array([-press_ext[0], -press_ext[1], -press_ext[2], 0.0, 0.0, 0.0], float)
    elif len(press_ext) == 6:
        sigma_mode = "full6"
        sigma_ext = -1.0 * np.array(press_ext, float)
    else:
        die("--press_ext must have 1, 3, or 6 components")

    return sigma_mode, sigma_ext


def build_ctx_state(args):
    """Build the frozen :class:`WarpContext` and the initial :class:`WarpState`.

    The working directory becomes ``ctx.root``; ``-m`` and ``-ucp`` are joined to
    it (the unit-cell file is read here only for ``--calculator VASP`` and is not
    otherwise checked for existence). Picks ``mawk`` when it is on PATH, else ``awk``
    (the choice is printed as a ``WARNING: Using ...`` line), selects the calculator
    command, parses the supercell flags and the external stress, splits
    ``--precond_schedule`` at commas into integers (empty entries skipped; ``None``
    when the option is absent), and bundles the physical constants used by
    :mod:`warpt.phonon_stress`. For ``--calculator VASP`` the template directory is
    checked against the elements of the unit cell
    (:func:`~warpt.VASP_interface.check_template`); each warning it returns is printed
    as a WARNING line, followed by a ``[warpt] VASP:`` line with the template, the
    command and ``MAX_CONC``.

    Parameters
    ----------
    args : argparse.Namespace
        Output of :func:`parse_arguments`.

    Returns
    -------
    tuple[WarpContext, WarpState]

    Raises
    ------
    Calls :func:`~warpt.utilities.die` (exit 1) when the master deck does not
    exist, ``--press_ext`` is absent, ``--ml_params`` is missing for ``--calculator MLIP``, ``--vasp_template``
    or ``--vasp_cmd`` is missing for ``--calculator VASP``, the VASP template check
    fails (an unreadable unit cell included), ``--press_ext`` has a count other
    than 1, 3 or 6, or ``--supercell`` has a count other than 1, 3 or 9.
    ``ValueError`` from ``int()`` on a non-integer ``--precond_schedule`` or
    ``--supercell`` entry.
    """
    cwd = os.getcwd()
    awk_cmd = "mawk" if shutil.which("mawk") else "awk"
    print(f"WARNING: Using {awk_cmd}")

    pwscf_run = _select_pw_command(args)
    sigma_mode, sigma_ext = _parse_sigma_ext(args.press_ext)

    master_scf = os.path.join(cwd, args.masterscf)
    ucp = os.path.join(cwd, args.ucp)

    precond_schedule = None
    if args.precond_schedule:
        precond_schedule = [int(x) for x in str(args.precond_schedule).split(",") if x.strip()]

    if not os.path.isfile(master_scf):
        die(f"Master SCF file not found: {master_scf}")

    masses = parse_masses(args.masses) if args.masses else None
    if masses:
        print("[warpt] masses (amu): " + ", ".join(f"{k}={v:g}" for k, v in masses.items())
              + "; other elements at their standard atomic weight")

    elastic_constants = None
    if args.elastic_constants:
        from .compliance_update import read_elastic_constants
        elastic_constants = os.path.abspath(os.path.join(cwd, args.elastic_constants))
        try:
            C_user = read_elastic_constants(elastic_constants)
        except ValueError as exc:
            die(f"--elastic_constants: {exc}")
        print(f"[warpt] elastic constants from {elastic_constants}: diag C11..C66 = "
              + " ".join(f"{c:.1f}" for c in np.diag(C_user))
              + " GPa; their inverse replaces outfile.compliance_tensor in every lattice update")

    if args.calculator == "VASP":
        from ase.io import read as _read
        from .VASP_interface import check_template
        try:
            for w in check_template(args.vasp_template, _read(ucp, format="vasp").get_chemical_symbols()):
                print(f"WARNING: {w}")
        except Exception as exc:
            die(f"VASP template: {exc}")
        print(f"[warpt] VASP: template {os.path.abspath(args.vasp_template)}, "
              f"command '{args.vasp_cmd}', MAX_CONC={os.environ['MAX_CONC']}")

    ctx = WarpContext(
        root=cwd,
        master_scf=master_scf,
        ucp=ucp,

        awk_cmd=awk_cmd,
        pwscf_run=pwscf_run,

        temp=int(args.temp),
        num_samples=int(args.n),
        quantum=bool(args.quantum),
        masses=masses,

        cutoffFC2=float(args.rc2),
        cutoffFC3=None if args.rc3 is None else float(args.rc3),
        wz_supercell=args.wz_supercell,

        supercell=args.supercell,
        sc_opts=parse_supercell(args.supercell),

        metric=str(args.metric),
        reuse_threshold=float(args.reuse_threshold),
        fc1_mean_tol=float(args.fc1_mean_tol),
        fc1_max_tol=float(args.fc1_max_tol),

        max_freq=args.max_freq,
        T_debye=args.Tdebye,

        sigma_mode=sigma_mode,
        sigma_ext=np.array(sigma_ext, float),
        press_conv_th=float(args.press_conv_th),

        # constants needed by phonon_stress
        kB_eV=kB_eV,
        hbar_eVs=hbar_eVs,
        EV_TO_GPA_A3=EV_TO_GPA_A3,

        precond_schedule=precond_schedule,
        wyckoff_damping=float(args.wyckoff_damping),
        lattice_damping=float(args.lattice_damping),
        elastic_constants=elastic_constants,
        lattice_damping_ladder=args.lattice_damping_ladder,
        lattice_noise_z0=float(args.lattice_noise_z0),
        press_conv_noise_k=float(args.press_conv_noise_k),
        fc3_gate_thr=None if args.fc3_gate_thr is None else float(args.fc3_gate_thr),
        fc3_gate_max_retries=int(args.fc3_gate_max_retries),
        fc3_gate_nscale=float(args.fc3_gate_nscale),
        firstorder_min_freq=float(args.firstorder_min_freq),
    )

    state = WarpState(
        current_dir=cwd,
        lattice_changed=False,
        last_update_significant=False,
    )

    return ctx, state


def main():
    """Entry point: print the banner, parse arguments, build the context, then run
    ``run_vc_relax`` (after printing the resume plan) or ``run_preconditioning``
    according to ``-mode``.
    """
    print(BANNER, flush=True)
    args = parse_arguments()
    ctx, state = build_ctx_state(args)

    if args.mode == "vc-relax":
        plan = find_resume_plan(ctx)
        if plan["dir"] is None:
            print("[restart] No incomplete run detected; starting fresh")
        else:
            print("[restart] Detected previous incomplete run")
            print(f"[restart] iteration = {plan['iter']}")
            print(f"[restart] kind      = {plan['kind']}")
            print(f"[restart] dir       = {plan['dir']}")
            print(f"[restart] stage     = {plan['stage']}")
        run_vc_relax(ctx, state, resume_plan=plan)
        return

    if args.mode == "preconditioning":
        # run_preconditioning skips geoms that first_missing_stage reports complete.
        run_preconditioning(ctx, state)
        return

    die(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()