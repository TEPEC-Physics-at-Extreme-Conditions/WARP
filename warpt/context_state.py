# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Run configuration and mutable state containers.

:class:`WarpContext` is the frozen bundle of everything that defines a run
(paths, calculator command, temperature, cutoffs, supercell, external stress,
convergence and damping settings, physical constants). :class:`WarpState` holds
the state that changes as the vc-relax proceeds. Neither reads nor writes files.

Public classes: ``WarpContext``, ``WarpState``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

@dataclass(frozen=True)
class WarpContext:
    """Immutable run configuration shared by every pipeline stage.

    Paths: ``root`` (run root, the launch directory), ``master_scf``, ``ucp``.
    Calculator: ``pwscf_run`` (command that runs one SCF or MLIP evaluation),
    ``awk_cmd`` (``mawk`` or ``awk``).
    Sampling: ``temp`` (K), ``num_samples``, ``quantum``, ``masses`` (per-element masses
    in amu, None = natural), and for the geom_0 bootstrap ``max_freq`` (THz) and
    ``T_debye`` (K).
    Force constants: ``cutoffFC2`` and ``cutoffFC3`` (Å; ``None`` = no third order),
    ``wz_supercell``.
    Supercell: ``supercell`` (raw CLI strings), ``sc_opts`` (generate_structure flags).
    Position loop: ``metric``, ``reuse_threshold``, ``fc1_mean_tol`` and
    ``fc1_max_tol`` (eV/Å), ``wyckoff_damping``.
    Lattice step: ``sigma_mode``, ``sigma_ext`` (GPa, tension positive, Voigt),
    ``press_conv_th`` (GPa), ``lattice_damping``, ``elastic_constants``, ``lattice_damping_ladder``,
    ``lattice_noise_z0``, ``press_conv_noise_k``, ``fc3_gate_thr`` (GPa),
    ``fc3_gate_max_retries``, ``fc3_gate_nscale``.
    Constants: ``kB_eV``, ``hbar_eVs``, ``EV_TO_GPA_A3``.
    Preconditioning: ``precond_schedule`` overrides the sample schedule and
    ``firstorder_min_freq`` (THz) gates the first-order structure update from geom_2 on.
    """
    root: str
    master_scf: str
    ucp: str

    awk_cmd: str
    pwscf_run: str

    temp: int
    num_samples: int
    quantum: bool

    cutoffFC2: float
    cutoffFC3: Optional[float]


    supercell: Sequence[str]
    sc_opts: Sequence[str]

    metric: str
    reuse_threshold: float

    max_freq: Optional[int]
    T_debye: Optional[int]

    # vc-relax controls
    sigma_mode: str
    sigma_ext: np.ndarray
    press_conv_th: float

    kB_eV: float
    hbar_eVs: float
    EV_TO_GPA_A3: float

    # Optional override of the preconditioning sample schedule (--precond_schedule);
    # sample counts that are multiples of MAX_CONC / MAX_CONC_MLIP avoid idle
    # processes. None -> the built-in default [8, 32, 32, 48, 48, 64, 64].
    precond_schedule: Optional[Sequence[int]] = None

    # Fraction of the Wyckoff-projected TDEP first-order displacement applied per
    # structure update (--wyckoff_damping).
    wyckoff_damping: float = 0.8

    # Preconditioning only (geom_2 onward): the first-order structure update is
    # fitted only when the previous geom's min(freq) in THz exceeds this value;
    # see --firstorder_min_freq.
    firstorder_min_freq: float = -1.0

    # Position-loop exit tolerances on the FC1 residual force (eV/Å). A tolerance tighter
    # than the calculator's own force error is not useful; see --fc1_mean_tol.
    fc1_mean_tol: float = 0.030
    fc1_max_tol: float = 0.075

    # Damping of the lattice update (lambda_l); see --lattice_damping. Default 1.0 = full step.
    lattice_damping: float = 1.0
    # Absolute path of a user-supplied 6x6 stiffness (GPa, Voigt, Cartesian frame of the
    # unit cell) whose inverse replaces outfile.compliance_tensor in every lattice update
    # (--elastic_constants); None = TDEP's clamped-ion compliance.
    elastic_constants: Optional[str] = None
    # Optional adaptive ladder 'thr:f,thr:f' (GPa -> factor); None = no ladder.
    # See --lattice_damping_ladder.
    lattice_damping_ladder: Optional[str] = None
    # Noise-referenced damping gain z^2/(z^2+z0^2), z = |dsigma|/SE from infile.stat
    # (noise_gain.py). 0.0 = off (default); recommended 3.0. See --lattice_noise_gain.
    lattice_noise_z0: float = 0.0
    # Stress gate floor k*SE (0.0 = off; recommended 2.0). See --pct_noise_floor.
    press_conv_noise_k: float = 0.0

    # Optional FC3/sigma_ph consistency gate on the lattice step (--fc3_gate_thr;
    # experimental). None = off (default). When set (GPa), a lattice step whose
    # max|sigma_FC3 + sigma_ph| exceeds the threshold is archived in
    # fc3_gate_attempt_<k>/ and the step is resampled (fresh configurations, n
    # scaled by fc3_gate_nscale per retry) up to fc3_gate_max_retries times; the
    # attempt with the smallest mismatch is kept. Each retry costs one extra sampling.
    fc3_gate_thr: Optional[float] = None
    fc3_gate_max_retries: int = 2
    fc3_gate_nscale: float = 1.0

    # Experimental. When set (e.g. "5 5 5") the second-order fit uses TDEP's -wz
    # (Voronoi) cutoff instead of -rc2, except during preconditioning; see
    # tdep_extract_fc.py. Must remain among the defaulted fields (dataclass ordering).
    wz_supercell: Optional[str] = None

    # Atomic masses in amu per element for the thermal sampling and the phonons
    # (--masses); None = natural masses. When set, infile.isotopes is written in every
    # step directory and canonical_configuration and phonon_dispersion_relations run
    # with --readiso.
    masses: Optional[dict] = None




@dataclass
class WarpState:
    """Mutable run-time state carried across a vc-relax.

    ``current_dir``: directory of the stage being executed. ``lattice_changed``:
    set after a lattice update and cleared by the next position step, which then
    samples fresh instead of merging the previous samples.
    ``last_update_significant``: the most recent position step moved the
    structure by more than ``reuse_threshold``, which blocks sample merging in
    the following step. ``use_fake_fc``: sample from TDEP's fake force constants
    (geom_0 only). ``fc3_gate_events``: one ``(iteration, attempt, n_samples,
    mismatch GPa)`` tuple per FC3-gate evaluation. ``last_sigma_int`` and
    ``resume_vc_dir`` are declared but not written or read by any stage.
    :func:`~warpt.workflow_vc_relax.run_vc_relax` also attaches
    ``last_lattice_rel_change`` and ``prev_sigma_ph`` at run time.
    """
    current_dir: str
    lattice_changed: bool = False
    last_update_significant: bool = False

    last_sigma_int: Optional[np.ndarray] = None
    resume_vc_dir: Optional[str] = None
    use_fake_fc: bool = False

    # FC3-gate bookkeeping: (iteration, attempt, n_samples, mismatch GPa) tuples.
    fc3_gate_events: list = field(default_factory=list)