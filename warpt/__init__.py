# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""WARP — a TDEP-based quantum/anharmonic variable-cell relaxation driver.

WARP automates a self-consistent vc-relax loop that relaxes a crystal structure at
finite temperature and pressure, including nuclear-quantum and anharmonic effects,
using TDEP effective force constants and a calculator backend (Quantum ESPRESSO,
VASP or a machine-learned interatomic potential). The package is organised as:

* :mod:`warpt.orchestrator` — CLI entry point and run setup.
* :mod:`warpt.workflow_vc_relax`, :mod:`warpt.workflow_preconditioning` — the
  outer relaxation loop and the FC-preconditioning bootstrap.
* :mod:`warpt.tdep_sampling`, :mod:`warpt.tdep_collect`,
  :mod:`warpt.tdep_extract_fc` — sampling, parsing and force-constant fitting.
* :mod:`warpt.phonon_stress`, :mod:`warpt.compliance_update`,
  :mod:`warpt.tensors` — the internal-stress / elastic-compliance lattice update.
* :mod:`warpt.Wyckoff_Projector` — symmetry projection of the structure update.
* I/O and support: :mod:`warpt.io_poscar`, :mod:`warpt.io_qe`,
  :mod:`warpt.io_workflow`, :mod:`warpt.runner`, :mod:`warpt.metrics`,
  :mod:`warpt.validity`, :mod:`warpt.restart_check`, :mod:`warpt.resume`,
  :mod:`warpt.context_state`, :mod:`warpt.constants`, :mod:`warpt.utilities`,
  :mod:`warpt.resources`, :mod:`warpt.timing`.
* :mod:`warpt.noise_gain` — standard error of the sampled stress; noise-referenced
  damping gain and convergence floor.
* :mod:`warpt.MLIP_interface` — the UPET/MACE calculator backend with QE-like output.
* :mod:`warpt.VASP_interface` — the VASP calculator backend with QE-like output.
* Standalone tools: :mod:`warpt.structconv` (POSCAR <-> Quantum ESPRESSO
  conversion), :mod:`warpt.get_frames` (collect a run's structures into one
  trajectory), :mod:`warpt.trajectory` (per-step tracking of a run),
  :mod:`warpt.plot_phonons_iter` (the phonon dispersion of every step),
  :mod:`warpt.band_yaml` (phonopy ``band.yaml`` from a step's dispersion),
  :mod:`warpt.upet_relax` (static symmetry-constrained MLIP relaxation),
  :mod:`warpt.static_elastic` (static elastic tensor of a potential for ``--elastic_constants``).

Importing the package reads and writes nothing; submodules are imported explicitly.
The only name defined here is ``__version__``, the single source of the version: the
banner printed at the start of every run (and above the ``--help`` text) shows it, and
``pyproject.toml`` reads it at build time.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
