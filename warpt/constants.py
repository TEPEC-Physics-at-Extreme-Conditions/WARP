# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Physical constants and unit conversions used throughout the package.

Every physical constant and unit conversion factor that WARP defines is set here.
Two conversions use values from elsewhere: :mod:`warpt.MLIP_interface` converts the ASE
calculator's eV and Å to Ry and bohr with ``ase.units.Rydberg`` and ``ase.units.Bohr``
(ASE's CODATA 2014 values, within 1e-8 relative of the CODATA 2018 values here), and
the ``kbar_to_gpa`` argument of :func:`warpt.io_qe.read_qe_stress_gpa` defaults to the
literal 0.1, equal to ``KBAR_TO_GPA``.

Imported by :mod:`warpt.tdep_collect` (energy, force and stress conversion of the Quantum
ESPRESSO-style output, the stress factor being passed on to
:func:`warpt.io_qe.read_qe_stress_gpa`), :mod:`warpt.io_qe` (the kbar values of the stress
block written by :func:`warpt.io_qe.print_qe_like`), :mod:`warpt.VASP_interface` (VASP
energy, forces and stress to Quantum ESPRESSO units), :mod:`warpt.MLIP_interface`,
:mod:`warpt.structconv` (``BOHR_TO_ANG``, ``ANG_TO_BOHR``), :mod:`warpt.upet_relax`
(``EV_TO_GPA_A3``) and :mod:`warpt.orchestrator`, which stores ``kB_eV``, ``hbar_eVs``
and ``EV_TO_GPA_A3`` in the run context for :mod:`warpt.phonon_stress`.

All values are module-level names so they can be imported directly
(``from warpt.constants import RYBOHR3_TO_GPA``). The primaries are the exact values
of the 2019 SI redefinition (``E_CHARGE``, ``K_BOLTZMANN``, ``H_PLANCK``) and the
CODATA 2018 recommended values of the Rydberg energy and the Bohr radius
(``RY_TO_EV``, ``BOHR_TO_ANG``); every other conversion factor follows from them:

* ``kB_eV`` / ``hbar_eVs`` — Boltzmann constant (eV/K) and reduced Planck constant
  (eV s) for the Bose-Einstein occupation in the phonon stress.
* ``EV_TO_GPA_A3`` — eV/Å³ to GPa, written out as the exact decimal value of
  ``E_CHARGE / (1e-30 m³ · 1e9 Pa)``.
* ``RY_TO_EV`` — Rydberg to eV; Quantum ESPRESSO prints total energies in Ry.
* ``BOHR_TO_ANG`` / ``ANG_TO_BOHR`` — bohr to Å and its inverse.
* ``RYBOHR_TO_EV_ANG`` — Ry/bohr to eV/Å; Quantum ESPRESSO prints forces in Ry/bohr.
* ``RYBOHR3_TO_GPA`` — Ry/bohr³ to GPa; Quantum ESPRESSO prints stress in Ry/bohr³.
* ``KBAR_TO_GPA`` — kbar to GPa, exact by definition; through
  ``RYBOHR3_TO_GPA / KBAR_TO_GPA`` it gives the kbar column and the pressure of the
  stress block written by :func:`warpt.io_qe.print_qe_like` for the MLIP and VASP
  backends.
"""

import math

# --- primaries: exact SI 2019 ---
E_CHARGE    = 1.602176634e-19    # elementary charge, C (exact)
K_BOLTZMANN = 1.380649e-23       # Boltzmann constant, J/K (exact)
H_PLANCK    = 6.62607015e-34     # Planck constant, J s (exact)

# --- primaries: CODATA 2018 recommended values ---
RY_TO_EV    = 13.605693122994    # Rydberg energy, eV
BOHR_TO_ANG = 0.529177210903     # Bohr radius, Angstrom

# --- derived conversions ---
kB_eV            = K_BOLTZMANN / E_CHARGE                  # eV/K
hbar_eVs         = H_PLANCK / (2.0 * math.pi * E_CHARGE)   # eV s
EV_TO_GPA_A3     = 160.2176634                             # E_CHARGE / 1e-30 m^3 / 1e9 Pa, exact
ANG_TO_BOHR      = 1.0 / BOHR_TO_ANG                       # bohr per Angstrom
RYBOHR_TO_EV_ANG = RY_TO_EV / BOHR_TO_ANG                  # (Ry/bohr) -> eV/Angstrom
RYBOHR3_TO_GPA   = RY_TO_EV * EV_TO_GPA_A3 / BOHR_TO_ANG**3  # (Ry/bohr^3) -> GPa
KBAR_TO_GPA      = 0.1                                     # exact definition

__all__ = [
    "E_CHARGE",
    "K_BOLTZMANN",
    "H_PLANCK",
    "RY_TO_EV",
    "BOHR_TO_ANG",
    "kB_eV",
    "hbar_eVs",
    "EV_TO_GPA_A3",
    "ANG_TO_BOHR",
    "RYBOHR_TO_EV_ANG",
    "RYBOHR3_TO_GPA",
    "KBAR_TO_GPA",
]
