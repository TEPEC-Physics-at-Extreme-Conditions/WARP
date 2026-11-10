# Changelog

All notable changes to WARP are listed here. Versions follow [semantic versioning](https://semver.org/);
the version of an installed copy is printed in the run banner and available as `warpt.__version__`.

## 1.0.0 — 2026-11-10

First public release, accompanying the WARP paper
(H. R. A. ten Eikelder, F. Knoop and F. Trybel, [arXiv:2611.12736](https://arxiv.org/abs/2611.12736)).

- Symmetry-constrained variable-cell relaxation at finite temperature and pressure on the
  stochastic temperature-dependent effective potential, with anharmonic and zero-point contributions
  to the internal stress.
- Force engines: Quantum ESPRESSO, VASP and machine-learned interatomic potentials (UPET, MACE).
- `tdep_warp/`: the TDEP version WARP runs on (upstream commit `dbc2b47` plus `ifc_pressure` and
  the other additions listed in `tdep_warp/WARP_NOTES.md`).
- Released under GPL-3.0-or-later.
