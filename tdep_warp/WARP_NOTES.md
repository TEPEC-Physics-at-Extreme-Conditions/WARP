# tdep_warp: the TDEP version WARP runs on

This folder is a copy of TDEP with the additions WARP needs. Upstream TDEP is MIT-licensed
(© Olle Hellman and friends, see `LICENSE.md`); the license and the upstream files are kept as they are.

- **Upstream:** https://github.com/tdep-developers/tdep, commit `dbc2b47`
  ("Merge pull request #164 from MaxMignolet/improveMeson").
- **Left out of this copy:** `examples/` (except the build templates in `examples/build/`), `paper/` and `.github/` (unchanged from upstream; see the upstream repository),
  and all build output and machine-specific files (`bin/`, `build/`, `lib/`, `inc/`, `important_settings`,
  `src/*/Makefile.inc`).

## What is different from upstream

**New program**

- `src/ifc_pressure/` — `ifc_pressure`: the stress from the sampled DFT forces and from the second- and third-order
  force constants, written as `outfile.stress_dft`, `outfile.stress_fc2`, `outfile.stress_fc3` and `outfile.stress_res`, plus the
  elastic compliance tensor of the fitted force constants (`outfile.compliance_tensor`).
  WARP's lattice update assembles its internal stress from these files.

**Changes to existing programs and the library**

| file | change |
|---|---|
| `src/extract_forceconstants/main.f90`, `Makefile`, new `symmetrize.f90` | first-order step: the mass-weighted right-hand side is built from the raw first-order forces, and the resulting unit-cell displacement is written to `outfile.uc_displacements` (read by WARP's position update); `symmetrize.f90` (force/stress symmetrization) is compiled in |
| `src/libolle/type_forceconstant_secondorder.f90` | the elastic compliance tensor (inverse of the stiffness) is computed and stored (`elastic_compliances_voigt`) |
| `src/libolle/type_forceconstant_thirdorder.f90` | additions used for the anharmonic (third-order) stress |
| `src/libolle/type_phonon_dispersions.f90`, `type_phonon_dispersions_generation.f90` | the anisotropic Grüneisen tensor (`gruneisen_tensors`) |
| `src/phonon_dispersion_relations/main.f90` | `--gruneisen` computes the mode Grüneisen parameters and the Grüneisen tensor |
| `src/canonical_configuration/main.f90`, `options.f90` | `--readiso`: thermal displacements with the mean masses of `infile.isotopes` instead of the natural masses (isotope sampling, e.g. deuterium) |
| `src/libolle/type_symmetryoperation.f90` | robustness fix (2026-10-28): products of symmetry operations are checked before they are stored, and operations that fail the strict tolerance checks are dropped with a WARNING instead of stopping the run ("Invalid operation. Could never happen") |
| `build_things.sh` | builds `ifc_pressure` |

## Building

Build with `build_things.sh` exactly as upstream TDEP (see `INSTALL.md`): copy one of the example settings to
`important_settings`, adjust compilers and library paths, run `bash build_things.sh`. The binaries land in `bin/`;
put that directory on `PATH` before running WARP.
The Meson build (`meson.build`) has **not** been extended to `ifc_pressure`; use `build_things.sh`.
