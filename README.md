<p align="center"><img src="docs/assets/warp_icon.png" alt="WARP" width="180"></p>

# WARP — Wyckoff-constrained Anharmonic Relaxation at Pressure and temperature

WARP relaxes the atomic positions and the cell of a crystal at finite temperature and pressure on
the stochastic temperature-dependent effective potential (s-TDEP). The thermal ensemble is sampled,
the mean force is driven to zero and the full internal stress (anharmonic contributions included)
is driven to a target stress. The phonon part of that stress always carries the Bose–Einstein
occupation with its zero-point term; `--quantum` switches the sampling itself from classical to
Bose–Einstein statistics, adding zero-point motion to the sampled ensemble. Every update is projected
onto the Wyckoff parameters and the symmetric strains of the current space group, so the symmetry
of a structure can rise but never fall. Forces come from Quantum ESPRESSO, VASP or a
machine-learned interatomic potential (UPET, MACE).

The Python package is `warpt` (the plain name `warp` is taken by NVIDIA Warp, a dependency of UPET).

**Manual:** <https://tepec-physics-at-extreme-conditions.github.io/WARP/>  
**Cite WARP:** <https://arxiv.org/abs/2611.12736>

## Requirements
- Python ≥ 3.10 with `numpy`, `spglib`, `h5py` and `ase` (all installed with the package)
- the TDEP binaries on `PATH`: `generate_structure`, `canonical_configuration`,
  `extract_forceconstants`, `ifc_pressure`, `phonon_dispersion_relations`.
  The TDEP version must provide `ifc_pressure` (stress decomposition and compliance tensor; a
  lattice step cannot run without it) and the pressure-adjusted force-constant extraction
  (included in this repository as [`tdep_warp/`](tdep_warp/WARP_NOTES.md): upstream TDEP plus these additions)
- a force engine: `pw.x` (Quantum ESPRESSO), a VASP executable, or a UPET/MACE model

## Install
Clone the repository, build the TDEP version in [`tdep_warp/`](tdep_warp/WARP_NOTES.md), and install WARP:

```bash
# 1. Get the code
git clone https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP.git
cd WARP

# 2. Build the TDEP version WARP runs on (needs a Fortran compiler, MPI, BLAS/LAPACK, FFTW
#    and HDF5 with Fortran support)
cd tdep_warp
cp examples/build/important_settings.gfortran important_settings   # pick the template closest
                                                                   # to your machine; edit its paths
bash build_things.sh --nthreads_make 4
source bashrc_tdep            # puts tdep_warp/bin on PATH; add this line to ~/.bashrc
cd ..

# 3. Install WARP
python3 -m pip install -e .
python3 -m warpt --help       # prints the banner and every option
```

Upstream TDEP will not do: WARP needs `ifc_pressure` and the other additions in `tdep_warp/`.
For a machine-learned force engine, install an extra instead of the plain package in step 3:
```bash
python3 -m pip install -e ".[upet]"       # adds upet
python3 -m pip install -e ".[mace]"       # adds mace-torch, and upet, which the MLIP interface imports
```
The [Installation](https://tepec-physics-at-extreme-conditions.github.io/WARP/installation/) page
of the manual has the details and the checks.

## Run
```bash
python3 -m warpt --calculator QE -pwx "mpirun -n 8 pw.x" \
    -m master.scf.in -ucp start.vasp --mode vc-relax \
    -t 300 --press_ext 0.0 -sc 2 2 2 -Td 400 \
    -n 64 -rc2 5.0 --rc3 3.5 -pct 0.05 --quantum
```
The other force engines replace `--calculator QE -pwx ...`:

- VASP: `-c VASP --vasp_template DIR --vasp_cmd CMD` (`DIR` holds a single-point `INCAR`,
  `KPOINTS` unless `INCAR` sets `KSPACING`, and one `POTCAR_<element>` per element; `CMD` is e.g.
  `"srun vasp_std"`);
- a machine-learned potential: `-c MLIP -mlp model.ckpt -mlv <head> -mlc` (`-mle mace` selects
  MACE, `-mld cuda` a GPU; `-mlv` names the fine-tuned energy head of a UPET checkpoint, without
  which the base head of the checkpoint is evaluated).

`-m` names a Quantum ESPRESSO input deck for every engine. `python3 -m warpt --help` lists every
option. A run resumes from its directory when it is started again with the same command.

## Modules
`orchestrator` (CLI and run setup), `workflow_preconditioning` and `workflow_vc_relax` (the
loops), `tdep_sampling`, `tdep_extract_fc`, `tdep_collect`, `phonon_stress`, `compliance_update`,
`tensors`, `noise_gain`, `Wyckoff_Projector`, `context_state`, `resume`, `restart_check`,
`io_poscar`, `io_qe`, `io_workflow`, `runner`, `validity`, `metrics`, `constants`, `utilities`,
`resources`, `timing`, `static_elastic` (the static elastic tensor of a potential, for `--elastic_constants`), `structconv` (POSCAR ↔ Quantum ESPRESSO conversion, also runnable on its
own; atoms are never reordered); the force backends `MLIP_interface` and `VASP_interface`; and the
standalone tools `upet_relax` (static symmetry-constrained relaxation on a UPET potential),
`get_frames`, `trajectory`, `plot_phonons_iter` and `band_yaml` (analysis of a run). The [module reference](https://tepec-physics-at-extreme-conditions.github.io/WARP/reference/)
documents each one.

## Citing
**Cite WARP:** <https://arxiv.org/abs/2611.12736>

> H. R. A. ten Eikelder, F. Knoop and F. Trybel,
> *Finite temperature and pressure structural relaxations including zero-point motion and anharmonicity*,
> arXiv:2611.12736 (2026), <https://arxiv.org/abs/2611.12736>.

Cite the TDEP papers as well; [Citing WARP](docs/citing.md) lists them, with BibTeX entries.

## Licence
WARP is free software, released under the [GNU General Public License, version 3 or later](https://www.gnu.org/licenses/gpl-3.0.html)
(GPL-3.0-or-later); see [`LICENSE`](LICENSE). The copyright is held by the developers listed in
[`NOTICE`](NOTICE), which also lists the contributors and the third-party parts of the repository: the
bundled TDEP copy in `tdep_warp/` keeps its MIT licence, and the packaged symmetry table keeps its
CC BY 4.0 licence.
