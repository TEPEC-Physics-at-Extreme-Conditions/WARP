# Installation

WARP is a Python package that drives external programs: the TDEP binaries do the sampling and the
force-constant fitting, and a separate force engine computes energies, forces and stresses. This
page lists what must be installed, how to install the package itself, and how to check the result.

## In short

Clone the repository, build the TDEP version shipped in it, and install WARP:

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

The rest of this page explains each step and how to check the result. The TDEP build is described
in [TDEP](#tdep) below, the package install in [Installing the package](#installing-the-package).

## The import name

The package is called `warpt`, not `warp`: the plain name belongs to NVIDIA Warp. The import, the
module command and the console script all use the longer name — `import warpt`,
`python3 -m warpt`, `warpt`.

## Python requirements

`pyproject.toml` requires **Python 3.10 or newer** and these packages:

| Package | Minimum | Needed for |
| --- | --- | --- |
| `numpy` | 1.24 | everything |
| `spglib` | 2.0 | space groups, Wyckoff orbits, strain projection |
| `h5py` | 3.9 | reading the TDEP dispersion and Grüneisen-grid files |
| `ase` | 3.22 | reading Quantum ESPRESSO decks and outputs (`warpt.io_qe`, which every run imports), the VASP and MLIP interfaces, and the analysis tools |

`matplotlib` is needed only for the figure of `warpt.trajectory`, which is skipped with a message
when it is missing.

Two optional extras add a machine-learned force engine:

```bash
python3 -m pip install -e ".[upet]"         # upet >= 0.2
python3 -m pip install -e ".[mace]"         # mace-torch >= 0.3.10 and upet >= 0.2
```

!!! note "The MACE engine also needs `upet`"
    `warpt.MLIP_interface` imports the UPET calculator when it starts, whichever engine is
    selected, so the `mace` extra installs the `upet` package as well. The UPET engine runs
    without `mace-torch`.

## External programs

### TDEP

WARP launches five TDEP binaries by name, so they must be on the `PATH`:

| Binary | Used for |
| --- | --- |
| `generate_structure` | building `infile.ssposcar` from the unit cell and the supercell flags |
| `canonical_configuration` | drawing the thermal configurations of every step |
| `extract_forceconstants` | fitting the second-, third- and first-order force constants |
| `ifc_pressure` | the stress decomposition and the compliance tensor of a fitted model |
| `phonon_dispersion_relations` | the phonon dispersion, the DOS gate and the Grüneisen grid |

`ifc_pressure` is the tool that writes `outfile.stress_dft`, `outfile.stress_fc2`,
`outfile.stress_fc3`, `outfile.stress_res` and `outfile.compliance_tensor`; WARP reads all of them
and cannot run a lattice step without it.

The upstream TDEP release contains neither `ifc_pressure` nor the pressure-adjusted force-constant
extraction that WARP relies on. Both are in the fork of TDEP maintained alongside WARP:

- **TDEP with `ifc_pressure`:** the `tdep_warp/` folder of this repository (upstream TDEP at commit `dbc2b47`
  plus the WARP additions; `tdep_warp/WARP_NOTES.md` lists every change).

The fork also gives `canonical_configuration` a `--readiso` switch, which `--masses` relies on:
upstream `canonical_configuration` always samples with the natural masses.

It is built in the same way as upstream TDEP, described in the
TDEP documentation (`tdep_warp/INSTALL.md`, or https://tdep-developers.github.io/tdep/):

1. Install the build requirements: a Fortran compiler recent enough for Fortran 2008, MPI,
   BLAS/LAPACK, FFTW and HDF5 **with Fortran support** (from a package manager, or as modules on
   a cluster).
2. In `tdep_warp/`, copy the template in `examples/build/` that is closest to your machine to
   `important_settings` (`important_settings.gfortran` for a Linux workstation,
   `important_settings.apple_silicon_gfortran12` or `important_settings.osx_gfortran_accelerate`
   on a Mac, `important_settings.conda` in a conda environment, or one of the cluster
   templates), and set the compiler and the library and include paths in it.
3. Run `bash build_things.sh --nthreads_make 4`. It builds every binary, `ifc_pressure` included,
   into `tdep_warp/bin/` and writes `tdep_warp/bashrc_tdep`, which puts that directory on the
   `PATH`; `source` it, and add the line to `~/.bashrc` so that every new shell finds the binaries.

The build reads the git revision of the checkout, so build from a clone of the repository rather
than from a downloaded archive. Use `build_things.sh`, not the Meson build: the Meson build does
not include `ifc_pressure`.

### A force engine

Choose one:

- **Quantum ESPRESSO** — `pw.x`. WARP runs the command given in `--pwx` (default
  `mpirun -n 6 pw.x`) and appends `-input <file>`, so `mpirun`, `srun` and wrapper scripts all
  work. The command is split on whitespace and started without a shell, so quotes, pipes and
  variable expansions inside it are not interpreted.
- **VASP** — a licensed VASP executable, started per sample by `python3 -m warpt.VASP_interface`
  with the command given in `--vasp_cmd`. See [Force engines](forces.md#vasp).
- **A machine-learned potential** — the `upet` or `mace-torch` stack from the extras above, run
  through `python3 -m warpt.MLIP_interface`. See [Force engines](forces.md).

The VASP and machine-learned backends are started as `python3 -m warpt.<module>` in separate
processes, so the `python3` found on the `PATH` must be an interpreter that can import `warpt`.

### Shell tools

The collector extracts energies and forces from the calculator outputs, and positions from the
per-sample inputs, with `grep` and `awk`. WARP prefers `mawk` when it is on the `PATH` and falls
back to `awk`; the choice is printed as the first line after the banner, as `WARNING: Using mawk`
or `WARNING: Using awk`.

### Visualisation (optional)

[CrysViz](https://crysviz.org) displays the structures and trajectories that WARP writes, for
example the `trajectory.xyz` produced by `python3 -m warpt.get_frames` from a run directory, and
animates phonon modes from a phonopy `band.yaml`, which `python3 -m warpt.band_yaml` writes for any
step. It is not needed to run WARP. Figures made with it should cite the CrysViz paper (in preparation; see
[Citing WARP](citing.md#crysviz)).

## Installing the package

From the top of the repository (the `WARP` directory of the clone):

```bash
python3 -m pip install -e .
```

or, for a machine-learned force engine, `python3 -m pip install -e ".[upet]"` or
`".[mace]"` (see [Python requirements](#python-requirements)).

An editable install is the usual choice, because it keeps the running code and the checked-out
source in step.

## Checking the install

```bash
python3 -m warpt --help
```

The banner and the full option list should appear. `pyproject.toml` also declares a console
script, so

```bash
warpt --help
```

is equivalent once the package is installed; the examples in this manual use `python3 -m warpt`
throughout, because it works from a source checkout on `PYTHONPATH` as well.

Check that the external programs answer too:

```bash
command -v generate_structure canonical_configuration extract_forceconstants
command -v ifc_pressure phonon_dispersion_relations
command -v pw.x                 # only for a Quantum ESPRESSO run
```

WARP does not check for the TDEP binaries when it starts; the stage that needs a missing one
fails instead. With no TDEP on the `PATH` at all, a fresh run stops with
`Error in WARP: generate_structure failed (no outfile.ssposcar)`, because `generate_structure` is
the first tool it calls. A missing `canonical_configuration` on its own ends the run with a
Python `FileNotFoundError`, because that tool is started without a shell. See
[Troubleshooting](troubleshooting.md#missing-external-programs).

## Physical constants

The physical constants and unit conversions of the package are defined in one module,
`warpt.constants`: the exact values of the 2019 SI redefinition and the CODATA 2018 values of the
Rydberg energy and the Bohr radius, with every conversion factor derived from them. The only values
taken from elsewhere are `ase.units.Rydberg` and `ase.units.Bohr`, with which
`warpt.MLIP_interface` converts the potential's energies, forces and stresses from eV and Å to
Rydberg atomic units; these are ASE's CODATA 2014 values, which differ from the CODATA 2018 values
of `warpt.constants` by less than one part in 10⁸. To check a conversion, or to reproduce one
outside WARP, read the value from `warpt.constants` rather than copying a number out of this
manual.
