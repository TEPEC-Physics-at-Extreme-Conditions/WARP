# Input files

A WARP run needs exactly two prepared files: a Quantum ESPRESSO input deck that serves as the
template for every sampled configuration, and the unit cell to start from. This page gives the
rules both must satisfy, lists the optional files the analysis tools look for, and then names
every file WARP itself writes and reads back, so that prepared inputs can be told apart from
generated ones.

## The two required files

### 1. The master deck — `-m, --masterscf`

A complete `pw.x` SCF input for the **unit cell**: pseudopotentials, cutoffs, functional,
smearing and k-point mesh. WARP resolves the path against the launch directory and stops at once
if the file is not there:

```text
Error in WARP: Master SCF file not found: ...
```

The deck is used in two ways. Its basename is copied into every step directory, and it is the
*seed* from which the structure converter builds one input per sampled configuration: `nat`, the
cell and the atomic positions are substituted from the sampled supercell, and every other line is
streamed through with its content unchanged (each kept line is re-indented with one leading space,
and a first line that is only a title is replaced by a timestamp banner; see
[Seed streaming](structconv.md#seed-streaming)). The per-sample inputs are named `<master>.samples1`,
`<master>.samples2`, … and their outputs add `.out`.

**Required content**

| The deck must contain | Because |
| --- | --- |
| `nat = ...` at the start of a line of its own | the whole line is replaced by `nat= <atom count of the supercell>`, so any other keyword on it is lost; a `nat` that does not start its line is not rewritten at all |
| `ibrav = 0` with a `CELL_PARAMETERS` block, or `ibrav = 1` with `celldm(1)` | the converter supports these two settings only; `ibrav = 1` is simple cubic and suits only a cubic supercell, so `ibrav = 0` is the general choice |
| `ATOMIC_POSITIONS` with one line per atom | replaced by the sampled positions, written as `(crystal)` |
| a `K_POINTS` line **after** `ATOMIC_POSITIONS`, with no other card between them except `CELL_PARAMETERS` | it ends the block of old coordinate lines that the converter drops; any other line in that block is discarded |
| a `&CONTROL` namelist closed by `/` on a line of its own | the per-sample `outdir` and `disk_io` lines are inserted before that `/` |
| settings that make `pw.x` print forces and the stress tensor (`tprnfor = .true.`, `tstress = .true.`) | the collector parses a `!` total-energy line, one `force =` line per atom and a `total stress` block out of every output |

The collector counts every line of the form `atom ... force =` as one atom. A `verbosity = 'high'`
setting makes `pw.x` also print the individual force contributions in that form, which breaks the
count; leave `verbosity` at its default.

Two lines are rewritten per sample whether or not the deck supplies them: `outdir` is pointed at
that sample's own `outdir_sample_<n>/`, and `disk_io` is set to `'none'`. They may be present in
`&CONTROL` or absent; either way each sample gets its own scratch directory and writes no
wavefunctions.

!!! warning "How `ibrav` is detected on write"
    The converter decides between its two branches by looking for the character `1` anywhere on
    the `ibrav` line. Write the line plainly — `ibrav= 0` or `ibrav= 1` — and keep comments and
    other numbers off it.

!!! note "The cell in the deck is only a starting point"
    The cell and the positions in the deck are overwritten for every sample, so they need not
    match `-ucp`. Everything else — species, masses, pseudopotential filenames, `pseudo_dir`,
    cutoffs, k-points — is taken from the deck as written and never touched.

!!! note "The deck with a machine-learned potential or VASP"
    The deck is required with every force engine. With `--calculator MLIP` or
    `--calculator VASP`, every sample is still written as a `pw.x` input built from it, and the
    backend reads the cell, the positions and the species from that file. The structural
    requirements in the table above still apply, and `ATOMIC_SPECIES` must list every element of
    the structure; the pseudopotentials, cutoffs, smearing, k-points and the force and stress
    flags are not used.

An example skeleton:

```text
 &CONTROL
 calculation= 'scf'
 pseudo_dir= '/path/to/pseudopotentials'
 tstress= .true.
 tprnfor= .true.
 /
 &SYSTEM
 ecutwfc= 100
 ecutrho= 800
 ibrav= 0
 nat= 5
 ntyp= 3
 occupations= 'smearing'
 degauss= 0.01
 smearing= 'gaussian'
 /
 &ELECTRONS
 conv_thr= 1E-6
 /
 ATOMIC_SPECIES
 ...
 ATOMIC_POSITIONS crystal
 ...
 CELL_PARAMETERS angstrom
 ...
 K_POINTS automatic
 2 2 2 0 0 0
```

!!! tip "The deck describes the unit cell, the samples are supercells"
    The k-point mesh in the deck is applied unchanged to every sampled supercell. Choose it for
    the supercell requested with `-sc`, not for the unit cell in the file.

### 2. The starting unit cell — `-ucp, --ucp, --unit-cell-poscar`

A VASP5 POSCAR: title, scale, three lattice vectors, a line of **species symbols**, a line of
counts, `Direct` or `Cartesian`, then the coordinates. An optional `Selective dynamics` line is
tolerated by the readers. A VASP4 file, with counts where the symbols should be, is not
supported: the file is handed unchanged to TDEP's `generate_structure` and then to WARP's POSCAR
normaliser, which reads the atom counts from line 7 and so expects the VASP5 layout. The run fails
at the start of preconditioning.

This file does two jobs:

- it is copied to `infile.ucposcar` in the preconditioning directory and starts the relaxation;
- its space group is the one the run starts in: every position and strain update is projected
  onto the space group of the current structure, which begins as this one.

The starting structure should therefore have exactly the intended symmetry, and should already be
the static minimum of the chosen force engine at the target pressure.

!!! warning "A missing `-ucp` file is not always reported early"
    With Quantum ESPRESSO or a machine-learned potential, only the master deck is checked when
    the run starts; a missing or unreadable `-ucp` file surfaces later, when the preconditioning
    stage tries to copy it. With `--calculator VASP` the file is read at start-up, to check the
    template's `POTCAR_<element>` files against its elements.

## Optional files the tools look for

These are not needed for a relaxation. `python3 -m warpt.trajectory` reads them, in the run root,
to label its plots:

| File | Read for |
| --- | --- |
| `warpt_run.log` | the target pressure, from a `press_ext=<value>` or `press_ext <value>` string; failing every source in this table, from the last `σ_ext: [...]` line of the stress diagnostic |
| `run_warpt.sh`, `run.sh`, `job*` | the target pressure, from `--press_ext <value>` (this spelling only, not `-pe`) |
| `start.vasp` | the fixed symmetry anchor of the forbidden-stress check; `infile.ucposcar` in the run root is used if it is absent, and otherwise the first structure of the run |

Only the first number is read, which for a three- or six-component target is its `xx`
component. The file patterns match unsigned numbers only, so a negative value is skipped, and the
`σ_ext` fallback takes the absolute value; the residual panel therefore assumes a hydrostatic,
compressive target. Redirecting the run's output to `warpt_run.log`, as
the [quick start](quickstart.md) does, is enough once the first lattice step has printed its
stress diagnostic.

## Files WARP creates and reads back

None of these is written by hand; they are listed so that they can be recognised. All of them
live inside a step directory. The full layout is in
[Run directory and resume](run-directory.md).

### Structure and force-constant inputs

| File | Written by | Read by |
| --- | --- | --- |
| `infile.ucposcar` | the geometry handover, from the best unit cell of the previous step | `canonical_configuration`, the lattice update, the Wyckoff projector |
| `infile.ssposcar` | `generate_structure`, from `infile.ucposcar` and the `-sc` flags | `canonical_configuration` |
| `infile.forceconstant` | copied forward from the previous step, then replaced by this step's own fit | `canonical_configuration`, `phonon_dispersion_relations`, `ifc_pressure` |
| `infile.forceconstant_thirdorder` | the same, when a third-order fit exists | the Grüneisen grid and `ifc_pressure` |

Both POSCARs are rewritten in place as scale-1, `Direct` files with coordinates wrapped into
$[0,1)$ before sampling.

### Sampling and calculator files

| File | Written by | Read by |
| --- | --- | --- |
| `infile.isotopes` | the sampling stage, only with `--masses`: one block per atom of `infile.ucposcar` (the number of isotopes, 1, then `concentration mass`) | `canonical_configuration` and `phonon_dispersion_relations`, through `--readiso` |
| `contcar_conf*` | `canonical_configuration`, one per sampled configuration | the sampling stage, which reuses a complete set |
| `Clean_<n>.pos` | the sampling stage: the header and exactly `nat` coordinate lines of one configuration | the structure converter |
| `<master>` | copied into the step directory | the structure converter, as the seed |
| `<master>.samples<n>` | the structure converter, then patched in `&CONTROL` | the calculator, and the collector for the positions |
| `<master>.samples<n>.out` | the calculator (stdout and stderr together) | the completion check and the collector |
| `mlip_batch.<k>.list`, `mlip_batch.<k>.list.log` | the sampling stage, when a potential is evaluated in chunked batches | `warpt.MLIP_interface` |
| `outdir_sample_<n>/` | the sampling stage | `pw.x` |

### Collected TDEP inputs

`infile.positions` (fractional), `infile.forces` (eV/Å), `infile.stat` (one row per configuration)
and `infile.meta` are produced by the collector from the calculator outputs and are what
`extract_forceconstants` is actually fitted to. They are copied to `calc_samples/` as they are
written, and a preconditioning geometry additionally keeps its own, unmerged copy in
`local_samples/`.

`infile.stat` carries, per configuration and in this order: the index, a zero, the total energy in
eV, three more zeros, the pressure in GPa and the six stress components in GPa. Its stress columns
are compression-positive and ordered `xx, yy, zz, xy, yz, xz`, which is the order TDEP's reader
expects and **not** WARP's own Voigt order; the pressure column is $-\mathrm{tr}(S)/3$ of that
tensor, and therefore negative under compression, opposite in sign to the `P=` the calculator
prints. The standard errors used by the noise-referenced damping and the noise floor are computed
from these columns.

### Outputs that are read again

`outfile.forceconstant*`, `outfile.uc_displacements`, `outfile.new_ucposcar`,
`outfile.dispersion_relations.hdf5`, `outfile.grid_dispersions.hdf5`, `outfile.stress_{dft,fc2,fc3,res}`,
`outfile.compliance_tensor`, `outfile.new_vc_ucposcar` and `outfile.sigma_ph` are written by TDEP
or by WARP and consumed by the next stage. They are described where they are used, in
[Method](method.md) and [Run directory and resume](run-directory.md).
