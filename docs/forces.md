# Force engines

WARP computes no forces itself: it writes one Quantum ESPRESSO input per sampled configuration and
runs a user-supplied command on each. This page compares the three engines the package supports
(Quantum ESPRESSO, a machine-learned potential, VASP), documents the two backends and the format
they print, and explains how the configurations a run sampled can be reused as training data.

## Choosing

|  | Quantum ESPRESSO | Machine-learned potential | VASP |
| --- | --- | --- | --- |
| Requirements | `pw.x`, pseudopotentials, a converged master deck | a UPET checkpoint or a MACE model, and the matching Python stack (see [Installation](installation.md)) | a licensed VASP, an `INCAR`/`KPOINTS`/`POTCAR` template |
| Cost per configuration | an SCF | one forward pass | an SCF |
| Practical ensemble size | tens of configurations | hundreds to thousands | tens of configurations |
| Accuracy | that of the functional and settings | that of the model at that geometry | that of the functional and settings |
| Selected with | `--calculator QE -pwx "<command>"` | `--calculator MLIP -mlp <model>`, or `-pwx "python3 -m warpt.MLIP_interface ..."` | `--calculator VASP --vasp_template <dir> --vasp_cmd "<command>"` |

The ensemble size is the reason the choice matters. Quantum statistics at low temperature and a
converged phonon stress both require large ensembles, and the standard error of the stress falls
only as $1/\sqrt{n}$. A potential makes that affordable; DFT removes the model error. The usual
compromise is to fine-tune a potential on DFT configurations of the system at hand, which is the
subject of the last section of this page.

## Quantum ESPRESSO

```bash
python3 -m warpt --calculator QE -pwx "mpirun -n 8 pw.x" ...
```

WARP splits the `--pwx` string on whitespace and appends `-input <file>`, so anything that accepts
that flag at the end works: a bare `pw.x`, an `mpirun` or `srun` line, a wrapper script. The
default is `mpirun -n 6 pw.x`.

Each sample runs as its own process. `MAX_CONC` (default 4) sets how many run at once; `pw.x`
parallelises internally on top of that, so the product of the two is what has to fit the
allocation. A sample is considered finished when its output contains `JOB DONE.`, which is why
WARP can reuse the finished SCFs of an interrupted step.

## VASP

```bash
python3 -m warpt --calculator VASP --vasp_template vasp_tpl --vasp_cmd "srun vasp_std" ...
```

`warpt.VASP_interface` plays the same role for VASP that the machine-learned backend plays for a
potential: it takes the `pw.x` input WARP wrote for a sample, runs VASP on it, and prints the
result in Quantum ESPRESSO's format, so everything downstream is unchanged. `-m` still names a
`pw.x` deck, because every sample is written as one; with VASP only the cell, positions and
species of the written inputs are used, and the deck's electronic settings are ignored.

The template directory holds `INCAR`, `KPOINTS` and one `POTCAR_<element>` per element (e.g.
`POTCAR_Mg`, `POTCAR_O`, `POTCAR_H`). `KPOINTS` must be for the supercell, not the unit cell, and may be
omitted when `INCAR` sets `KSPACING`. The `INCAR` must describe a single point: before sampling
anything, WARP checks the template against the elements of the `-ucp` unit cell, refuses a missing
file, `NSW` > 0 or `ISIF` < 2 (no stress), and warns when `LWAVE` or `LCHARG` is not `.FALSE.`.

Per sample `<input>` the backend uses the directory `<input>.vasp/`:

1. `POSCAR` from the input, atoms in the input's order; `INCAR` and `KPOINTS` copied from the
   template; `POTCAR` concatenated from the `POTCAR_<element>` files in the order of the
   `POSCAR` species line.
2. The `--vasp_cmd` command runs there; its standard output and error go to `vasp.stdout`.
3. The run counts only when VASP exits with status 0, the `OUTCAR` is complete (its timing
   summary is present), the electronic loop converged ("EDIFF is reached", and no "EDIFF was not
   reached"), its atoms match the input in species, order and position (to 10⁻³ Å), and it
   contains a stress. Anything else prints `JOB FAILED. <reason>` and the run stops with that
   reason.
4. The free energy (TOTEN, the energy the forces are the gradient of), forces and stress are
   printed in QE units, converted with WARP's own constants, followed by `JOB DONE.`.
5. `WAVECAR`, `CHG`, `CHGCAR`, `vasprun.xml` and the other large files are deleted, whether the
   run succeeded or not (`--keep_all` on the backend keeps them); `OUTCAR`, `OSZICAR`, `CONTCAR`
   and `vasp.stdout` stay.

WARP splits the calculator command on whitespace, so the VASP command, which has spaces of its
own, reaches the samples through the `WARPT_VASP_CMD` environment variable that the orchestrator
sets. Each VASP run is already parallel, so `MAX_CONC` defaults to 1 for this engine.

## The machine-learned backend

`warpt.MLIP_interface` is a small program that reads a `pw.x` input, evaluates it with an ASE
calculator, and prints the result in Quantum ESPRESSO's own format. Everything downstream — the
completion check, the collector, the stress parser — therefore treats it exactly like `pw.x`.

### Two ways to reach it

```bash
# 1. through the orchestrator's own options
python3 -m warpt --calculator MLIP -mle upet -mld cpu -mlp /path/to/model.ckpt -mlv <head> -mlc ...

# 2. through --pwx, for backend options the first route does not expose
python3 -m warpt --calculator QE \
    -pwx "python3 -m warpt.MLIP_interface --engine upet --device cpu \
          --ml_params /path/to/model.ckpt --variant <head> --conservative" ...
```

The first route builds the command
`python3 -m warpt.MLIP_interface --engine <-mle> --device <-mld> --ml_params <-mlp>`, plus
`--variant <-mlv>` and `--conservative` when `-mlv` and `-mlc` are given, with `-mlp` made
absolute. The second is required for `--rot-avg`, a named UPET model without a checkpoint, and
the MACE options. Because the samples run in per-step subdirectories, give `--ml_params` in this
route as an absolute path. The sampling stage recognises the backend by the substring `MLIP_interface` in the
command, so batching and the memory cap still apply.

!!! warning "A fine-tuned head must be named"
    `--variant` selects which energy head of a UPET checkpoint is evaluated. A fine-tuned
    checkpoint evaluated **without** its head silently returns the base model's energies. Pass
    `-mlv <head>` (route 1) or `--variant <head>` (route 2).

### Backend options

```bash
python3 -m warpt.MLIP_interface -input scf.in --engine upet --ml_params model.ckpt
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-input` | — | The `pw.x`-style input to evaluate. Required unless `--batch` is given; the job runner appends it automatically. |
| `--engine` | `upet` | `upet` or `mace`. |
| `--device` | `cpu` | `cpu` or `cuda`, passed to the calculator. |
| `--ml_params` | none | The model file: a UPET `.ckpt`, which overrides `--model`/`--version`, or a MACE `.model`, which is required for that engine. |
| `--model` | `pet-mad-s` | UPET model name when no checkpoint is given: `pet-mad-xs`, `pet-mad-s`, `pet-omat-s`, `pet-omat-m`, `pet-omat-l` or `pet-omatpes-l`. UPET fetches it when it is not cached locally. |
| `--version` | per model | UPET model version; defaults to the version recorded for `--model`, and to `latest` for a name with no recorded version. |
| `--conservative` | off | Compute UPET forces and stresses as energy gradients instead of using the direct force and stress heads. Uses more memory. |
| `--variant` | none | The UPET energy head to evaluate, for example the head of a fine-tuned model. |
| `--batch` | none | A file of `<input> <output>` pairs, one per line, with blank and `#` lines ignored. All of them are evaluated in this one process with a single calculator. |
| `--rot-avg` | `0` | Lebedev–Laikov order for UPET's rotational averaging; 0 disables the averaging wrapper entirely. |
| `--rot-avg-batch` | `8` | Rotated copies evaluated per batch when `--rot-avg` is greater than zero. |
| `--dispersion` | off | Enable MACE's dispersion correction. |
| `--dtype` | `float64` | MACE floating-point precision: `float32` or `float64`. |
| `--allow_no_stress` | off | Print a zero stress tensor when the calculator provides none, instead of failing. Not recommended for a variable-cell relaxation, which is driven by that stress. |

!!! note "Why rotational averaging is off by default"
    The averaging wrapper aliases its results dictionary to the inner calculator's, so a
    calculator reused across structures can return a previous structure's forces. Leaving
    `--rot-avg` at 0 keeps the wrapper out, which is what makes batch mode safe.

### What it prints

Both backends emit the lines the collector parses, so that one set of parsers serves every engine:

```text
Fake QE output generated with <backend>
number of atoms/cell = <N>

! total energy = <E> Ry

 Forces acting on atoms (cartesian axes, Ry/au):

 atom    1 type 1 force = <fx> <fy> <fz>
 ...

 Computing stress (cart. axes) and pressure

 total stress (Ry/bohr**3) (kbar) P= <p>
 <three rows in Ry/bohr³, each followed by its kbar values>
JOB DONE.
```

Units and signs follow Quantum ESPRESSO exactly: the energy in Ry, forces in Ry/bohr with the
force-on-atom sign preserved, and the stress in Ry/bohr³ with the sign flipped to QE's
compression-positive convention, since ASE reports it tension-positive.

Completion is signalled the same way too. `JOB DONE.` is printed only after a successful
evaluation; a failure prints `JOB FAILED. <error>` instead, and a run that sees that marker aborts
at once with the backend's own message rather than waiting for a timeout. In batch mode each
structure writes its own output with its own marker, so the outputs that finished before a failure
stay valid and are reused on the next attempt.

### CPU and GPU

- `--device cpu` is the default, on both the orchestrator and the backend.
- When no CUDA device is visible, the backend wraps `torch.load` and `torch.jit.load` at import
  time so that they deserialise on the CPU. A CUDA-trained checkpoint therefore loads on a
  CPU-only machine without any extra flag.
- Process start-up — interpreter, imports, checkpoint load — dominates a single evaluation, so
  configurations are grouped into **chunked batches** by default: the queued jobs are split
  round-robin into at most `MAX_CONC_MLIP` chunks and each chunk is walked by one process that
  loads the checkpoint once. Set `WARPT_MLIP_BATCH=0` to return to one process per configuration.
- Each chunk is a separate interpreter holding the whole model, so node memory scales with the
  number of chunks, not with the core count, and the peaks coincide because the chunks start
  together. When a SLURM memory budget is visible and the checkpoint size can be read, WARP caps
  the chunk count accordingly and reports it in one line (wrapped here):

```text
[warpt] MLIP batching: memory budget ... GB, ~... GB per process (... GB checkpoint)
        -> at most N concurrent; REQUESTED M, CAPPED TO N
```

  Without such a budget nothing caps it, and the line reads
  `no Slurm memory limit visible -- concurrency left at N`; when the checkpoint size cannot be
  read it begins `checkpoint size unknown (--ml_params unreadable)` and ends
  `memory NOT verified`. On a workstation, set `MAX_CONC_MLIP` explicitly. `WARPT_MLIP_PROC_GB`
  overrides the fixed per-process part of the estimate for large supercells. See [Environment variables](environment.md).

## Starting from the potential's own minimum

A WARP run driven by a potential should start where that potential itself has vanishing forces and
the target pressure, so that the residuals WARP measures are thermal rather than a mismatch between
the starting geometry and the model. For a UPET checkpoint, `warpt.upet_relax` produces exactly
that geometry:

```bash
python3 -m warpt.upet_relax \
    --structure start_guess.vasp --press 100 \
    --ml_params /path/to/model.ckpt --variant <head> \
    --out start.vasp --report start.json
```

It relaxes the cell and the positions with ASE's `FrechetCellFilter` under a `FixSymmetry`
constraint, then applies an acceptance gate:

| Check | Option | Default |
| --- | --- | --- |
| the optimizer converged | `--fmax`, `--steps` | 0.001 eV/Å, 500 steps |
| the pressure hit the target | `--gate-press` | 0.05 GPa |
| the deviatoric stress is small | `--gate-aniso` | 0.10 GPa |
| the symmetrised residual force is small | `--gate-fmax` | 2.0 meV/Å |
| the space group is unchanged | — | always |

`--structure`, `--press`, `--ml_params` and `--variant` are required; pass `--variant none` to use
the checkpoint's base head. The exit code is 0 when the gate passed, 3 when it failed — the
structure is still written, so it can be inspected — and 1 on an error. The *symmetrised* force
is gated rather than the raw one, because on a special position the symmetry-projected force is
exactly zero while the raw force is the model's equivariance error, which no symmetry-preserving
move can reduce; `--warn-fmax-raw` (default 10 meV/Å) reports the raw value above that level
without gating it. `--eval-only` prints the starting state and stops without writing a file,
`--no-fixsym` relaxes without the constraint, and `--optimizer` chooses between `lbfgs` and `fire`.

The tool evaluates forces and stress as energy gradients by default, whereas the sampling backend
uses the direct heads unless `-mlc` is given. For the start geometry to be the minimum of the
model WARP actually samples with, either pass `-mlc` to WARP or `--non-conservative` to
`warpt.upet_relax`.

## Fine-tuning a potential from a run

A relaxation is also a generator of training data: every configuration it sampled was evaluated
with the chosen force engine, and every one of those evaluations is kept on disk. WARP has no
converter to a training format — that belongs to the potential's own toolchain — but it does leave
the data in one place per step, in a form that is easy to read.

### The relaxation path

```bash
python3 -m warpt.get_frames -r <run root>
```

writes `trajectory.xyz` in the run root with one frame per step directory that has produced an
updated cell, in execution order, and, unless `--no-copy` is given, a `frame_NNNN_<step>.vasp` for each, named after its step (for
example `frame_0007_iter_2_position_1.vasp`), with the step as its title line; in `trajectory.xyz`
the step is the `step=` key of each frame's comment line. Each frame is the **updated unit
cell** that step produced — the first `outfile.new*ucposcar` found below it — so this is the path
the structure took, not the ensemble around it. It shows where the relaxation went and provides a
set of distinct reference structures. Every frame is also checked for symmetry: a table gives
the space group of each frame at the tolerances listed by `--symprec` (default 1e-5, 1e-4, 1e-3
and 1e-2 Å), and `trajectory.xyz` stores them as `spg_<symprec>`. A group that appears only at the
looser tolerances is one the structure approaches but does not have. `-o` names another output
file and `-q` silences the table and the summary; the exit status is 1 when no frame was found. `-r` (default: the
current directory) must be the **run root**, the directory that holds `iter_*`, because the step
directories are looked up inside those `iter_*` directories.

### The sampled supercells

The ensemble lives in the step directories, in the files the fit was performed on:

| File | Content |
| --- | --- |
| `infile.ssposcar` | the supercell every configuration of that step is a displacement of |
| `infile.positions` | the fractional positions of every configuration, concatenated, `nat` lines per configuration |
| `infile.forces` | the matching forces in eV/Å, in the same order |
| `infile.stat` | one row of 13 columns per configuration: the index, the total energy in eV (column 3), the pressure in GPa (column 7, minus the mean of the three diagonal stress components, so negative under compression and opposite in sign to the calculator's `P=`) and the six stress components in GPa (columns 8–13, compression-positive, ordered `xx yy zz xy yz xz`); the remaining columns are zero |
| `infile.meta` | the atom count and the number of configurations, which determine how to split the two long files |
| `calc_samples/` | a copy of those four files, refreshed whenever they change |
| `<master>.samples<n>` and `<master>.samples<n>.out` | the untouched calculator input and output of configuration `n` |

Three details matter when assembling a training set from them:

- **A step's files may hold more than that step sampled.** When a structure update was small, the
  previous step's fitted sample set (its `infile.positions`, `infile.forces` and `infile.stat`,
  which already include any samples that step had itself merged) is appended to the current one
  before the fit. Over several consecutive small updates the set therefore keeps growing, and
  `infile.stat` can be several times longer than `-n`. Read the count from `infile.meta` rather
  than assuming it.
- **A preconditioning geometry keeps its own set separately** in `local_samples/`, snapshotted
  before any merging. That is the unmerged ensemble of that geometry alone.
- **A lattice step resampled by `--fc3_gate_thr`** keeps the rejected attempts in
  `fc3_gate_attempt_<k>/`; they are evaluated configurations as well. An attempt restored as the
  least-bad one is moved back into the step directory; the attempt it replaces is first archived
  in its own `fc3_gate_attempt_<k>/` and stays there.

The raw `.out` files are the most faithful source: they are the calculator's own output, so any
training pipeline that already reads a `pw.x` run works on them unchanged. With VASP the
calculator's own output is the `OUTCAR` kept in each `<master>.samples<n>.vasp/`.
