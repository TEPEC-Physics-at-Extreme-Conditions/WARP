# Run directory and resume

WARP takes the directory it is launched in as the run root and builds its whole tree below it.
This page maps that tree, lists which stage writes what, describes the CSV the trajectory tool
produces, and explains exactly how an interrupted run continues.

## The tree

```text
<run root>/
├── <master deck>                  input deck; copied into every step directory
├── <starting POSCAR>              input structure
├── vc_restart.ucposcar            copy of the newest updated cell, rewritten at every lattice step
├── iter_0/
│   ├── preconditioning/
│   │   ├── infile.ucposcar        copy of the starting POSCAR
│   │   ├── infile.ssposcar
│   │   ├── outfile.ssposcar
│   │   ├── geom_0/                bootstrap force constants
│   │   ├── geom_1/
│   │   └── geom_.../              one per entry of the sample schedule, until the gate passes
│   ├── position_0/
│   ├── position_1/                one per position step, until the residual force is in tolerance
│   └── lattice_/                  exactly one per iteration; note the trailing underscore
├── iter_1/
│   ├── position_0/ ...
│   └── lattice_/
└── iter_N/
    └── lattice_/
        └── final_structure.vasp   the converged structure; only the last lattice step has it
```

Only `iter_0` has a `preconditioning/` directory: later iterations start from the previous
iteration's `lattice_`. Each iteration has exactly one `lattice_` directory, and the run ends
inside the last one.

!!! note "`--mode preconditioning` builds a different tree"
    Run on its own, the preconditioning stage is not wrapped in an iteration: it creates
    `preconditioning/geom_<k>` **directly** under the launch directory and stops when the gate
    passes.

## What each stage writes

Every step directory — a preconditioning geometry, a position step or the lattice step — passes
through the same pipeline, so they share most of their contents.

### In every step directory

| Files | Stage |
| --- | --- |
| `infile.ucposcar`, `infile.ssposcar`, `outfile.ssposcar` | setup: the unit cell is taken from the previous step, preferring `outfile.new_vc_ucposcar`, then `outfile.new_ucposcar`, then `infile.ucposcar`, and `generate_structure` builds the supercell from it |
| `infile.forceconstant`, `infile.forceconstant_thirdorder` | setup: copied forward from the previous step, then replaced by this step's own fit |
| `infile.isotopes` | sampling, only with `--masses`: the per-atom masses that `canonical_configuration` and `phonon_dispersion_relations` read |
| `contcar_conf*` | sampling: one configuration per file, from `canonical_configuration` |
| `Clean_<n>.pos`, `<master>`, `<master>.samples<n>`, `outdir_sample_<n>/` | sampling: the per-configuration calculator input and its scratch directory |
| `<master>.samples<n>.out` | forces: the calculator's stdout and stderr, which must end in `JOB DONE.`; with the VASP backend each sample also has its own `<master>.samples<n>.vasp/` run directory |
| `mlip_batch.<k>.list`, `mlip_batch.<k>.list.log` | forces, potential backend only: the chunk list and that chunk process's own log |
| `infile.positions`, `infile.forces`, `infile.stat`, `infile.meta`, `calc_samples/` | collect: the parsed ensemble, and a copy of it |
| `extract_forceconstants.log`, `outfile.forceconstant`, `outfile.forceconstant_thirdorder`, `outfile.forceconstant_firstorder`, `outfile.uc_displacements`, `outfile.new_ucposcar` | fit |
| `outfile.uc_projected_structure` | fit: the Wyckoff-projected structure, which is then copied over `outfile.new_ucposcar` |
| `outfile.dispersion_relations`, `.hdf5`, `.gnuplot` | phonons |

### Only in a preconditioning geometry

| File | Why |
| --- | --- |
| `local_samples/infile.{positions,forces,stat,meta}` | this geometry's own, unmerged samples, snapshotted before the previous geometry's are appended, so the next geometry can use the sliding window |
| `outfile.stress_*`, `outfile.compliance_tensor` | preconditioning geometries run `ifc_pressure`; position steps do not |

### Only in a lattice step

| File | Written by |
| --- | --- |
| `outfile.stress_dft`, `outfile.stress_fc2`, `outfile.stress_fc3`, `outfile.stress_res`, `outfile.compliance_tensor` | `ifc_pressure` |
| `outfile.grid_dispersions.hdf5` | `phonon_dispersion_relations --dumpgrid --gruneisen`, and **deleted again** as soon as the phonon stress has been extracted from it |
| `outfile.sigma_ph` | WARP: the six components of the phonon stress, tension-positive, in GPa |
| `outfile.new_vc_ucposcar` | the lattice update |
| `outfile.fc3_gate` | WARP, only with `--fc3_gate_thr`: one row per attempt with its sample count and mismatch |
| `fc3_gate_attempt_<k>/` | WARP, only when that gate trips: the whole failed attempt, moved aside |
| `final_structure.vasp` | WARP, only on convergence: a copy of the step's `infile.ucposcar`, the structure on which the converged stress was evaluated; no `outfile.new_vc_ucposcar` is written in that step |

A position step writes neither `outfile.stress_*` nor a compliance tensor: it never runs
`ifc_pressure`.

## Logs

WARP writes no run log of its own. Everything goes to standard output, tagged by stage — for
example `[restart]`, `[preconditioning]`, `[position]`, `[lattice]`, `[samples]`, `[convergence]`,
`[volume-guard]`, `[wyckoff]`, `[sanity]`, `[fc3-gate]`, `[canonical]`, `[warpt]`,
`[time]` — so redirect it to a file. `warpt_run.log` in the run root is the name the trajectory tool
looks for when it needs the target pressure.

## The trajectory CSV

```bash
python3 -m warpt.trajectory <run root>
```

writes one row per step directory that has a readable `infile.ucposcar`, in execution order, to
`warpt_trajectory.csv` in the current directory (`--csv` changes the name):

| Column | What it holds |
| --- | --- |
| `iter`, `kind`, `index`, `path` | where the step sits: `kind` is `precond`, `position` or `lattice` |
| `natoms`, `a`, `b`, `c`, `alpha`, `beta`, `gamma`, `vol_per_atom`, `c_over_a` | the cell of that step |
| `spacegroup`, `tightest_symprec` | the highest space group found over a ladder of tolerances, and the tightest tolerance at which it still holds |
| `fc1_mean_meV_A`, `fc1_max_meV_A` | the residual forces the position loop is judged on |
| `n_configs` | rows in that step's `infile.stat`, i.e. the size of the fitted ensemble |
| `min_freq_THz` | the minimum phonon frequency of that step's dispersion |
| `step_rmsd_A` | Cartesian RMSD against the previous step |
| `complete` | whether the step has a full internal stress; incomplete steps are left out of the stress panels |
| `P_int_GPa`, `dev_int_GPa` | the internal pressure and the $\sigma_{xx}-\sigma_{zz}$ deviatoric component |
| `forbidden_GPa` | the largest symmetry-forbidden component of the internal stress, measured against a **fixed** anchor structure so that a drifting cell shows up as a growing value |
| `site_spg`, `site_group` | the space group of the step at symprec 0.01 Å, number and symbol; these set the tinted bands of the figure |
| `w_<atom>_x/y/z`, `site_<atom>` | the fractional coordinates, in the cell of the run, of one atom per symmetry orbit of the first and of the last step, and the Wyckoff site it occupies at that step; `<atom>` is the element and the atom's number in the file (`H9`), the site is element, multiplicity and letter (`H32f`), with `-2`, `-3`, ... when several orbits share a site |
| `press_resid_GPa` | the largest stress residual against the target pressure; needs that pressure to have been found, and reads it as an unsigned number (the first, i.e. xx, component of `--press_ext`), so a tensile target is not recovered with its sign |
| `dsigma_ph_GPa` | the change in the phonon stress since the previous step that had one |

The figure that accompanies it, `warpt_trajectory.png` (`--plot` changes the name, `--no-plot`
skips it), shows how the structure evolves, step by step: the lattice parameters $a$, $b$ and $c$,
each on its own axis; the cell angles $\alpha$, $\beta$ and $\gamma$; the volume per atom; and the
atomic coordinates that move, relative to the first step. Vertical lines mark the lattice steps,
and the title gives the space group at the first and the last step. The background is tinted by
the space group of each step at symprec 0.01 Å, one tint per group, and the line under the title
lists which steps are in which group. The coordinates are those of the same atom in the cell of
the run at every step: WARP never reorders atoms, so the curves stay continuous through a change
of space group, and a coordinate pinned by site symmetry stays flat. The legend names each atom by
its element and number in the file and lists every Wyckoff site it occupied, for example
`H#1 6c → 8c z`. The convergence quantities stay in the CSV.

To collect the structures themselves rather than the numbers:

```bash
python3 -m warpt.get_frames -r <run root>
```

writes `trajectory.xyz` and, unless `--no-copy` is passed, one `frame_NNNN_<step>.vasp` per frame,
both in the run root; every frame carries its step (`iter_2/position_1`, `iter_2/lattice`, ...) in
the file name, in the POSCAR title line and as `step=` in the `trajectory.xyz` comment line. The two tools select steps differently, so the frame count and the step count can
differ: `get_frames` includes every directory matching `iter_*`, so a sidelined copy such as
`iter_6.stale` contributes frames, but takes each frame from a step's `outfile.new*ucposcar` and
skips steps that have not yet written one (a lattice step contributes its `outfile.new_ucposcar`,
which is found before `outfile.new_vc_ucposcar`, so its frame is in the cell before the lattice
update); `trajectory` reads only directories named exactly
`iter_<N>` and lists a step as soon as its `infile.ucposcar` exists.

To see the phonons of every step:

```bash
python3 -m warpt.plot_phonons_iter <run root>
```

reads the `outfile.dispersion_relations.hdf5` (and, when present, `outfile.phonon_dos.hdf5`) of
every step in execution order and writes, into `phonon_plots/`, one figure per step named after it
(`000_iter0_geom0.png`, `007_iter2_position1.png`, `012_iter2_lattice.png`, ...) and
`overview.png`, a grid of all dispersions on one frequency axis. Steps with imaginary modes have a
red title and are flagged in the printed summary. `--band-yaml` also writes a phonopy `band.yaml` per
step; `python3 -m warpt.band_yaml <step dir>` does the same for a single step.

## Resume

Restart a run by issuing **the same command in the same directory**. There is no separate restart
flag and no checkpoint file.

### How completion is decided

WARP keeps no record of its progress; it re-derives completion from the output files themselves.
A step directory passes through eight stages in order —

`setup` → `sampling` → `scf` → `collect` → `extract` → `ifc_pressure` → `phonons` → `lattice`

— and each has a predicate:

| Stage | Complete when |
| --- | --- |
| `setup` | `infile.ucposcar` and `infile.ssposcar` exist and are non-empty |
| `sampling` | `contcar_conf*` files exist, and reach the expected count when one is known |
| `scf` | at least one `<master>.samples<n>` exists and **every** one has a `.out` containing `JOB DONE.` |
| `collect` | the four collected files exist and their line counts agree with `infile.meta` |
| `extract` | `outfile.forceconstant` is non-empty |
| `ifc_pressure` | `outfile.stress_res` parses as six numbers — required for preconditioning geometries and lattice steps, not for position steps |
| `phonons` | where a DOS dispersion is required, `outfile.dispersion_relations.hdf5` has a `frequencies` dataset; where the Grüneisen grid is required, `outfile.grid_dispersions.hdf5` has frequencies and Grüneisen data |
| `lattice` | `outfile.new_vc_ucposcar` exists and is non-empty |

The two HDF5 checks need `h5py`; without it the `phonons` stage always counts as missing.

### What is redone after a kill

The continuation is a forward walk from iteration 0 over the existing tree. Each step directory
either recognises itself as finished and is skipped, or is rebuilt from its setup onwards while
keeping every force evaluation that already finished:

- a preconditioning geometry is finished when every stage through `phonons` (DOS dispersion) is
  complete; from `geom_1` on, its gate is then re-checked from the files;
- a position step is finished when every stage through `extract` is complete; its phonon check
  is informational and is not repeated;
- a lattice step is finished when `outfile.new_vc_ucposcar` or `final_structure.vasp` exists.

Within a step that is rebuilt:

| Interrupted during | Redone on the next run |
| --- | --- |
| sampling | a **complete** set of `contcar_conf*` is reused as it is; a partial set is deleted and the whole set drawn again, because the sampler writes it in one shot |
| a force evaluation | only the samples whose output lacks `JOB DONE.`; an existing output without the marker is deleted first |
| collecting, the fit, `ifc_pressure`, the phonons or the lattice update | every stage from collection onwards, on the force outputs already present; no force evaluation is repeated |

With `--fc3_gate_thr`, a rebuilt lattice step starts again at the first gate attempt; existing
`fc3_gate_attempt_<k>/` directories are not taken into account.

A finished lattice step is recognised by `outfile.new_vc_ucposcar`: WARP re-reads its
`outfile.sigma_ph`, recomputes the relative volume change, refreshes `vc_restart.ucposcar` and
advances to the next iteration without redoing any force evaluation. A converged one is recognised
by `final_structure.vasp`, and the run returns immediately.

### The restart report

In `--mode vc-relax`, before the walk starts, the orchestrator prints one plan:

```text
[restart] Detected previous incomplete run
[restart] iteration = 1
[restart] kind      = lattice
[restart] dir       = .../iter_1/lattice_
[restart] stage     = phonons
```

This is a report only; the continuation is the forward walk, not the plan. Two entries are
routinely misleading and both are harmless:

- a **finished** `lattice_` directory is reported at stage `phonons`, because the Grüneisen grid
  file it needed was deleted after use;
- a **finished** `position_*` directory is reported at stage `ifc_pressure`, because the scan asks
  for that stage by default and position steps never run it.

### Two caveats

!!! warning "The tree, not the command line, defines the run"
    Resume reads files, not arguments. Restarting a directory with a different temperature,
    cutoff, supercell or target pressure reuses the completed stages exactly as they are and
    silently mixes the two settings. Start a fresh directory whenever anything physical changes.

!!! note "A stage-granular helper that the command line does not use"
    `warpt.resume.complete_dir` runs only the stages a directory is missing, starting at the
    first incomplete one, and can be called on a fresh, partial or complete directory. It is a
    library entry point for callers who drive a single directory themselves; the command-line
    workflow resumes through the forward walk described above and never calls it.
