# Command line

WARP has two main command-line tools: the orchestrator, which runs a relaxation, and the
trajectory tool, which reports on one afterwards. This page lists every option of both, grouped by
what it controls, with its default. The force-engine backends have their own options; those are on
[Force engines](forces.md).

```bash
python3 -m warpt      [options]        # or the equivalent console script: warpt [options]
python3 -m warpt.trajectory [run_dir] [options]
```

Most orchestrator options have several spellings — a short form, an underscore form and usually a
hyphen form. They are listed together; any one of them works. `-h`, `--help` prints the option
list and exits.

## `python3 -m warpt` — the orchestrator

### Required

Five options have no default and the run refuses to start without them.

| Option | Meaning |
| --- | --- |
| `-m`, `--masterscf` | Quantum ESPRESSO input deck, relative to the working directory, used as the template for every sampled configuration. Its basename is copied into each step directory and prefixes the `<master>.samplesN` inputs and their `.out` files. |
| `-ucp`, `--ucp`, `--unit-cell-poscar` | Starting unit cell in POSCAR format, relative to the working directory. Copied to `infile.ucposcar` in the preconditioning directory (`iter_0/preconditioning/` in a `vc-relax` run). |
| `-sc`, `--supercell` | Supercell passed on to `generate_structure`: 1 integer = target number of atoms (`--desired_na`), 3 integers = diagonal repetitions (`--dimensions`), 9 integers = row-major non-diagonal matrix (`--nondiagonal_dimensions`). Any other count aborts the run. |
| `-mode`, `--mode` | `vc-relax` runs the full loop — the preconditioning schedule in `iter_0`, then position and lattice steps per `iter_<n>`. `preconditioning` runs the force-constant bootstrap only, in a `preconditioning/` directory under the working directory, and stops. |
| `-pe`, `--press_ext`, `--press-ext` | Target external pressure or stress in GPa, **compression positive**: 1 value = hydrostatic with a zero shear target, 3 values = diagonal `xx yy zz`, 6 values = Voigt `xx yy zz yz xz xy`. Stored internally with the opposite sign (tension positive). The count also selects which components the convergence gate compares. |

### Thermodynamic state and ensemble

| Option | Default | Meaning |
| --- | --- | --- |
| `-t`, `--t`, `--temp`, `--temperature` | `300` | Temperature in K (integer). Used as `canonical_configuration -t` and as the Bose–Einstein factor of the phonon stress. |
| `-q`, `--quantum` | off | Sample with Bose–Einstein (quantum) statistics: adds `--quantum` to `canonical_configuration`. Without it the sampling is classical Boltzmann and carries no zero-point motion. |
| `--masses` | natural | Atomic masses for the sampling and the phonons, as `ELEMENT=MASS[,ELEMENT=MASS...]` in atomic mass units; `H=2.01410` puts deuterium on every H site. Elements not listed keep their standard atomic weight. WARP writes `infile.isotopes` in every step directory and runs `canonical_configuration` and `phonon_dispersion_relations` with `--readiso`, so the masses enter the sampled displacements and the phonon stress; the force-constant fit and `ifc_pressure` do not depend on them. Needs the TDEP fork (see [Installation](installation.md#tdep)). |
| `-n`, `--n`, `--num-samples`, `--num_samples` | `64` | Configurations sampled per position step and per lattice step. Preconditioning uses `--precond_schedule` instead; FC3-gate retries scale this by `--fc3_gate_nscale`. |

### Force constants

| Option | Default | Meaning |
| --- | --- | --- |
| `-rc2`, `--rc2` | `5.0` | Second-order force-constant cutoff in Å, passed as `extract_forceconstants -rc2`. |
| `-rc3`, `--rc3` | none | Third-order force-constant cutoff in Å, passed as `extract_forceconstants -rc3`. Without it there is no third-order fit, and therefore no phonon stress. |
| `--wz_supercell`, `--wz-supercell` | none | **Experimental and hidden from `--help`.** Passes its value, split at whitespace, to TDEP's undocumented `extract_forceconstants -wz` option, which replaces the spherical second-order cutoff by the Voronoi (super Wigner–Seitz) cell of the supercell; `-rc2` is then not passed at all. Skipped during preconditioning, where the sample sets are too small to determine that many pair shells. The run prints a `[wz][EXPERIMENTAL]` warning. Do not use it for production without validating it against a spherical-cutoff run on the same configurations. |

### Force engine

| Option | Default | Meaning |
| --- | --- | --- |
| `-c`, `--calculator` | `QE` | `QE` runs the `--pwx` command once per sample. `MLIP` instead runs `python3 -m warpt.MLIP_interface --engine <-mle> --device <-mld> --ml_params <-mlp>`, plus `--variant <-mlv>` and `--conservative` when those are given. `VASP` runs `python3 -m warpt.VASP_interface --template <--vasp_template>` with the command from `--vasp_cmd`; see [Force engines](forces.md#vasp). |
| `-pwx`, `--pwx` | `mpirun -n 6 pw.x` | The command that runs one SCF when `--calculator QE`; `-input <file>` is appended to it. Ignored with `--calculator MLIP` or `VASP`. A `python3 -m warpt.MLIP_interface ...` command may be given here instead, to reach backend options that `-c MLIP` does not expose. |
| `-mle`, `--ml_engine` | `upet` | Forwarded as `--engine` to the backend; `upet` or `mace`. Used only with `--calculator MLIP`. |
| `-mlp`, `--ml_params` | none | Path to the model parameters, a UPET `.ckpt` or a MACE `.model`, forwarded as `--ml_params` and made absolute. Required with `--calculator MLIP`; the run stops immediately without it. |
| `-mld`, `--ml_device` | `cpu` | Forwarded as `--device` to the backend; `cpu` or `cuda`. Used only with `--calculator MLIP`. |
| `-mlv`, `--ml_variant` | none | Forwarded as `--variant`: the UPET energy head to evaluate, e.g. `lah` for a checkpoint fine-tuned on the head `energy/lah`. Without it a fine-tuned checkpoint silently evaluates its base head. Used only with `--calculator MLIP`. |
| `-mlc`, `--ml_conservative` | off | Forwarded as `--conservative`: UPET forces and stress as energy gradients instead of the direct heads. Used only with `--calculator MLIP`. |
| `--vasp_template` | none | Directory with `INCAR` (a single point: `NSW = 0`, `ISIF` ≥ 2), `KPOINTS` for the **supercell** (unless `INCAR` sets `KSPACING`) and one `POTCAR_<element>` per element. Checked against the elements of `-ucp` before any sampling. Required with `--calculator VASP`. |
| `--vasp_cmd` | none | The command that runs VASP in a sample directory, e.g. `srun vasp_std`. Required with `--calculator VASP`; passed to the backend through the `WARPT_VASP_CMD` environment variable. Samples run `MAX_CONC` at a time, which defaults to 1 for VASP. |

### Preconditioning

| Option | Default | Meaning |
| --- | --- | --- |
| `-ps`, `--precond_schedule`, `--precond-schedule` | `8,32,32,48,48,64,64` | Comma-separated sample counts for the preconditioning geometries, for example `20,40,40,60,60`. Counts that are multiples of the `MAX_CONC` / `MAX_CONC_MLIP` concurrency ([Environment variables](environment.md#concurrency)) avoid idle processes. |
| `-mf`, `--max_freq`, `--maximum_frequency` | not passed | Maximum phonon frequency in THz (integer) for the `geom_0` bootstrap, passed as `canonical_configuration --maximum_frequency` while sampling from the bootstrap (Debye-model) force constants. |
| `-Td`, `--Tdebye`, `--debye_temperature` | not passed | Debye temperature in K (integer) for the `geom_0` bootstrap, passed as `canonical_configuration --debye_temperature` in the same situation. |
| `-fomf`, `--firstorder_min_freq`, `--firstorder-min-freq` | `-1.0` | Preconditioning only. From `geom_2` on, the first-order structure update is fitted only when the minimum frequency of the previous geometry's dispersion exceeds this value in THz; otherwise that geometry refines the force constants by sampling alone, because a Newton step on an indefinite second-order Hessian points uphill along the imaginary branches. Position and lattice steps always fit with the first-order update. |

### Position loop

| Option | Default | Meaning |
| --- | --- | --- |
| `--fc1_mean_tol`, `--fc1-mean-tol` | `0.030` | Exit criterion: mean per-atom residual force in eV/Å. The loop ends when the mean and the maximum are both below their tolerances. It cannot usefully be tighter than the calculator's own force error; 0.0026 corresponds to `forc_conv_thr = 1e-4` Ry/bohr in Quantum ESPRESSO. |
| `--fc1_max_tol`, `--fc1-max-tol` | `0.075` | Exit criterion: maximum per-atom residual force in eV/Å. Guards against a single stuck atom; keep it about 2.5 times `--fc1_mean_tol` so that the mean stays the binding test. |
| `-wd`, `--wyckoff_damping`, `--wyckoff-damping` | `0.8` | Fraction of the projected first-order displacement applied per structure update. Lower it to 0.1–0.3 if the structure runs away. |
| `-met`, `--metric` | `frac_rmsd` | Metric comparing a step's input and updated unit cell, used with `--reuse_threshold` to decide whether the previous step's samples are merged into the current fit. `frac_rmsd`, `max_comp_frac` and `pair_spectrum` are dimensionless fractional measures; `cart_rmsd` and `max_atom` are in Å. |
| `-rt`, `--reuse_threshold`, `--reuse-threshold` | `0.002` | Threshold on `--metric`, fractional or in Å depending on it, below which a step's structure update counts as small and its samples are merged into the next fit (the next position step, or the lattice step). Never applied to the first position step after a lattice update, which always samples fresh. |

### Lattice step and convergence

| Option | Default | Meaning |
| --- | --- | --- |
| `-pct`, `--press_conv_th`, `--press-conv-th`, `--pressure_convergence_threshold` | `0.005` | Stress convergence gate in GPa: every compared component of the stress residual must be below it — all six in `scalar` and `full6` mode, the three diagonals in `diag3` mode. The same value gates the change of the phonon stress between lattice steps. |
| `-ld`, `--lattice_damping`, `--lattice-damping` | `1.0` | The damping $\lambda_l$ of the lattice update, so that the strain is $-\lambda_l S(\sigma_\mathrm{int}-\sigma_\mathrm{ext})$. 1.0 is the full elastic step; it is multiplied further only by the two options below, when they are given. |
| `-ec`, `--elastic_constants`, `--elastic-constants` | none | A file with the 6×6 elastic-constant (stiffness) tensor in GPa, Voigt order $xx, yy, zz, yz, xz, xy$, in the Cartesian frame of the unit cell; `#` lines are comments. Its inverse is the compliance $S$ of every lattice update, in place of `outfile.compliance_tensor`. TDEP's compliance is the clamped-ion one of the fitted force constants: the atoms do not relax inside the strained cell, which makes it far too stiff where they would (along $c$ in brucite it is ~10× the measured $C_{33}$). The file is checked before the run starts: 6×6, symmetric, positive definite. |
| `--lattice_damping_ladder`, `--lattice-damping-ladder` | none | An adaptive ladder `thr1:f1,thr2:f2,...`, thresholds in GPa. When the largest compared component of the stress residual is below `thr`, the step becomes $\lambda_l\cdot f$; the rung with the smallest satisfied threshold wins. |
| `--lattice_noise_gain`, `--lattice-noise-gain` | `0.0` (off) | Noise-referenced damping: $\lambda_l$ is multiplied by $z^2/(z^2+Z_0^2)$, where $z$ is the largest stress-residual component in units of the standard error of the sample-mean stress, computed from `infile.stat`. The full step is taken far from convergence and shrinks smoothly once the residual is within a few standard errors. Recommended value: 3.0. Stored in the run context as `lattice_noise_z0`. |
| `--pct_noise_floor`, `--pct-noise-floor` | `0.0` (off) | Per-component noise floor on the stress gate: component $i$ must satisfy a residual below $\max(\texttt{-pct},\,K\cdot\mathrm{SE}_i)$, with the three shears sharing their largest standard error. Also applied to the phonon-stress change gate. Recommended value: 2.0. Stored in the run context as `press_conv_noise_k`. |

### The FC3 consistency gate

Experimental, and off unless a threshold is set. The real-space third-order stress
$\sigma_\mathrm{FC3}$ and the reciprocal-space phonon stress $\sigma_\mathrm{ph}$ are the same
third-order quantity evaluated twice, and as they enter the internal stress they cancel
([Method](method.md#the-internal-stress)), so their sum should nearly vanish — a few tenths of a GPa
in a healthy fit. A large sum means the third-order fit is ill-conditioned. The gate acts on
lattice steps only, never in preconditioning.

| Option | Default | Meaning |
| --- | --- | --- |
| `--fc3_gate_thr`, `--fc3-gate-thr` | off | Gate threshold in GPa on the mismatch $\max|\sigma_\mathrm{FC3}+\sigma_\mathrm{ph}|$. Above it the fit is treated as ill-conditioned: the step's files are archived in `fc3_gate_attempt_<k>/` and the lattice step is resampled. Each retry costs one extra sampling of the lattice step. The attempts are logged in `outfile.fc3_gate`. |
| `--fc3_gate_max_retries`, `--fc3-gate-max-retries` | `2` | Maximum number of resamples per lattice step when the gate trips. After the last retry the attempt with the smallest mismatch is used. |
| `--fc3_gate_nscale`, `--fc3-gate-nscale` | `1.0` | Sample-count multiplier per retry: retry $k$ samples $\mathrm{round}(n\cdot\text{nscale}^{\,k})$ configurations, $n$ being `-n`. 1.0 keeps the same count; 2.0 doubles the set each retry to reduce the noise. |

## `python3 -m warpt.trajectory` — reference-free tracking of a run

Walks every step directory of a finished or running relaxation in execution order and records, per
step, the cell parameters, the volume per atom, $c/a$, the space group and the tightest tolerance
at which it holds, the residual forces, the sample count, the minimum phonon frequency, the
internal stress and its symmetry-forbidden part, the step RMSD and the coordinates of one atom per symmetry orbit. It
then writes a CSV with all of these and a six-panel figure of the cell and the moving atomic
coordinates, and prints a summary. It needs ASE and numpy; without
spglib the symmetry columns are skipped, and without matplotlib the figure is. `-h`, `--help`
prints the option list.

```bash
python3 -m warpt.trajectory .           # the run root, i.e. the directory WARP was launched in
python3 -m warpt.trajectory . --no-plot --csv steps.csv
```

| Option | Default | Meaning |
| --- | --- | --- |
| `run_dir` | `.` | Positional. The run root: the directory that holds `iter_*`. Only directories named exactly `iter_<N>` are counted as steps. |
| `--csv` | `warpt_trajectory.csv` | The CSV to write; a relative path is taken from the current directory, not from `run_dir`. |
| `--plot` | `warpt_trajectory.png` | The figure to write, PNG at 150 dpi, likewise relative to the current directory. |
| `--no-plot` | off | Write the CSV and print the summary, but draw no figure. |
| `--no-show` | off | Write the figure but do not open it in a viewer. By default the finished figure is opened. |
| `--press-ext` | read from the run | Target pressure in GPa for the residual column. Without it the value is read from `warpt_run.log`, `run_warpt.sh`, `run.sh` or `job*` in the run root. |
| `--n-dofs` | `3` | How many of the most active atomic coordinates to plot. |

The tool exits with a message when it finds no step under `run_dir`. It is safe to run on a
partial run: a step is listed as soon as its `infile.ucposcar` exists, and lattice steps that have
not finished their stress evaluation are reported with `complete=False` and left out of the stress
panels. The target pressure for the residual panel is read from files in the run root, listed
under [Input files](inputs.md#optional-files-the-tools-look-for).

## `python3 -m warpt.plot_phonons_iter` — the phonons of every step

Plots the phonon dispersion (and the DOS beside it, when the step has one) of every step of a run in
execution order, all on one frequency axis, and prints the lowest and the highest frequency of each step. Steps
whose lowest frequency is below `-IMAG_TOL` have imaginary modes: their titles are red and the
summary flags them. Needs numpy, h5py and matplotlib.

```bash
python3 -m warpt.plot_phonons_iter .                  # writes phonon_plots/
python3 -m warpt.plot_phonons_iter . --overview-only --band-yaml
```

| Option | Default | Meaning |
| --- | --- | --- |
| `run_dir` | `.` | Positional. The run root: the directory that holds `iter_*`. |
| `-o`, `--outdir` | `phonon_plots` | Directory for the figures, created when missing. |
| `--overview-only` | off | Write only `overview.png`, not one figure per step. |
| `--band-yaml` | off | Also write `<step>.band.yaml` per step, as `warpt.band_yaml` does. |
| `--imag-tol` | `0.05` | A lowest frequency below minus this value, in THz, counts as imaginary. |
| `--show` | off | Open `overview.png` in a viewer after writing it. |

A step is read from `outfile.dispersion_relations.hdf5` (and `outfile.phonon_dos.hdf5`) or, when
it kept only TDEP's text output, from `outfile.dispersion_relations`, its `.gnuplot` file for the
labels and `outfile.phonon_dos`. Steps with neither are skipped; the tool exits with a message
when no step has a dispersion.

## `python3 -m warpt.band_yaml` — phonopy band.yaml from one step

Converts the `outfile.dispersion_relations.hdf5` and `infile.ucposcar` of one step directory into
the `band.yaml` that phonopy writes for a band structure: the q-path with its labels, frequencies in
THz (imaginary ones negative) and the complex eigenvectors. Programs that read phonopy output, such
as `phonopy-bandplot` and CrysViz, can then show or animate the modes of that step.

```bash
python3 -m warpt.band_yaml iter_8/lattice_            # writes iter_8/lattice_/band.yaml
python3 -m warpt.band_yaml iter_8/lattice_ -o iter8.band.yaml
```

| Option | Default | Meaning |
| --- | --- | --- |
| `step_dir` | `.` | Positional. The step directory. |
| `-o`, `--output` | `band.yaml` in `step_dir` | The file to write. |

## `python3 -m warpt.static_elastic` — the elastic tensor of a potential, for `--elastic_constants`

Computes the static elastic constants of a unit cell with a machine-learned potential by
stress–strain finite differences, with the internal coordinates relaxed at every strain, and
writes them in the file format of [`--elastic_constants`](#lattice-step-and-convergence). The
lattice step of WARP is a preconditioned stress update, so the compliance only sets the size of
each step, never the converged structure; TDEP's `outfile.compliance_tensor` is the clamped-ion
compliance of the fitted force constants and is far too stiff where the atoms relax inside the
strained cell (see [Method](method.md#the-elastic-step)). The tensor is averaged over the point
group of the cell and checked to be positive definite; a 5-atom cell takes about twenty seconds.

```bash
python3 -m warpt.static_elastic -ucp start.vasp -mle upet -mlp model.ckpt -mlv brucite -mlc -o C_static.dat
python3 -m warpt -c MLIP ... --elastic_constants C_static.dat
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-ucp`, `--unit_cell` | required | The unit cell (POSCAR); the tensor refers to its Cartesian frame. |
| `-mle`, `-mlp`, `-mlv`, `-mlc`, `-mld` | as in the orchestrator | The force engine: `upet` or `mace`, the model file, the UPET energy head, energy-gradient forces, the device. |
| `--strain` | `0.005` | The finite strain $h$ of the central differences. |
| `--fmax` | `2e-4` | BFGS force tolerance (eV/Å) of the internal relaxations. |
| `--clamped` | off | Clamped-ion tensor: the atoms keep their fractional coordinates under the strain (what TDEP computes). |
| `--no-relax-start` | off | Do not relax the internal coordinates of the given cell before straining it. |
| `--no-symmetrize` | off | Skip the point-group average. |
| `-o`, `--output` | `C_static.dat` | The file to write. |

## The other tools

Six more modules have command lines of their own and are documented where they belong:

| Command | Purpose | Page |
| --- | --- | --- |
| `python3 -m warpt.MLIP_interface` | evaluate a structure with a machine-learned potential and print the result in Quantum ESPRESSO format | [Force engines](forces.md#the-machine-learned-backend) |
| `python3 -m warpt.VASP_interface` | run one VASP single point for a sampled structure and print the result in Quantum ESPRESSO format | [Force engines](forces.md#vasp) |
| `python3 -m warpt.upet_relax` | symmetry-constrained static relaxation onto a potential's own minimum at a target pressure | [Force engines](forces.md#starting-from-the-potentials-own-minimum) |
| `python3 -m warpt.get_frames` | collect a run's structures, in run order, into one trajectory and report the space group of each at several tolerances | [Run directory and resume](run-directory.md#the-trajectory-csv) |
| `python3 -m warpt.structconv` | convert between POSCAR and Quantum ESPRESSO formats | [Structure converter](structconv.md) |
| `python3 -m warpt.Wyckoff_Projector` | project TDEP's first-order displacements onto the Wyckoff parameters of a structure and write the updated structure | [Method](method.md#wyckoff-projection) |
