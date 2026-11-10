# Quick start

This page walks through the smallest complete WARP run: the two files to put in a directory, the
one command that starts the relaxation, what it prints and what it writes, and how to tell whether
it has finished. [Input files](inputs.md) gives the full rules for the two inputs and
[Command line](cli.md) lists every option.

## 1. Prepare the directory

WARP takes the directory it is launched in as the run root and creates everything below it. Put
two files there:

```text
run_300K/
├── master.scf.in     # the Quantum ESPRESSO deck used as a template for every sample
└── start.vasp        # the starting unit cell, VASP5 POSCAR
```

`master.scf.in` is an ordinary `pw.x` SCF input for the **unit cell**, with the pseudopotentials,
cutoffs, smearing and k-point mesh of the calculation, and with forces and stresses switched on.
WARP substitutes `nat`, the cell and the atomic positions of each sampled supercell into a copy of
it and rewrites `outdir` and `disk_io` per sample; every other line is passed through with its
content unchanged, apart from a title-only first line, which is replaced by a timestamp banner.
That includes the k-point mesh, which is therefore applied to the supercell: choose it for the
supercell given with `-sc`, not for the unit cell.

The deck is required with every force engine. With a machine-learned potential or VASP, each
sample is still written as a `pw.x` input built from it, and the backend reads the cell, the
positions and the species from that file; the pseudopotentials, cutoffs, smearing and k-points in
it are then not used.

`start.vasp` is the unit cell the run starts from, in VASP5 POSCAR format (species symbols on
line 6).

!!! tip "Start from the right structure"
    The starting cell should already be the static minimum of the **same force engine** at the
    **same target pressure**. Residuals that come from a mismatch between the starting geometry
    and the engine are not thermal, and WARP spends iterations chasing them. For a UPET potential,
    `python3 -m warpt.upet_relax` produces exactly this geometry and checks it against an
    acceptance gate; see
    [Starting from the potential's own minimum](forces.md#starting-from-the-potentials-own-minimum).

## 2. Run it

```bash
export PATH="/path/to/tdep/bin:$PATH"

python3 -m warpt \
    --calculator QE -pwx "mpirun -n 8 pw.x" \
    -m master.scf.in -ucp start.vasp \
    --mode vc-relax \
    -t 300 --press_ext 0.0 \
    -sc 2 2 2 -Td 400 \
    -n 64 -rc2 5.0 --rc3 3.5 \
    -pct 0.05 --quantum \
    2>&1 | tee -a warpt_run.log
```

Five options are required: `-m`, `-ucp`, `-sc`, `--mode` and `--press_ext`. Everything else has a
default, although a fresh run also needs `-Td` or `-mf` (see the note below). What the rest of
this command sets:

- `-t 300` — the temperature in K (an integer), used both for the sampling and for the
  Bose–Einstein factor of the phonon stress;
- `--press_ext 0.0` — the target pressure in GPa, compression positive; one value means
  hydrostatic with zero shear;
- `-sc 2 2 2` — the supercell, here 2×2×2 repetitions (one integer means "about this many atoms",
  nine integers a non-diagonal matrix);
- `-Td 400` — a Debye temperature in K (an integer) that scales the bootstrap force constants of
  the first preconditioning geometry;
- `-n 64` — configurations per position step and per lattice step (64 is also the default);
- `-rc2` / `--rc3` — the second- and third-order force-constant cutoffs in Å. Without `--rc3`
  there is no third-order fit, and then there is no phonon stress either;
- `-pct 0.05` — the stress convergence gate in GPa (default 0.005), which also bounds the change
  of the phonon stress between lattice steps;
- `--quantum` — sample with Bose–Einstein statistics. Without it the sampling is classical.

Saving the output in `warpt_run.log` keeps a record of the run, and the file is also one of the
places `python3 -m warpt.trajectory` reads the target pressure from.

!!! note "Starting the bootstrap"
    The very first geometry, `geom_0`, has no force constants yet. It samples from a model set
    that `canonical_configuration` builds from a pair potential scaled to a Debye temperature
    (`-Td <K>`, passed on as `--debye_temperature`) or to a maximum phonon frequency
    (`-mf <THz>`, passed on as `--maximum_frequency`); both values are integers. Give one of
    them on a fresh run. With neither, WARP passes neither flag on, `canonical_configuration`
    looks for an `infile.forceconstant` that `geom_0` does not have, and the run stops there.

The `warpt` console script is equivalent to `python3 -m warpt` once the package is installed.

## 3. What the run prints

The run opens with the banner, the `awk` it picked and the supercell it will build, then the
restart check:

```text
WARNING: Using mawk
  Using a 2x2x2 supercell
[restart] No incomplete run detected; starting fresh

=== TDEP_vc-relax Iteration 0 ===
  Creating directory iter_0!

=== Preconditioning ===
  Creating directory preconditioning!
[preconditioning] sample schedule = [8, 32, 32, 48, 48, 64, 64]
  Creating directory geom_0!
 Using Bose-Einstein statistics
[canonical] canonical_configuration -t 300 -n 8 --debye_temperature 400 --quantum
```

From the second geometry on, each preconditioning geometry ends with its gate:

```text
[preconditioning] min(freq) = ...
[preconditioning] cross-val R^2: 2nd=... 3rd=... (gate: 2nd >= 0.6)
[preconditioning] stable AND well cross-validated → proceed
```

Then the position loop runs, one block per step, each ending in the residual force it is judged
on:

```text
============================== Position update 0 ==============================
...
[position] FC1 residual: mean = ... meV/A (tol 30.00), max = ... meV/A (tol 75.00)
```

and when both are under tolerance:

```text
[position] converged: mean and max FC1 below tolerance after 2 step(s)
```

The lattice step prints the stress decomposition, the convergence verdict, the damped and
symmetry-projected strain and the new cell:

```text
============================== lattice update ==============================
...
=== STRESS DIAGNOSTIC ===
σ_DFT: ...
σ_FC2: ...  (+, nuclear kinetic / momenta term)
σ_FC3: ...  (+, real-space Grüneisen; ~cancels σ_ph)
σ_0  : ...  (= σ_DFT + σ_FC2 + σ_FC3, all-plus)
σ_ph : ...
σ_int: ...
σ_ext: ...
=========================

[fc3-gate] |σ_FC3+σ_ph| max = ... GPa (gate off)
[convergence]
  pressure : scalar elementwise |Δσ| = [...] GPa; max = ... GPa -> NOT OK
  ph_sigma : phonon elementwise |Δσ_ph| = N/A (no previous σ_ph) -> N/A
  volume   : volume guard skipped (requires pressure + ph_sigma convergence)
[lattice] damping lambda_l=1.000 (|Δσ|=... GPa) (no ladder: full step scaled by lambda_l)
[lattice] strain projected onto the invariant subspace of ... point-group operations; symmetry-forbidden component removed = ...

=== LATTICE UPDATE ===
```

followed by the handover to the next iteration:

```text
...
[volume-guard] recorded last lattice step |ΔV/V| = ...
[lattice] Restart prepared: .../vc_restart.ucposcar
```

From the second iteration on, a timing table is printed at the top of every iteration (unless
`WARPT_TIMING=0`), so even a run killed by a scheduler leaves one behind.

## 4. What lands on disk

```text
run_300K/
├── master.scf.in
├── start.vasp
├── vc_restart.ucposcar        # copy of the newest updated cell, rewritten every lattice step
├── iter_0/
│   ├── preconditioning/
│   │   ├── infile.ucposcar, infile.ssposcar, outfile.ssposcar
│   │   └── geom_0/ geom_1/ ...     one directory per schedule entry, up to the first that passes the gate
│   ├── position_0/ position_1/ ... one directory per position step
│   └── lattice_/                   the single lattice step of this iteration
└── iter_1/
    └── ...
```

Every step directory holds its own sampled configurations (`contcar_conf*`), the calculator input
and output of each sample (`<master>.samples<n>` and `<master>.samples<n>.out`), the collected
TDEP inputs (`infile.positions`, `infile.forces`, `infile.stat`, `infile.meta`) and the fitted
force constants. [Run directory and resume](run-directory.md) lists them all.

## 5. Telling whether it finished

A finished run prints

```text
Converged.
```

and writes `final_structure.vasp` into the `lattice_` directory of the last iteration. That file
is the result: the relaxed unit cell at the requested temperature and pressure. Nothing else marks
completion — WARP writes no lock or status file — so a run that stops without that line stopped
early, either because it hit an error or because the process was killed. A converged run that is
started again in the same directory walks forward to that file and prints
`[lattice][resume] iteration already converged here.` instead.

To see the whole path in one place afterwards:

```bash
python3 -m warpt.trajectory .        # writes warpt_trajectory.csv and warpt_trajectory.png
python3 -m warpt.get_frames -r .     # writes trajectory.xyz and frame_NNNN_<step>.vasp
python3 -m warpt.plot_phonons_iter . # writes phonon_plots/: one dispersion per step and overview.png
```

The [post-processing showcase](postprocessing.md) runs these tools on a finished run and shows
what they write.

## 6. Resuming

Run **exactly the same command in the same directory**. WARP walks the existing tree forward and
skips what is already complete: finished configurations are reused, finished calculator outputs
are recognised by their `JOB DONE.` marker and not recomputed, and a finished lattice step is
detected from the file it wrote. A resumed run announces itself:

```text
[restart] Detected previous incomplete run
[restart] iteration = 1
[restart] kind      = lattice
[restart] dir       = .../iter_1/lattice_
[restart] stage     = phonons
[restart] incomplete lattice detected at .../iter_1/lattice_; resuming via idempotent forward walk
```

!!! warning "Same directory, same options"
    The resume logic reads the files on disk, not the command line that made them. If a directory
    is restarted with a different temperature, cutoff or supercell, the finished stages are reused
    as they are and the run silently mixes settings. Start a new directory whenever anything
    physical changes.

The `[restart]` lines are a report, not a plan of action: the continuation itself is the forward
walk, and the `stage` field does not always mean that work is missing. A *finished* `lattice_`
directory is reported at stage `phonons`, because the Grüneisen grid file, which can reach
gigabytes, is deleted as soon as the phonon stress has been extracted from it, and a *finished*
`position_*` directory is reported at stage `ifc_pressure`, because position steps never run that
tool. Both are normal.
