# Troubleshooting

Every message below is one WARP can actually print, quoted as the code writes it. Most fatal
errors go to standard error with the prefix `Error in WARP:` and end the process with exit
status 1; the few raised as Python exceptions (marked below) appear as the last line of a
traceback. Everything tagged `[warn]` or `[info]` is printed to standard output and the run
carries on. Use this page as a lookup: find the line in the log, read the cause, apply the remedy.

## Starting up

### `Master SCF file not found: ...`

The path given to `-m` does not exist, resolved against the launch directory. The master deck is
the only input file whose existence is checked at start-up for every calculator; the `-ucp` file
is read up front only with `--calculator VASP` (see below). Otherwise a missing `-ucp` file
surfaces later, on a fresh run as a `FileNotFoundError` when the preconditioning stage copies it.
Check the spelling and the working directory.

### `--ml_params is required when --calculator MLIP`

`--calculator MLIP` has no model to evaluate with. Give `-mlp /path/to/model`, or use
`--calculator QE` with a `--pwx` command that runs the backend.

### `--vasp_template and --vasp_cmd are required when --calculator VASP` / `VASP template: ...`

The first means one of the two options is missing. The second quotes the template check: a
missing `INCAR`, `KPOINTS` (when `INCAR` sets no `KSPACING`) or `POTCAR_<element>`, an `INCAR`
with `NSW` > 0 or `ISIF` < 2, or a `-ucp` file that cannot be read. See
[Force engines](forces.md#vasp).

### `--press_ext must have 1, 3, or 6 components`

`--press_ext` takes one value (hydrostatic), three (the diagonal `xx yy zz`) or six (the full Voigt
vector). Any other count is rejected.

### `Supercell requires 1, 3 or 9 integers`

`-sc` takes one integer (a target atom count), three (diagonal repetitions) or nine (a row-major
matrix). A non-integer entry ends the run with a Python `ValueError` traceback instead.

## Missing external programs

WARP does not check that the TDEP binaries exist; the stage that needs one fails instead. Any of
the following on a first run points at the `PATH`:

| Message | Missing or failing tool |
| --- | --- |
| `generate_structure failed (no outfile.ssposcar)` | `generate_structure` |
| `canonical_configuration failed (rc=...)`, or a `FileNotFoundError` naming it when it is not installed | `canonical_configuration` |
| `extract_forceconstants failed (rc=...) with no outfile.forceconstant. See extract_forceconstants.log` | `extract_forceconstants` |
| `ifc_pressure failed` | `ifc_pressure` |

```bash
command -v generate_structure canonical_configuration extract_forceconstants \
          ifc_pressure phonon_dispersion_relations
```

A non-zero return code from a tool that *is* installed means the tool itself failed: read
`extract_forceconstants.log` in the step directory, or rerun the tool by hand there.

## Sampling and the calculator

### `No contcar_conf* files found after canonical_configuration/resume.`

The sampler reported success but wrote nothing. Usually the force constants it sampled from are
unusable, or the step directory is missing `infile.ucposcar` / `infile.ssposcar`.

### `SCF/MLIP/VASP run failed for <output> (no 'JOB DONE.')`

A force-evaluation process exited without the completion marker. This is a `RuntimeError`, not a
`die`, and it names the output file. When that file contains a `JOB FAILED.` line, its reason is
appended to the message after a colon. Otherwise open the file: for Quantum ESPRESSO the cause is
usually a failed SCF, a missing pseudopotential or an out-of-memory kill. For a chunked batch of
the machine-learned backend the file is the chunk's `mlip_batch.<k>.list.log`, whose
`[MLIP_interface] Error on <input>: ...` lines name the failing structures; each of those
structures' own outputs ends in `JOB FAILED. <error>`.

Processes still running when this happens are not terminated.

### `Calculator failed for <output>: <reason>`

The collector found a `JOB FAILED.` marker while waiting. The reason quoted is the backend's own
exception; the run aborts at once rather than waiting for a completion marker that will never
appear.

### `Calculator did not provide stress; required (use --allow_no_stress to force zeros).`

Printed by the machine-learned backend after `JOB FAILED.`: the calculator returns no stress, for
instance because the model has no stress output. A variable-cell relaxation is *driven* by that
stress, so forcing zeros gives a meaningless lattice step; use a model that provides a stress
instead.

### `Timeout waiting for SCF outputs. Missing: [...]`

Only possible when a caller sets a timeout on the collection barrier; the command-line workflow
waits indefinitely, so this message comes from custom code that calls the collector.

### `canonical_configuration` rejects `--readiso`

`--masses` was given, but the `canonical_configuration` on the path is an upstream build without
the `--readiso` switch. Install the TDEP fork named under [Installation](installation.md#tdep), or
run without `--masses`.

## Collecting the outputs

These mean the calculator output is not in the format the parsers expect:

- `Could not parse number of atoms from <output> (...)` — no `number of atoms/cell` line.
- `Could not parse total energy from <output>` — no `!` total-energy line.
- `Mismatch natoms vs parsed data for <output>/<input>: natoms=..., positions=..., forces=...` — the
  number of force lines or position lines does not match the atom count. With Quantum ESPRESSO
  this almost always means forces were not printed at all; switch `tprnfor` on in the master deck.
- `Could not find 'total stress' block in QE output` — a `RuntimeError` from the stress parser.
  Switch `tstress` on.
- `No SCF input files (*.samplesN) found; nothing to collect.` — the collector ran in a directory
  where the sampling stage never wrote anything.

## The force-constant fit

### `extract_forceconstants failed (missing/empty outfile.forceconstant)`

The fit produced no force constants. Read `extract_forceconstants.log` in that directory; the usual
causes are too few configurations for the requested cutoffs, or a cutoff larger than the supercell
supports.

### `[warn] extract_forceconstants exited rc=... but wrote a valid outfile.forceconstant (likely a benign IEEE_OVERFLOW at exit); continuing.`

Not an error. Some builds raise a floating-point flag at termination, *after* the fit has been
completed and written. WARP treats a non-zero exit as fatal only when the expected output is
actually missing, rather than throwing away a finished fit.

### `[warn] extract_forceconstants did not write outfile.new_ucposcar; using infile.ucposcar as fallback.`

No structure update was produced, so the step keeps its input structure. Expected whenever the fit
ran without the first-order update.

### `[info] no --rc3 or mode 'second'; skipping third-order extraction.`

Informational. Without `--rc3` there is no third-order fit — and therefore, later,
`[info] No FC3 → σ_ph = 0`: no Grüneisen stress at all. To include the anharmonic stress, give
`--rc3`.

## The Wyckoff projection

The projector refuses to commit a structure that fails one of its three guardrails, and WARP turns
that refusal into a fatal error:

```text
[wyckoff] symmetry guardrail violated -> aborting:
```

followed by one of:

| Guardrail | Message | Meaning |
| --- | --- | --- |
| displacement | `[guardrail:displacement] Orbit [...] has unphysically large displacements (threshold = 2.00 Å):` | one atom of an orbit moved more than 2 Å in a single update |
| proximity | `[guardrail:proximity] Atoms are unphysically close (threshold = 0.30 Å):` | the update brought two atoms within 0.3 Å |
| space group | `[guardrail:spacegroup] Projection changed space group: expected N, got M` | the staged structure no longer has the space group the step started in, at a tolerance of 10⁻⁵ Å; this includes a structure that reaches a higher-symmetry group within one update |

The first two normally mean the structure update is running away: lower `--wyckoff_damping` to
0.1–0.3, or enlarge the ensemble so the residual force it is built from is less noisy. All three
also point at a starting structure that was not a proper minimum of the force engine.

### `[wyckoff] projection failed -> aborting (no silent fallback): ...`

Any other failure of the projector, most often

```text
orbit [...] (spglib letter '...') is reproduced by no Wyckoff record of space group N in the table (multiplicity M)
```

The orbit match is setting-dependent: the projector tries the table record of spglib's Wyckoff
letter and then every other record of the group, but applies no transformation to a different
setting, so a cell in a setting the table does not carry fails here. Re-standardise the starting
cell — for instance with `refine_structure` — and start again. TDEP's own unprojected structure
update is deliberately **not** used as a fallback, because it would quietly break the symmetry.

### `[sanity] ... is unphysical: min interatomic distance ... A < 0.50 A -- aborting (soft-mode runaway in the first-order structure update?).`

The committed structure has atoms closer than 0.5 Å. This check is stricter than the projector's own
0.3 Å guardrail and applies to the committed structure whether it came from the projection or from
TDEP's update. The bracketed guess is usually right: a soft or imaginary mode has driven the
first-order update. Lower `--wyckoff_damping`, raise the sample count, or run a longer
preconditioning schedule so that the fit is stable before positions are moved.

## Preconditioning

### `Preconditioning failed: unstable phonons after full schedule`

The last geometry of the schedule still failed the gate: either the minimum phonon frequency is
not above $-10^{-5}$ THz, or the second-order cross-validation $R^2$ is below 0.60. The gate lines
printed for that geometry show which:

```text
[preconditioning] min(freq) = ...
[preconditioning] cross-val R^2: 2nd=... 3rd=... (gate: 2nd >= 0.6)
```

If the frequency is the problem, the structure may be genuinely unstable at that pressure and
temperature, or the bootstrap was badly scaled — set `--max_freq` or `--Tdebye` to a realistic
value. If the cross-validation is the problem, the fit needs more configurations: extend the
schedule with `--precond_schedule`, for instance `16,32,64,128,256`.

A related, non-fatal line is

```text
[preconditioning] phonons stable but 2nd-order CV R^2 (...) < 0.6 → keep sampling (likely overfit)
```

which is the schedule doing its job: a small ensemble can produce a real-looking spectrum that the
data do not support.

### `[preconditioning][warn] third-order CV R^2 is negative (third-order fit likely overfit)`

The third-order fit is worse than predicting zero. It does not stop the run, but the phonon stress
built from those force constants is not trustworthy. Increase the sample count or reduce `--rc3`.

### `Preconditioning failed: phonon_dispersion_relations did not produce outfile.dispersion_relations.hdf5`

No q-grid in the ladder produced a valid dispersion file. Check that
`phonon_dispersion_relations` runs by hand in that geometry directory.

## The lattice step

### `Failed to generate grid_dispersions.hdf5`

A `RuntimeError`: none of the Grüneisen q-grids succeeded, so the phonon stress cannot be computed.
It is preceded by `[σ_ph] Running: ...` lines showing each attempt.

### `[WARN] σ_ph failed: ...`

The phonon-stress evaluation raised; the exception is re-raised immediately afterwards, so this
line is the explanation for the traceback that follows.

### `[lattice][warn] strain projection skipped (...); using the unprojected strain`

spglib could not be imported, could not read the cell, or found no symmetry, so the strain was
applied without being projected onto the invariant subspace. The run continues, but the cell is
now free to drift off its symmetry — treat any step with this line as suspect.

### `[lattice] noise gain requested (z0=...) but infile.stat is unreadable here (... rows) -> damping kept at ...`

`--lattice_noise_gain` needs the per-configuration stresses to estimate their standard error, and
that needs at least three rows and thirteen columns in `infile.stat`. With fewer, the gain is
skipped and the step is taken with the damping printed at the end of the line. Sample more
configurations.

### The pressure gate never closes

There is no error for this; the run simply keeps iterating, with a line such as (for a hydrostatic
target)

```text
  pressure : scalar elementwise |Δσ| = [...] GPa; max = ... GPa -> NOT OK
```

at every lattice step. Two causes are common.

The first is that `-pct` is below the resolution of the ensemble. The residual is a sample mean, so
it cannot be driven below its own standard error; asking for 0.005 GPa from an ensemble whose
standard error is 0.05 GPa is asking for noise. Either raise `-pct`, or raise `-n`, or switch on
`--pct_noise_floor 2`, which judges each component against
$\max(\texttt{-pct},\,K\cdot\mathrm{SE}_i)$ and prints the gate it actually used:

```text
[convergence] pressure gate per component = max(pct ..., ... x SE) = [...] GPa (SE from N samples)
```

The second is that the step keeps overshooting because the cell is being moved by sampling noise.
`--lattice_noise_gain 3` shrinks the step smoothly once the residual is within a few standard
errors, which lets the last steps average over several sample sets instead of copying the noise
into the cell.

The stress gate is also only the first of three conditions: after it holds, the change in the
phonon stress must hold as well, and only then is the volume guard evaluated. Lines such as

```text
  volume   : volume guard skipped (requires pressure + ph_sigma convergence)
```

are normal until then.

### `[fc3-gate][WARNING] lattice step taken with |σ_FC3+σ_ph| above the threshold (... GPa)`

Only with `--fc3_gate_thr`. The retries were exhausted and the step was taken with the least-bad
attempt. The real-space and reciprocal-space third-order stresses should nearly cancel; a
persistent mismatch means the third-order fit is ill-conditioned. Raise `-n`, raise
`--fc3_gate_nscale` so each retry samples more, or reduce `--rc3`.

### `[fc3-gate] cannot restore attempt <k>: ... missing`

The archived attempt the gate wanted to restore is not there — usually because a
`fc3_gate_attempt_*` directory was removed by hand. Delete the whole `lattice_` directory and let
it be recomputed.

## Warnings that are not problems

| Line | Why it appears |
| --- | --- |
| `WARNING: Using awk` | `mawk` was not found, so the slower `awk` is used for the output parsing (with `mawk` present the line reads `WARNING: Using mawk`). Harmless. |
| `[position][warn] phonon check failed, continuing based on FC1 only` | no q-grid produced a dispersion for this position step; the exit test uses the residual forces alone, which is what it is based on anyway |
| `[wz][info] --wz_supercell set but SKIPPED during preconditioning (underdetermined bootstrap; see tdep_extract_fc.py).` | the experimental Voronoi cutoff is deliberately not used while the ensembles are small |
| `[cleanup] removed outfile.grid_dispersions.hdf5 (sigma_ph extracted)` | routine: that file is very large and is regenerated when the lattice step is rerun |
| `[samples] prev RMSD=... thr=... -> fresh (lattice changed)` | the first position step after a lattice update always samples fresh |

## The analysis tools

### `no WARP steps found under <dir>`

`python3 -m warpt.trajectory` found no directory named exactly `iter_<N>` holding a readable
`infile.ucposcar`. Point it at the run root, not at a step directory.

### `not a directory: <root>`

`python3 -m warpt.get_frames` was given a `-r` that is not a directory. It also exits 1 when no
frame was found at all.

### `[trajectory] SPACE GROUP CHANGED during the run: ... -> ...`

Not an error, but it deserves a careful look. Because the updates cannot lower symmetry, a change
here is either a rise onto a supergroup — the physically interesting case — or an artefact of the
tolerance ladder the tool scans. Compare with the `tightest_symprec` column before drawing a
conclusion.

## The structure converter

The converter raises these as Python exceptions (`ValueError`, or `RuntimeError` for the
atom-order check).

| Message | Cause |
| --- | --- |
| `structconv: POSCAR must list species symbols on line 6 (VASP5 format)` | a VASP4 POSCAR, with counts where the symbols belong |
| `structconv: atom order would change on write -- refusing` | the invariant that protects the force-to-atom mapping was about to be broken |
| `structconv: only ibrav=0 and ibrav=1 are supported (got N)` | any other Bravais-lattice code in a deck being read |
| `structconv: ibrav=0 but no CELL_PARAMETERS found` | the deck declares `ibrav = 0` without a cell |
| `structconv: alat (celldm(1)) required to scale CELL_PARAMETERS` / `structconv: ibrav=1 needs celldm(1)` | a cell given in `alat` units, or a cubic `ibrav = 1` deck, without the lattice constant |
| `structconv: could not locate nat / ATOMIC_POSITIONS` | the deck has no atom count or no positions block |
| `structconv: output 'pwscf' needs a seed/master QE input` | writing a QE input without giving a seed |
| `structconv: no "Begin final coordinates" block (not a relax/vc-relax output?)` | reading `pwscf_out` from an SCF output, or from a run that did not finish |
| `structconv: singular lattice (zero volume)` | the lattice in the file cannot be inverted |

## The static relaxation tool

| Message | Meaning |
| --- | --- |
| `[upet_relax] checkpoint not found: ...` | the `--ml_params` path does not exist; exit status 1 |
| `[upet_relax][warn] checkpoint is named '...'. UPET infers the architecture from the FILENAME; rename it to e.g. pet-omat-l-v1.0.0.ckpt if the load fails or the numbers look wrong.` | a warning only, printed for any name not of the form `pet-omat-<size>-v<version>.ckpt` with size `s`, `m`, `l` or `xl`; the name is tested as given, not after symbolic links are resolved |
| `[upet_relax] GATE FAILED -- inspect the structure before using it as the -ucp for WARP; an automated scan should skip this point.` | the acceptance gate failed; the structure is still written, and the exit status is 3 |

## Still stuck

Two moves solve most of what is left:

1. **Reproduce the failing stage by hand.** Every step directory is self-contained. Change into it
   and run the tool WARP ran — `extract_forceconstants`, `ifc_pressure`,
   `phonon_dispersion_relations` — with the same arguments; the error is usually much clearer from
   the tool itself.
2. **Inspect the step's own numbers.** `python3 -m warpt.trajectory <run root>` puts the residual
   forces, the minimum frequency, the ensemble size, the stress and the symmetry-forbidden stress
   of every step into one table, which is normally enough to see where a run started to go wrong.
