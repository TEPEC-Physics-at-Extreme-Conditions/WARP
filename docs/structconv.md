# Structure converter

`warpt.structconv` converts crystal structures between VASP5 POSCAR and Quantum ESPRESSO formats.
It is the piece that turns each sampled supercell into a `pw.x` input during a run, and it is
usable on its own from the command line. Its central promise is that atoms are never reordered.

## Why it exists

WARP matches the forces the calculator returns to the configuration it sampled **by position in
the file**. If a converter regrouped atoms by species — as most do, to write a tidy POSCAR — the
forces would be attached to the wrong atoms and every fitted force constant would be wrong, with
nothing obviously broken to show for it.

The invariant is therefore enforced rather than assumed: every reader keeps file order, every
writer emits it unchanged, and `assert_same_order` runs on every write. If a write would reorder or
perturb the atoms, it raises a `RuntimeError` instead:

```text
structconv: atom order would change on write -- refusing
```

The POSCAR writer honours this by emitting **repeated species blocks** where it has to. An
interleaved sequence Mg, H, O, O, H is written as

```text
 Mg H O H
 1 1 2 1
```

which is valid VASP5 and preserves the order, rather than being regrouped into `Mg H O` / `1 2 2`.

## Formats

| Format name | Direction | What it handles |
| --- | --- | --- |
| `poscar` | read and write | VASP5 POSCAR. An optional `Selective dynamics` line is accepted; `Direct` and `Cartesian` coordinates are both read, and Cartesian coordinates are converted with the scaled lattice. The scale factor multiplies the lattice only. |
| `pwscf` | read and write | A `pw.x` input. Reading supports `ibrav = 0`, with the lattice from `CELL_PARAMETERS` in angstrom, bohr or alat units, and `ibrav = 1`, a cubic cell from `celldm(1)`; positions in crystal, angstrom, bohr or alat units are converted to fractional. |
| `pwscf_out` | read only | The final structure of a `pw.x` vc-relax output, taken from between `Begin final coordinates` and `End final coordinates`, with the atom count from the `number of atoms/cell` line. The final lattice may be in angstrom, bohr or alat units; positions in crystal, angstrom or bohr units are converted to fractional. A position block in alat units is not rescaled: it is read as if it were in angstrom. |

A VASP4 POSCAR, with counts on line 6 where the species symbols belong, is rejected. Any `ibrav`
other than 0 or 1 is rejected on reading. A fixed-cell relax output has no final
`CELL_PARAMETERS` block and is rejected as well. Each rejection raises a `ValueError` whose message
starts with `structconv:`.

## Command line

```bash
python3 -m warpt.structconv <in-fmt> <out-fmt> [seed] < input > output
```

The structure is read from standard input and written to standard output. `<in-fmt>` is `poscar`,
`pwscf` or `pwscf_out`; `<out-fmt>` is `poscar` or `pwscf`. Writing `pwscf` needs the third
argument, the seed file. With fewer than two arguments the command prints its usage and exits.

```bash
# a QE input's structure as a POSCAR
python3 -m warpt.structconv pwscf poscar < scf.in > POSCAR

# the relaxed structure of a vc-relax, as a POSCAR
python3 -m warpt.structconv pwscf_out poscar < relax.out > relaxed.vasp

# a POSCAR into a QE input, using an existing deck as the template
python3 -m warpt.structconv poscar pwscf master.scf.in < POSCAR > scf.in
```

Inside a run the same conversion happens in process, not through this command line: the sampling
stage calls `poscar_to_pwscf` directly.

## Seed streaming

There is no template language. Writing a `pw.x` input means streaming an existing deck — the seed —
line by line and substituting only what describes the structure. Everything else, including species
blocks, pseudopotential names, cutoffs and any namelist or keyword the converter does not know,
passes through untouched.

The substitution rules, in the order the writer applies them:

- The seed's **first line** is replaced by a timestamp banner, unless it starts with `&` or
  contains `=`, in which case it is real content and is kept directly after the banner.
- The **`nat`** line is replaced as a whole by `nat= N`, the structure's atom count, so `nat` must
  stand on a line of its own: another keyword on the same line is lost. A `nat` that does not
  start its line is not recognised and passes through unchanged.
- The **`ibrav`** line is passed through, and selects the branch below. It is taken as `ibrav = 1`
  when it contains the character `1` anywhere on the line.
- With **`ibrav = 0`**: the `CELL_PARAMETERS` header is kept, the three vectors are replaced by the
  structure's, and the seed's three old vector lines are dropped.
- With **`ibrav = 1`**: those of `celldm(1)`, `celldm(2)` and `celldm(3)` that the seed contains
  are rewritten from the lattice — $a_x$ in bohr, $b_y/a_x$ and $c_z/a_x$ — and the whole
  `CELL_PARAMETERS` block is dropped. `pw.x` reads only `celldm(1)` for `ibrav = 1`, so this
  branch is correct only for a cubic cell.
- **`ATOMIC_POSITIONS`** is rewritten as `ATOMIC_POSITIONS (crystal)` followed by the fractional
  coordinates, and the seed's own coordinate lines after it are dropped until a line starting with
  `K_POINTS`.
- While those lines are being dropped, a `CELL_PARAMETERS`, `ibrav` or `nat` line — or, with
  `ibrav = 1`, a `celldm` line — is still substituted and ends the dropping, so a deck whose cell
  block sits *after* its positions is handled correctly. After a `CELL_PARAMETERS` header only its
  three vector lines are dropped. With `ibrav = 0`, a `celldm` line met while dropping is dropped.

The rule that matters most is the first half of that: **`K_POINTS` must come after
`ATOMIC_POSITIONS` in the seed**, because anything between them that the writer does not recognise
is discarded.

## Number formats

- In a `pw.x` input, fractional coordinates are written with C `%.17g`, which round-trips a double
  exactly. This is deliberate: the sampled displacements are small, and printing them with fewer
  digits would put noise into the fitted force constants.
- Lattice vectors are written with `%15.10f`, and `celldm` values with `%.8f`.
- A negative zero is written without its sign.
- The POSCAR writer emits scale 1.0 and `Direct` coordinates, written with `%15.10f`.

## The Python API

```python
from warpt import structconv

s = structconv.read_poscar(open("POSCAR").read())        # -> Structure
text = structconv.write_pwscf(s, open("master.scf.in").read())
structconv.poscar_to_pwscf("POSCAR", "master.scf.in", "scf.in")
```

`Structure` is a named tuple of `lattice` (3×3, rows are vectors, in Å), `symbols` (one per atom, in
file order) and `frac` (fractional coordinates, in file order). `convert(text, fmt_in, fmt_out,
seed_text=None, banner=None)` is the single-call form, and `assert_same_order(src, symbols_out,
frac_out)` is the check the writers run, exposed so that callers can apply it to their own output.
The readers take file contents, not paths; only `poscar_to_pwscf` opens files. The optional
`banner` argument of the writers replaces the timestamp title line.

The module needs no third-party package: it uses the standard library and `warpt.constants`, and
nothing else, so it can be imported and used on its own in an environment without ASE or pymatgen.
A full WARP run still needs ASE: the output collector imports `warpt.io_qe` for its stress parser,
and `warpt.io_qe` imports `ase.io.espresso` when it loads; see
[Installation](installation.md#python-requirements).
