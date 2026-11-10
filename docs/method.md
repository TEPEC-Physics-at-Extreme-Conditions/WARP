# Method

This page sets out the physics of a WARP run: the effective model that is sampled, the fit that
produces its force constants, the two updates that move the structure, and the gates that stop the
loop. Everything here describes what the code does; where a formula and an implementation could be
read differently, the implementation is what is written down. The per-module API is in the
[module reference](reference/index.md).

## The effective model

WARP works on the temperature-dependent effective potential: a model Hamiltonian
$H=\sum_i \mathbf{p}_i^2/2m_i + U$ whose potential energy is a polynomial in the displacements
$u$ from a reference structure $\mathbf{R}^0$,

$$
U = U_0
  + \sum_{i\alpha}\Phi^{(1),\alpha}_{i}u^{\alpha}_i
  + \frac{1}{2!}\sum_{ij\alpha\beta}\Phi^{\alpha\beta}_{ij}u^{\alpha}_i u^{\beta}_j
  + \frac{1}{3!}\sum_{ijk\alpha\beta\gamma}\Psi^{\alpha\beta\gamma}_{ijk}
      u^{\alpha}_i u^{\beta}_j u^{\gamma}_k ,
$$

with roman indices labelling atoms and greek indices Cartesian directions. The force constants
$\Phi^{(1)}$, $\Phi$ and $\Psi$ are not derivatives of the Born–Oppenheimer surface: they are
fitted to forces computed on displaced supercells, so they are effective constants of the model at
the current reference structure and at the current temperature.

Two of them drive the relaxation. $\Phi^{(1)}$ is a constant force on the atoms, non-zero exactly
when the reference positions are not the finite-temperature equilibrium. The stress of the same
ensemble, compared with the target stress, is what moves the cell.

## Stochastic sampling

Configurations are drawn from the harmonic canonical ensemble of the *current* second-order force
constants. For mode $s$ with frequency $\omega_s$ and eigenvector $\varepsilon_{is}$ on atom $i$,

$$
u_i = \sum_{s} \varepsilon_{is}\,\langle A_{is}\rangle\,
      \sqrt{-2\ln\xi_{1s}}\,\sin 2\pi\xi_{2s} ,
$$

a Box–Muller draw of a Gaussian normal coordinate whose root-mean-square width is

$$
\langle A_{is}\rangle = \sqrt{\frac{\hbar\,(2n_s+1)}{2m_i\omega_s}} ,
\qquad
n_s = \bigl[\exp(\hbar\omega_s/k_BT)-1\bigr]^{-1} .
$$

Because $2n_s+1=\coth(\hbar\omega_s/2k_BT)$, that variance is the exact mean-square displacement of
a quantum harmonic oscillator in the canonical ensemble. It has the two limits that bracket the
problem: for $\hbar\omega_s \gg k_BT$ it becomes the zero-point amplitude
$\sqrt{\hbar/2m_i\omega_s}$, finite at $T=0$; for $\hbar\omega_s \ll k_BT$ it becomes
$\omega_s^{-1}\sqrt{k_BT/m_i}$, the classical equipartition result with $\hbar$ cancelled.

Each configuration is an independent draw, so there is no correlation time to discard and the
ensemble can be enlarged simply by drawing more.

!!! note "Quantum statistics are not the default"
    WARP passes `--quantum` to `canonical_configuration` only when `--quantum` (`-q`) is given on
    the WARP command line. Without it the sampling is classical Boltzmann, with no zero-point
    motion. The two cost the same; they differ only in the width assigned to each mode.

The masses in the drawing are the natural ones unless `--masses` names others, for example
`--masses H=2.01410` for a deuterated crystal. The mass changes the width of every mode in the
sampling and, through the frequencies, the phonon stress; the force constants themselves and the
`ifc_pressure` decomposition do not depend on it. A deuterated hydrate sampled with the protium mass
has too large a hydrogen amplitude, and its anharmonic cell responds to that.

The drawing itself is done by TDEP's `canonical_configuration`, from `infile.forceconstant` in the
step directory. The first geometry of a run has no force constants yet and samples from a bootstrap
set, whose scale is set with `--max_freq` (a maximum phonon frequency in THz) or `--Tdebye` (a
Debye temperature in K). When neither is given, WARP passes no bootstrap flag,
`canonical_configuration` looks for an `infile.forceconstant` that `geom_0` does not have, and a
fresh run stops there; give one of the two.

## Fitting

The forces computed on the sampled configurations are fitted order by order by TDEP's
`extract_forceconstants`, with `-rc2` (default 5.0 Å) and, when it is given, `-rc3` as the pair
and triplet cutoffs; without `-rc3` no third order is fitted. The second order is fitted first; the
first-order constants are then fitted to what is left,

$$
\Phi^{(1),\alpha}_{i} = -\bigl\langle f^{\alpha}_i - f^{(2),\alpha}_i \bigr\rangle ,
$$

the mean residual force after the harmonic prediction has been subtracted; the third order is
fitted to the remainder. WARP requests the first-order fit with `--firstorder` in every position
step and every lattice step, and during preconditioning only under the condition in
[Preconditioning](#preconditioning) below.

The fit is cross-validated by `extract_forceconstants`. During preconditioning WARP reads the
second- and third-order $R^2$ back out of its log: the second-order value gates the schedule, the
third-order value is reported, with a warning when it is negative.

## The position update

Within the harmonic model, the force on atom $i$ is
$f^{\alpha}_i=-\sum_{j\beta}\Phi^{\alpha\beta}_{ij}u^{\beta}_j-\Phi^{(1),\alpha}_i$, so the
displacement that makes it vanish is the least-squares solution

$$
(u^{\alpha}_i)^{\mathrm{eq}}
  = -\sum_{j\beta}(\Phi^{+})^{\alpha\beta}_{ij}\,\Phi^{(1),\beta}_j ,
$$

with $\Phi^{+}$ the Moore–Penrose pseudo-inverse, needed because $\Phi$ is singular by
translational invariance. TDEP computes this field and writes it to
`outfile.uc_displacements`, one fractional vector per atom.

That field is **not** applied as it stands. The force constants are symmetry-reduced in the fit, so
$\Phi^{(1)}$ respects the space group to the precision of the ensemble average — but the
least-squares displacement field is built from a finite number of sampled forces, and its
statistical noise is not symmetric. Applied step after step, that noise walks a symmetric crystal
downhill in symmetry.

### Wyckoff projection

WARP therefore projects the field onto the degrees of freedom the space group allows. spglib
identifies the space group $\mathcal{G}$ of the current unit cell and partitions the atoms into
Wyckoff orbits $\mathcal{O}_a$ of multiplicity $m_a$. The site symmetry of the representative
leaves $n_a\in\{0,1,2,3\}$ free parameters, encoded in a basis
$\mathsf{A}^{(a)}\in\mathbb{R}^{3\times n_a}$: a site $(x,x,0)$ has $\mathsf{A}=(1,1,0)^{\mathsf T}$,
a site $(0,0,z)$ has $(0,0,1)^{\mathsf T}$, a site $(\tfrac14,\tfrac14,\tfrac14)$ has $n_a=0$. Any
symmetry-compatible displacement of the orbit is then

$$
\mathbf{u}^{(a)}_k = \mathbf{R}_k\,\mathsf{A}^{(a)}\,\mathbf{p}^{(a)},
\qquad k=1,\dots,m_a ,
$$

one parameter vector per orbit. Given the raw displacements $\tilde{\mathbf{u}}^{(a)}_k$, the
increment is the least-squares solution over the whole orbit,

$$
\mathbf{p}^{(a)} = \arg\min_{\mathbf{p}}
  \sum_{k=1}^{m_a}\bigl\lVert \tilde{\mathbf{u}}^{(a)}_k
  - \mathbf{R}_k \mathsf{A}^{(a)} \mathbf{p}\bigr\rVert^2 ,
$$

overdetermined because $n_a\le 3\le 3m_a$. The update is made **in parameter space**: the site
parameters move by

$$
\mathbf{p}_{\mathrm{new}} = \mathbf{p}_{\mathrm{site}} - \lambda_w\,\mathbf{p}^{(a)} ,
$$

with $\lambda_w$ the damping of `--wyckoff_damping` (default 0.8; the sign follows TDEP, which
writes a displacement $d$ and then applies $r\leftarrow r-d$), and every member of the orbit is
regenerated from the new parameters by its own integer rotation. An orbit with $n_a=0$ therefore
receives no displacement at all, and one with $n_a=1$ moves only along its own line.

Three consequences follow, and they are the reason for the name of the code:

- the update **cannot lower** the symmetry, by construction rather than by tolerance;
- it **can raise** it, because $\mathbf{p}^{(a)}$ may converge to a value at which the orbit sits on
  a special position of a supergroup;
- averaging over all $m_a$ members makes the stochastic error of $\mathbf{p}^{(a)}$ smaller than
  that of any single atom's displacement.

The residual of the fit, the part of the raw field the projection removed, is printed per orbit as
`RMS fit residual (fractional)`. It is a direct measure of the sampling noise, and therefore a
check that the ensemble is large enough.

### Guardrails

Before an orbit is committed, its staged positions must pass three checks: no atom may move more
than `max_disp_angstrom` (2.0 Å), no pair may come closer than `min_dist_angstrom` (0.3 Å), and the
space-group number of the staged structure must still be the input one. A failure raises a
`[guardrail:...]` message and ends the run. The committed structure is then checked once more,
against a stricter 0.5 Å minimum interatomic distance.

!!! note "The tolerance used inside a run"
    The pipeline calls the projector with a symmetry tolerance of $10^{-5}$ Å. The function's own
    default, and the default of its standalone command line, is $10^{-12}$ Å.

### When the position loop stops

After each position step WARP reads `outfile.forceconstant_firstorder` and takes the magnitude of
each atom's row. The loop ends when the **mean** is below `--fc1_mean_tol` (default 0.030 eV/Å)
**and** the **maximum** is below `--fc1_max_tol` (default 0.075 eV/Å). The mean is the binding
criterion; the maximum guards against a single stuck atom.

The residual cannot be driven below the force error of the calculator itself, so these tolerances
belong on the scale of that error, not below it.

## The lattice update

### The internal stress

After the fit, `ifc_pressure` writes the stress of the fitted model, split into three parts and in
the compression-positive sense. WARP negates each on reading and adds the phonon stress:

$$
\sigma_{\mathrm{int}}
  = -\bigl(\sigma_{\mathrm{DFT}} + \sigma_{\mathrm{FC2}} + \sigma_{\mathrm{FC3}}\bigr)_{\text{file}}
  + \sigma_{\mathrm{ph}} ,
$$

tension-positive throughout. Term by term:

- $\sigma_{\mathrm{DFT}}$ is the mean Born–Oppenheimer stress of the sampled configurations. The
  configurations are displaced on the full potential-energy surface, so this already contains the
  whole thermal stress, harmonic and anharmonic.
- $\sigma_{\mathrm{FC2}}$ is the virial of the fitted harmonic model on the same displacements. It
  enters with a plus sign because it supplies the nuclear-momentum contribution that the mean
  Born–Oppenheimer stress of static snapshots lacks. For a harmonic crystal it cancels the tensile
  virial of the displacements exactly, so a harmonic crystal does not expand — everything by which
  $\sigma_{\mathrm{DFT}}+\sigma_{\mathrm{FC2}}$ departs from the static stress is anharmonic.
- $\sigma_{\mathrm{FC3}}$ and $\sigma_{\mathrm{ph}}$ are the same third-order (Grüneisen) quantity
  evaluated twice: once in real space on the sampled supercell, once on a phonon grid. They cancel
  up to the difference between the two grids. The optional, experimental
  [FC3 consistency gate](cli.md#the-fc3-consistency-gate) (`--fc3_gate_thr`) uses this: when
  $\max_i\lvert\sigma_{\mathrm{FC3},i}+\sigma_{\mathrm{ph},i}\rvert$ exceeds the threshold (in
  GPa), the third-order fit is treated as ill-conditioned and the lattice step is resampled, up to
  `--fc3_gate_max_retries` times.

The reciprocal-space term is computed by WARP itself from the Grüneisen grid that
`phonon_dispersion_relations --dumpgrid --gruneisen` writes:

$$
\sigma_{\mathrm{ph}}
  = -\frac{1}{N_{\mathbf q}\,V}\sum_{\mathbf{q}s}
    \gamma_{\mathbf{q}s}\;\hbar\omega_{\mathbf{q}s}
    \Bigl(n_{\mathbf{q}s}+\tfrac12\Bigr) ,
$$

a mean over the $N_{\mathbf q}$ grid points of the mode sum, with $V$ the cell volume from the same
file and $n$ the Bose–Einstein occupation at `--temp`. The $\tfrac12$ is the zero-point term, so
$\sigma_{\mathrm{ph}}$ is finite at $T=0$. Modes with $\hbar\omega \le 10^{-8}$ eV are skipped.

If the file carries `gruneisen_tensors`, $\gamma_{\mathbf{q}s}$ is a tensor and
$\sigma_{\mathrm{ph}}$ is anisotropic; if it carries only the scalar `gruneisen_parameters`, the
scalar multiplies the identity and $\sigma_{\mathrm{ph}}$ is hydrostatic. Without third-order force
constants there is no Grüneisen grid at all and $\sigma_{\mathrm{ph}}$ is zero — WARP says so with
`[info] No FC3 → σ_ph = 0`. The grid file itself, `outfile.grid_dispersions.hdf5`, is deleted once
the stress has been extracted, because it can reach gigabytes and is regenerated whenever the
lattice step is rerun.

Conventions, in one place: `--press_ext` is given compression-positive in GPa and stored negated,
so $\sigma_{\mathrm{ext}}$, $\sigma_{\mathrm{int}}$ and `outfile.sigma_ph` are all tension-positive,
while the `outfile.stress_*` files written by `ifc_pressure` are compression-positive. WARP's Voigt
order is $xx, yy, zz, yz, xz, xy$.

### The elastic step

The compliance $S$ turns the stress residual into a strain. By default it is that of the fitted
second-order force constants, `outfile.compliance_tensor` in 1/GPa, which TDEP derives with the
atoms clamped to their fractional coordinates under the strain. Where the atoms would relax inside
the strained cell, the hydroxyl layers of brucite under a strain along $c$ for instance, that
clamped-ion compliance is much too small, and `--elastic_constants` supplies a stiffness tensor
(relaxed-ion, measured or computed elsewhere) whose inverse is used instead:

$$
\varepsilon_{\rho} = -\lambda_l\,S_{\rho\xi}\,
  \bigl(\sigma_{\mathrm{int},\xi}-\sigma_{\mathrm{ext},\xi}\bigr) ,
$$

an engineering Voigt strain, whose shear components are halved when it is restored to a symmetric
$3\times3$ tensor.

$S$ is a preconditioner. The run stops when $\sigma_\mathrm{int}=\sigma_\mathrm{ext}$ within the gate,
and that condition does not contain $S$: a compliance that is too stiff makes every step too small
and the run slow (a soft direction, such as the stacking axis of a layered crystal, then needs many
lattice steps to reach the target), a compliance that is too soft overshoots, and the damping catches that. The converged
structure is the same either way. The static relaxed-ion tensor of the potential, computed with
[`warpt.static_elastic`](cli.md#python3-m-warptstatic_elastic-the-elastic-tensor-of-a-potential-for-elastic_constants)
in seconds, or a measured tensor, is therefore the natural choice for `--elastic_constants`;
there is no need for the compliance to carry the temperature or the anharmonicity, which live in
$\sigma_\mathrm{int}$.

### Strain projection

Before it is applied, the strain is replaced by its average over the point-group operations of the
current cell,

$$
\varepsilon \;\leftarrow\; \frac{1}{|\mathcal{P}|}\sum_{R\in\mathcal{P}} R\,\varepsilon\,R^{\mathsf T},
$$

the projector, by Neumann's principle, onto the strains a crystal of that symmetry may undergo: one
independent component for a cubic cell, two for hexagonal, trigonal and tetragonal, three for
orthorhombic. The rotations come from spglib in fractional coordinates and are conjugated to
Cartesian ones with the lattice matrix. The removed part is printed at every lattice step:

```text
[lattice] strain projected onto the invariant subspace of N point-group operations;
symmetry-forbidden component removed = ...
```

and, exactly like the Wyckoff residual, it is the sampling noise of the stress and of the
compliance. As with the position update, a supergroup's invariant subspace lies inside the
subgroup's, so the projection never blocks a rise in symmetry.

If spglib is unavailable or fails, WARP prints `[lattice][warn] strain projection skipped` and
applies the unprojected strain rather than stopping.

### Applying it

The projected strain acts on the lattice matrix whose **rows** are the lattice vectors,

$$
\mathscr{A}^{n+1} = \mathscr{A}^{n}\,(\mathcal{I} + \varepsilon) ,
$$

and components of the result below $10^{-6}$ Å are set to zero. The species and coordinate block of
the POSCAR is copied through unchanged, so only the cell moves in this step.

### Damping

$\lambda_l$ starts at `--lattice_damping` (default 1.0, the full elastic step) and is multiplied by
two optional factors:

- **a ladder**, `--lattice_damping_ladder 'thr1:f1,thr2:f2,...'`, thresholds in GPa. The rung with
  the smallest threshold that the largest compared residual component still satisfies sets the
  factor. Off by default.
- **a noise-referenced gain**. With `--lattice_noise_gain Z0` greater than zero,

$$
\lambda_l \;\leftarrow\; \lambda_l\,\frac{z^2}{z^2+Z_0^2} ,
\qquad
z = \max_i \frac{\lvert \sigma_{\mathrm{int},i}-\sigma_{\mathrm{ext},i}\rvert}{\mathrm{SE}_i} ,
$$

  where $\mathrm{SE}_i = s_i/\sqrt{n}$ is the standard error of the sample-mean stress over the $n$
  configurations of `infile.stat` (the three shears share their largest). Far from convergence
  $z\gg Z_0$ and the full step is taken; once the residual is within a few standard errors the step
  shrinks smoothly, so the last steps average over several sample sets instead of copying the
  sampling noise of the stress into the cell. Off by default; the recommended value is 3.0.

### Symmetry of the updated cell

Because the strain is projected onto the point group of the current cell before it is applied, the
updated cell keeps the space group to rounding error and is used as it is; no separate
symmetrisation follows. The positions are kept in the same way by the Wyckoff projection, whose
guardrail stops the run if an update would leave the space group of the step
(`[guardrail:spacegroup]`).

## Preconditioning

Stochastic sampling needs force constants to sample from, and the first ones are a guess. The
preconditioning schedule turns that guess into a fitted, stable model before the relaxation proper
starts. It runs under `iter_0/preconditioning/`, one directory `geom_<k>` per entry of the sample
schedule (default `8,32,32,48,48,64,64`, overridden with `--precond_schedule`):

- `geom_0` samples from the bootstrap force constants;
- every later geometry samples from the force constants of the one before it, and is fitted on its
  own samples **plus** the previous geometry's snapshot — a sliding window two geometries wide;
- from `geom_1` on, each geometry must pass a gate: the minimum frequency of its dispersion above
  $-10^{-5}$ THz, **and** the second-order cross-validation $R^2$ at least 0.60. The first geometry
  that passes ends the schedule; if the last one fails, the run stops with
  `Preconditioning failed: unstable phonons after full schedule`.

The cross-validation half of the gate matters as much as the stability half: a small ensemble can
produce a spectrum that is real but overfitted, and the schedule keeps growing until the fit is
supported by the data.

The first-order structure update is deliberately withheld early. It is a Newton step on the fitted
second-order Hessian; when that Hessian is indefinite — imaginary modes — the step points *uphill*
along those branches. So it is requested only from `geom_2` on, and only when the previous
geometry's minimum frequency exceeds `--firstorder_min_freq` (default $-1.0$ THz). Until then the
geometries refine the force constants by sampling alone. Position and lattice steps always fit with
the first-order update.

`--mode preconditioning` runs this stage alone, in a `preconditioning/` directory directly under the
launch directory, and stops.

## Sample reuse

A step that barely moved the structure produced samples that are still valid for the next one.
After each step WARP compares the step's input cell with its updated cell using `--metric` and
`--reuse_threshold`; if the change is below the threshold, the previous step's fitted sample set
(its `infile.positions`, `infile.forces` and `infile.stat`, which already include any samples that
step had itself merged) is appended to the current fit, so the effective ensemble grows without
extra force evaluations. Over several consecutive small updates the ensemble therefore keeps
growing; `infile.meta` in the step directory gives the true count. The metrics
`frac_rmsd`, `max_comp_frac` and `pair_spectrum` are dimensionless; `cart_rmsd` and `max_atom` are
in Å.

One exception is hard-coded: the first position step after a lattice update always samples fresh,
because the force constants of the old cell no longer describe the new one.

## When the run stops

Three conditions must hold together at a lattice step, and they are evaluated in this order:

1. **the stress gate** — every compared component of
   $\lvert\sigma_{\mathrm{int}}-\sigma_{\mathrm{ext}}\rvert$ below `--press_conv_th`. All six
   components are compared in `scalar` and `full6` mode, the three diagonal ones in `diag3` mode.
   With `--pct_noise_floor K` greater than zero, component $i$ is instead judged against
   $\max(\texttt{pct},\,K\cdot\mathrm{SE}_i)$, so the gate can never ask for more resolution than
   the ensemble has.
2. **the phonon-stress gate** — every component of the change in $\sigma_{\mathrm{ph}}$ since the
   previous lattice step below the same per-component threshold, noise floor included. It is only
   treated as a gate once the stress gate holds, and it is reported as `N/A` at the first lattice
   step, which has nothing to compare with.
3. **the volume guard** — the relative volume change of the *previous* lattice step below 1 %. It
   is only evaluated once the first two hold.

The last two exist because a single step can cross the target by accident. Requiring that the
phonon stress has stopped changing and that the previous step was already small turns a crossing
into a genuine stationary point.

When all three pass, the current `infile.ucposcar` — the structure the stress was evaluated on — is
copied to `final_structure.vasp` in that `lattice_` directory and the run returns. The structures
of all steps are collected afterwards with `python3 -m warpt.get_frames -r <run root>`.

## What the code does not do

- It does not compute forces itself. Everything above is driven by the calculator selected with
  `--calculator`: the `--pwx` command for Quantum ESPRESSO, or the MLIP or VASP interface.
- It does not verify the free energy directly; it drives the first derivatives of the free energy,
  the mean force and the stress, to zero.
- It does not enforce a target space group. It preserves, and may raise, whatever spglib finds in
  the input structure.
