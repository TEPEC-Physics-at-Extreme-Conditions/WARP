# Citing WARP

WARP builds directly on the temperature-dependent effective potential method and on its reference
implementation, so a paper that uses WARP should cite them. This page gives the references and the
contacts. In short: **cite WARP: <https://arxiv.org/abs/2611.12736>**, plus TDEP.

## The WARP paper

Cite WARP as

> H. R. A. ten Eikelder, F. Knoop and F. Trybel,
> *Finite temperature and pressure structural relaxations including zero-point motion and anharmonicity*,
> arXiv:2611.12736 (2026), <https://arxiv.org/abs/2611.12736>.

```bibtex
@article{teneikelder2026warp,
  author        = {ten Eikelder, H. R. A. and Knoop, F. and Trybel, F.},
  title         = {Finite temperature and pressure structural relaxations including
                   zero-point motion and anharmonicity},
  journal       = {arXiv preprint},
  eprint        = {2611.12736},
  archivePrefix = {arXiv},
  year          = {2026},
  url           = {https://arxiv.org/abs/2611.12736}
}
```

together with the TDEP references below, and name the code and its version in the methods
section.

## TDEP

WARP drives the TDEP binaries for the sampling, the force-constant fitting, the phonon dispersion
and the stress decomposition. Cite:

> O. Hellman, I. A. Abrikosov and S. I. Simak,
> *Lattice dynamics of anharmonic solids from first principles*,
> Phys. Rev. B **84**, 180301 (2011).

> O. Hellman, P. Steneteg, I. A. Abrikosov and S. I. Simak,
> *Temperature dependent effective potential method for accurate free energy calculations of solids*,
> Phys. Rev. B **87**, 104111 (2013).

> F. Knoop *et al.*,
> *TDEP: Temperature Dependent Effective Potentials*,
> J. Open Source Softw. **9**, 6150 (2024).
{% raw %}
```bibtex
@article{hellman2011,
  author  = {Hellman, O. and Abrikosov, I. A. and Simak, S. I.},
  title   = {Lattice dynamics of anharmonic solids from first principles},
  journal = {Phys. Rev. B},
  volume  = {84},
  pages   = {180301},
  year    = {2011}
}

@article{hellman2013,
  author  = {Hellman, O. and Steneteg, P. and Abrikosov, I. A. and Simak, S. I.},
  title   = {Temperature dependent effective potential method for accurate free
             energy calculations of solids},
  journal = {Phys. Rev. B},
  volume  = {87},
  pages   = {104111},
  year    = {2013}
}

@article{knoop2024,
  author  = {Knoop, F. and others},
  title   = {{TDEP}: Temperature Dependent Effective Potentials},
  journal = {J. Open Source Softw.},
  volume  = {9},
  pages   = {6150},
  year    = {2024}
}
```
{% endraw %}
## CrysViz

[CrysViz](https://crysviz.org), a viewer for the structures and trajectories that WARP writes, is
described in a paper that is in preparation. Until it appears, cite the website.

## What else to cite

Cite the source of the forces as well: the Quantum ESPRESSO or VASP release, the functional and
the pseudopotentials or PAW datasets, or the machine-learned potential and the data it was trained
on. WARP does not collect this information into a summary of the run; take it from the inputs of
the run.

## Authors and contact

WARP is developed by

- H. R. A. ten Eikelder — <henricus.ten.eikelder@liu.se>
- F. Trybel — <florian.trybel@liu.se>

as an interface to Olle Hellman's TDEP. The same two names and addresses are printed in the banner
at the top of every run and above the `--help` text, together with the version of the package;
quote that version in a bug report. It is also available as `warpt.__version__`.

## Licence

WARP is free software, released under the [GNU General Public License, version 3 or later](https://www.gnu.org/licenses/gpl-3.0.html)
(GPL-3.0-or-later); see [`LICENSE`](https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP/blob/main/LICENSE). The copyright is held by the developers listed in
[`NOTICE`](https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP/blob/main/NOTICE), which also lists the contributors and the third-party parts of the repository: the
bundled TDEP copy in `tdep_warp/` keeps its MIT licence, and the packaged symmetry table keeps its
CC BY 4.0 licence.
