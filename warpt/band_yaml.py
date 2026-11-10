#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""
Phonopy ``band.yaml`` from the phonon dispersion of a WARP step.

Converts TDEP's ``outfile.dispersion_relations.hdf5`` and the unit cell ``infile.ucposcar`` of
one step directory into the ``band.yaml`` format that phonopy writes for a band structure, so
that viewers and scripts that read phonopy output (for example CrysViz, which animates the
modes) can show WARP phonons: the q-path with its labels, the frequencies in THz and the
complex eigenvectors of every mode.

Conventions carried over from the TDEP file: frequencies in THz, imaginary frequencies stored
as negative numbers; ``distance`` is TDEP's cumulative path length in Å⁻¹ without the factor
2π; ``q-position`` is in reduced coordinates of the reciprocal lattice of ``infile.ucposcar``
(TDEP stores the Cartesian q in Å⁻¹ without 2π, so the reduced coordinates are the cell
vectors dotted into it); the eigenvectors are the orthonormal eigenvectors of the dynamical
matrix, one complex number per Cartesian component, as in phonopy. The path is split into
TDEP's segments, each of equal length with the joints repeated, and the label ``G`` is written
as ``$\\Gamma$``.

Command line::

    python3 -m warpt.band_yaml [step_dir] [-o FILE]

``step_dir`` defaults to the current directory; the output defaults to ``band.yaml`` inside it.

Public functions: ``read_dispersion``, ``write_band_yaml``, ``main``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DISPERSION = "outfile.dispersion_relations.hdf5"
UNITCELL = "infile.ucposcar"


def _label(name):
    """TDEP path label in the form phonopy band structures use (``G`` becomes Gamma)."""
    return "$\\Gamma$" if name in ("G", "GM", "Gamma") else name


def read_dispersion(step_dir):
    """Read the dispersion of one step directory.

    Parameters
    ----------
    step_dir : str or Path
        Directory holding ``outfile.dispersion_relations.hdf5`` and ``infile.ucposcar``.

    Returns
    -------
    dict
        ``atoms`` (ase.Atoms of the unit cell), ``q_red`` (nq, 3) reduced q-points,
        ``distance`` (nq,) in Å⁻¹, ``freq`` (nq, nmodes) in THz, ``eigvec`` (nq, nmodes,
        nmodes) complex, ``segments`` (list of point counts) and ``labels`` (list of
        ``[start, end]`` pairs, one per segment).

    Raises
    ------
    FileNotFoundError
        When either file is missing.
    ValueError
        When the number of points cannot be split evenly into the labelled segments.
    """
    import h5py
    from ase.io import read

    d = Path(step_dir)
    for name in (DISPERSION, UNITCELL):
        if not (d / name).is_file():
            raise FileNotFoundError(f"{d / name} not found")
    atoms = read(d / UNITCELL, format="vasp")
    with h5py.File(d / DISPERSION, "r") as f:
        q_cart = np.array(f["q_vector"][()], dtype=float)
        distance = np.array(f["q_values"][()], dtype=float)
        freq = np.array(f["frequencies"][()], dtype=float)
        eigvec = np.array(f["eigenvectors_re"][()]) + 1j * np.array(f["eigenvectors_im"][()])
        raw = f.attrs.get("q_tick_labels", b"")
    names = (raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)).split()
    nseg = max(len(names) - 1, 1)
    if len(distance) % nseg:
        raise ValueError(f"{len(distance)} q-points cannot be split into {nseg} segments")
    per = len(distance) // nseg
    q_red = q_cart @ np.array(atoms.cell).T
    labels = ([[_label(names[i]), _label(names[i + 1])] for i in range(nseg)]
              if len(names) > 1 else [])
    return dict(atoms=atoms, q_red=q_red, distance=distance, freq=freq, eigvec=eigvec,
                segments=[per] * nseg, labels=labels)


def write_band_yaml(step_dir, out=None):
    """Write the ``band.yaml`` of one step directory.

    Parameters
    ----------
    step_dir : str or Path
        Directory holding ``outfile.dispersion_relations.hdf5`` and ``infile.ucposcar``.
    out : str or Path or None
        File to write; default ``band.yaml`` in ``step_dir``.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    data = read_dispersion(step_dir)
    atoms, q, dist, freq, ev = (data[k] for k in ("atoms", "q_red", "distance", "freq", "eigvec"))
    lat = np.array(atoms.cell)
    rec = np.linalg.inv(lat).T
    nat = len(atoms)
    out = Path(out) if out else Path(step_dir) / "band.yaml"
    ends = set()
    pos = 0
    for n in data["segments"]:
        ends.update((pos, pos + n - 1))
        pos += n
    seg_label = {}
    pos = 0
    for n, (lo, hi) in zip(data["segments"], data["labels"]):
        seg_label[pos], seg_label[pos + n - 1] = lo, hi
        pos += n

    L = [f"nqpoint: {len(dist)}", f"npath: {len(data['segments'])}", "segment_nqpoint:"]
    L += [f"- {n}" for n in data["segments"]]
    if data["labels"]:
        L.append("labels:")
        L += [f"- [ '{lo}', '{hi}' ]" for lo, hi in data["labels"]]
    L.append("reciprocal_lattice:")
    L += [f"- [ {v[0]:14.8f}, {v[1]:14.8f}, {v[2]:14.8f} ] # {ax}*" for v, ax in zip(rec, "abc")]
    L.append(f"natom: {nat}")
    L.append("lattice:")
    L += [f"- [ {v[0]:21.15f}, {v[1]:21.15f}, {v[2]:21.15f} ] # {ax}" for v, ax in zip(lat, "abc")]
    L.append("points:")
    for i, (sym, p, m) in enumerate(zip(atoms.get_chemical_symbols(),
                                        atoms.get_scaled_positions(), atoms.get_masses()), 1):
        L.append(f"- symbol: {sym} # {i}")
        L.append(f"  coordinates: [ {p[0]:18.15f}, {p[1]:18.15f}, {p[2]:18.15f} ]")
        L.append(f"  mass: {m:.6f}")
    L.append("")
    L.append("phonon:")
    for iq in range(len(dist)):
        L.append(f"- q-position: [ {q[iq, 0]:12.7f}, {q[iq, 1]:12.7f}, {q[iq, 2]:12.7f} ]")
        L.append(f"  distance: {dist[iq]:12.7f}")
        if iq in seg_label:
            L.append(f"  label: '{seg_label[iq]}'")
        L.append("  band:")
        for im in range(freq.shape[1]):
            L.append(f"  - # {im + 1}")
            L.append(f"    frequency: {freq[iq, im]:15.10f}")
            L.append("    eigenvector:")
            vec = ev[iq, im].reshape(nat, 3)
            for ia in range(nat):
                L.append(f"    - # atom {ia + 1}")
                for c in vec[ia]:
                    L.append(f"      - [ {c.real:17.14f}, {c.imag:17.14f} ]")
        L.append("")
    out.write_text("\n".join(L) + "\n")
    return out


def main():
    """Command-line entry point: write ``band.yaml`` for ``step_dir``; exit 1 with a message
    when an input file is missing or cannot be read."""
    p = argparse.ArgumentParser(description="Write a phonopy band.yaml from the phonon "
                                            "dispersion of a WARP step directory.")
    p.add_argument("step_dir", nargs="?", default=".",
                   help="step directory with outfile.dispersion_relations.hdf5 and "
                        "infile.ucposcar (default: current directory)")
    p.add_argument("-o", "--output", default=None,
                   help="file to write (default: band.yaml in step_dir)")
    a = p.parse_args()
    try:
        out = write_band_yaml(a.step_dir, a.output)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    print(f"[band_yaml] wrote {out}")


if __name__ == "__main__":
    main()
