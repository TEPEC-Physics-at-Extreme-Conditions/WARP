#!/usr/bin/env python3
# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Collect a WARP run's structures, in run order, into one trajectory.

Reads, for each step directory below the run root, the first ``outfile.new*ucposcar`` found
below it (recursive glob, natural name order; a ``lattice_`` directory holding both
``outfile.new_ucposcar`` and ``outfile.new_vc_ucposcar`` yields ``outfile.new_ucposcar``).
Step order is execution order: ``iter_0/preconditioning/geom_*`` first, then for every
directory matching ``iter_*`` (any suffix, so sidelined copies such as ``iter_6.stale`` are
included) its ``position_*`` directories followed by ``lattice_*``, each group sorted by the
number in its name (``geom_2`` before ``geom_10``). A step without an
``outfile.new*ucposcar`` (not yet past force-constant extraction) contributes no frame.
:mod:`warpt.trajectory` reads only directories named exactly ``iter_<N>`` but lists a step
as soon as its ``infile.ucposcar`` exists, so its step count can differ from the frame count
here.

Writes ``<root>/trajectory.xyz`` (or ``-o FILE``) with the step of every frame as ``step=``
in its comment line and, unless ``--no-copy``, a copy of every frame named after its step,
``<root>/frame_NNNN_<step>.vasp`` (for example ``frame_0007_iter_2_position_1.vasp``), whose
title line is the step. Every frame is also checked for symmetry with spglib at each
``--symprec`` (Å; default 1e-5, 1e-4, 1e-3 and 1e-2): the space-group number at each tolerance is
printed in a table, one line per frame, and stored in ``trajectory.xyz`` as ``spg_<symprec>``.
A group that appears only at the looser tolerances is a symmetry the structure is approaching but
does not yet have. Requires ASE; without spglib the symmetry columns are left out.

Command line::

    python3 -m warpt.get_frames [-r ROOT] [-o FILE] [--no-copy] [--symprec S ...] [-q]

``-r`` is the run root, the directory WARP was launched in and that holds ``iter_*``
(default: current directory); ``-q`` suppresses the table and the summary. Exit
status is 1 when no frame was found.

Public functions: ``natural_key``, ``first_ucposcar``, ``stage_dirs``, ``step_label``,
``fmt_symprec``, ``spacegroups``, ``collect``, ``main``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ase.io import read, write


def natural_key(p):
    """Sort key that orders geom_2 before geom_10, not after it."""
    s = p.name if isinstance(p, Path) else str(p)
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


def first_ucposcar(d: Path):
    """First ``outfile.new*ucposcar`` below ``d`` (recursive glob, natural name order), or None.

    A ``lattice_`` directory holding ``outfile.new_ucposcar`` and ``outfile.new_vc_ucposcar``
    yields ``outfile.new_ucposcar``. Only file names are compared, so equally named files in
    different subdirectories keep the ``rglob`` order.
    """
    files = sorted(d.rglob("outfile.new*ucposcar"), key=natural_key)
    return files[0] if files else None


def stage_dirs(root: Path):
    """Step directories below the run root ``root`` in execution order.

    ``root/iter_0/preconditioning/geom_*`` first, then for every ``root/iter_*`` directory
    (any suffix, natural order) its ``position_*`` directories followed by ``lattice_*``.
    """
    out = []
    precond = root / "iter_0" / "preconditioning"
    if precond.is_dir():
        out += [d for d in sorted(precond.glob("geom_*"), key=natural_key) if d.is_dir()]
    for iterdir in sorted(root.glob("iter_*"), key=natural_key):
        if not iterdir.is_dir():
            continue
        for pat in ("position_*", "lattice_*"):     # position first: WARP's order
            out += [d for d in sorted(iterdir.glob(pat), key=natural_key) if d.is_dir()]
    return out


def step_label(d: Path, root: Path) -> str:
    """The step a directory belongs to, relative to the run root, e.g. ``iter_2/position_1``.

    A trailing ``_`` is dropped from every part, so ``iter_2/lattice_`` reads ``iter_2/lattice``.
    """
    return "/".join(part.rstrip("_") for part in d.relative_to(root).parts)


SYMPRECS = (1e-5, 1e-4, 1e-3, 1e-2)


def fmt_symprec(sp: float) -> str:
    """A tolerance as written in the table and the ``spg_`` keys: ``1e-4``, ``1.5e-2``."""
    mant, exp = f"{sp:e}".split("e")
    return f"{float(mant):g}e{int(exp)}"


def spacegroups(atoms, symprecs=SYMPRECS):
    """Space group of ``atoms`` at each symprec (Å), as ``(number, symbol)`` or None.

    Returns None for every entry without spglib, and None for one entry when spglib fails at
    that tolerance.
    """
    try:
        import spglib
    except ImportError:
        return [None] * len(symprecs)
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    out = []
    for sp in symprecs:
        try:
            d = spglib.get_symmetry_dataset(cell, symprec=sp)
        except Exception:  # noqa: BLE001
            d = None
        out.append(None if d is None else (int(d.number), str(d.international)))
    return out


def collect(root: Path, out_xyz: Path, copy_frames: bool = True, quiet: bool = False,
            symprecs=SYMPRECS):
    """Write ``out_xyz`` (via ASE) from the first structure of every step directory under ``root``.

    Every frame carries its step: as ``step=<label>`` in the comment line of ``out_xyz`` and,
    with ``copy_frames``, in the name and the first (comment) line of the copy written to
    ``root/frame_NNNN_<label with / replaced by _>.vasp``, for example
    ``frame_0007_iter_2_position_1.vasp``; the rest of the copy is the step's file unchanged.
    The space group at each of ``symprecs`` is stored as ``spg_<symprec>`` in the frame's
    comment line (see :func:`spacegroups`). Unless ``quiet``, prints one table line per frame
    (the file written, or the step with ``copy_frames`` off, and the space group at each
    symprec) and a summary. Returns the number of frames; nothing is written when it is zero.
    """
    images, n = [], 0
    if not quiet and symprecs:
        print(f"{'frame':46s}" + "".join(f"{'symprec ' + fmt_symprec(sp):>18s}" for sp in symprecs))
    for d in stage_dirs(root):
        f = first_ucposcar(d)
        if f is None:
            continue
        label = step_label(d, root)
        name = label
        if copy_frames:
            dst = root / f"frame_{n:04d}_{label.replace('/', '_')}.vasp"
            lines = f.read_text().splitlines(keepends=True)
            dst.write_text("".join([f"{label}\n"] + lines[1:]))
            name = dst.name
        img = read(f, format="vasp")
        img.info["step"] = label
        spgs = spacegroups(img, symprecs)
        for sp, g in zip(symprecs, spgs):
            if g is not None:
                img.info[f"spg_{fmt_symprec(sp)}"] = g[0]
        if not quiet:
            print(f"{name:46s}" + "".join(
                f"{'-' if g is None else f'{g[0]} {g[1]}':>18s}" for g in spgs))
        images.append(img)
        n += 1
    if n:
        write(out_xyz, images)
        if not quiet:
            print(f"Wrote {out_xyz} with {n} frames")
    elif not quiet:
        print("No frames found; nothing was written.")
    return n


def main():
    """Command-line entry point: parse the options, run collect and exit 0 if frames were
    found, 1 otherwise (also exits 1 with a message when ``root`` is not a directory)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-r", "--root", default=".",
                   help="run root: the directory WARP was launched in, holding iter_* "
                        "(default: current directory)")
    p.add_argument("-o", "--output", default=None,
                   help="trajectory file to write (default: <root>/trajectory.xyz)")
    p.add_argument("--no-copy", action="store_true",
                   help="do not write the per-frame frame_NNNN_<step>.vasp files")
    p.add_argument("--symprec", type=float, nargs="+", default=list(SYMPRECS),
                   help="tolerances (Å) at which the space group of every frame is reported "
                        "(default: 1e-5 1e-4 1e-3 1e-2)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="print neither the per-frame table nor the summary")
    a = p.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    out = Path(a.output) if a.output else root / "trajectory.xyz"
    n = collect(root, out, copy_frames=not a.no_copy, quiet=a.quiet, symprecs=a.symprec)
    sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
