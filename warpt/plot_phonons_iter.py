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
Phonon dispersions of every step of a WARP run, in execution order.

Walks the step directories of a run in the order WARP visits them (the same order as
:mod:`warpt.trajectory`) and, for every step that has an ``outfile.dispersion_relations.hdf5``,
draws the dispersion along TDEP's q-path with the high-symmetry labels and, when
``outfile.phonon_dos.hdf5`` is present, the phonon density of states beside it. Steps that kept
only TDEP's text output are read from ``outfile.dispersion_relations`` (labels from
``outfile.dispersion_relations.gnuplot``) and ``outfile.phonon_dos`` instead. All figures
share one frequency axis, so steps can be compared at a glance. A step whose lowest frequency
lies below ``-imag_tol`` has imaginary modes; its title is drawn in red and it is flagged in
the summary. Every title and the summary line give the lowest and the highest frequency.

Writes into the output directory (``-o``, default ``phonon_plots`` in the current directory):
one PNG per step, named after the step in execution order (``000_iter0_geom0.png``,
``007_iter2_position1.png``, ``012_iter2_lattice.png``, ...); ``overview.png``, a grid with the
dispersion of every step; and, with ``--band-yaml``, a phonopy ``band.yaml`` per step with the
same base name (:mod:`warpt.band_yaml`). Prints one summary line per step. Requires numpy,
h5py and matplotlib.

Command line::

    python3 -m warpt.plot_phonons_iter [run_dir] [-o OUTDIR] [--overview-only] [--band-yaml]
        [--imag-tol THZ] [--show]

``run_dir`` is the run root (the directory WARP was launched in, holding ``iter_*``) and
defaults to the current directory.

Public functions: ``step_name``, ``read_step``, ``read_step_text``, ``plot_step``,
``plot_overview``, ``main``.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

from .trajectory import step_dirs

DISPERSION = "outfile.dispersion_relations.hdf5"
DOS = "outfile.phonon_dos.hdf5"
DISPERSION_TXT = "outfile.dispersion_relations"
DOS_TXT = "outfile.phonon_dos"
COLOR = "#3B6EA8"
RED = "#C0392B"


def step_name(order, it, kind, idx):
    """File base name of a step: execution index, iteration and step, e.g. ``007_iter2_position1``.

    Parameters
    ----------
    order : int
        Position of the step in execution order.
    it, kind, idx
        Iteration number, step kind (``precond``, ``position`` or ``lattice``) and step index,
        as returned by :func:`warpt.trajectory.step_dirs`.
    """
    if kind == "precond":
        return f"{order:03d}_iter{it}_geom{idx}"
    if kind == "lattice":
        return f"{order:03d}_iter{it}_lattice"
    return f"{order:03d}_iter{it}_{kind}{idx}"


def _gamma(labels):
    """TDEP's Gamma labels (``G``, ``GM``, ``Gamma``) written as Γ."""
    return ["Γ" if s in ("G", "GM", "Gamma") else s for s in labels]


def read_step_text(path):
    """Dispersion and DOS of one step from TDEP's text files, or None without
    ``outfile.dispersion_relations``.

    The first column of ``outfile.dispersion_relations`` is the path coordinate and every further
    column one branch; the high-symmetry points are the ``xtics`` of
    ``outfile.dispersion_relations.gnuplot`` (none when that file is missing). The first two
    columns of ``outfile.phonon_dos`` are the frequency and the total DOS. Returns the same dict
    as :func:`read_step`.
    """
    p = Path(path)
    if not (p / DISPERSION_TXT).is_file():
        return None
    data = np.loadtxt(p / DISPERSION_TXT, ndmin=2)
    ticks = labels = None
    gp = p / (DISPERSION_TXT + ".gnuplot")
    if gp.is_file():
        found = re.findall(r'"([^"]*)"\s+([-+0-9.Ee]+)', gp.read_text(errors="replace"))
        if found:
            labels = _gamma([name for name, _ in found])
            ticks = np.array([float(x) for _, x in found])
    dos_f = dos = None
    if (p / DOS_TXT).is_file():
        d = np.loadtxt(p / DOS_TXT, ndmin=2)
        dos_f, dos = d[:, 0], d[:, 1]
    return dict(q=data[:, 0], freq=data[:, 1:], ticks=ticks, labels=labels, dos_f=dos_f, dos=dos)


def read_step(path):
    """Dispersion and, when present, DOS of one step directory, or None without a dispersion.

    Reads TDEP's HDF5 files and falls back to its text files (:func:`read_step_text`) when the
    step has no ``outfile.dispersion_relations.hdf5``.

    Returns
    -------
    dict or None
        ``q`` (nq,) path coordinate in Å⁻¹, ``freq`` (nq, nmodes) in THz, ``ticks`` and
        ``labels`` of the high-symmetry points, and ``dos_f`` / ``dos`` (None when the step has
        no ``outfile.phonon_dos.hdf5``). DOS frequencies given as bin edges are converted to
        bin centres.
    """
    import h5py

    p = Path(path)
    if not (p / DISPERSION).is_file():
        return read_step_text(p)
    with h5py.File(p / DISPERSION, "r") as f:
        q = np.array(f["q_values"][()], dtype=float).ravel()
        freq = np.array(f["frequencies"][()], dtype=float)
        ticks = np.array(f["q_ticks"][()], dtype=float).ravel() if "q_ticks" in f else None
        raw = f.attrs.get("q_tick_labels")
    labels = None
    if raw is not None:
        labels = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, np.bytes_))
                  else str(raw)).split()
        labels = _gamma(labels)
    dos_f = dos = None
    if (p / DOS).is_file():
        with h5py.File(p / DOS, "r") as f:
            dos_f = np.array(f["frequencies"][()], dtype=float).ravel()
            dos = np.array(f["dos"][()], dtype=float).ravel()
        if dos_f.size == dos.size + 1:
            dos_f = 0.5 * (dos_f[:-1] + dos_f[1:])
        order = np.argsort(dos_f)
        dos_f, dos = dos_f[order], dos[order]
    return dict(q=q, freq=freq if freq.ndim == 2 else freq[:, None], ticks=ticks,
                labels=labels, dos_f=dos_f, dos=dos)


def _draw_dispersion(ax, d, color):
    """Bands, high-symmetry lines and labels, and the zero line, on ``ax``."""
    for j in range(d["freq"].shape[1]):
        ax.plot(d["q"], d["freq"][:, j], color=color, lw=1.0)
    if d["ticks"] is not None:
        for t in d["ticks"]:
            ax.axvline(t, color="0.8", lw=0.7, zorder=0)
        ax.set_xticks(d["ticks"])
        if d["labels"] is not None and len(d["labels"]) == len(d["ticks"]):
            ax.set_xticklabels(d["labels"])
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
    lo, hi = d["q"].min(), d["q"].max()
    if d["ticks"] is not None and len(d["ticks"]):
        lo, hi = min(lo, d["ticks"].min()), max(hi, d["ticks"].max())
    ax.set_xlim(lo, hi)


def plot_step(d, title, out, ylim, imaginary):
    """Save the dispersion (left) and the DOS (right, when available) of one step to ``out``.

    Parameters
    ----------
    d : dict
        Output of :func:`read_step`.
    title : str
        Figure title (the step and its lowest and highest frequency).
    out : str or Path
        PNG to write (150 dpi).
    ylim : tuple of float
        Frequency limits shared by all steps, THz.
    imaginary : bool
        Draw the title in red.
    """
    import matplotlib.pyplot as plt

    if d["dos"] is not None:
        fig = plt.figure(figsize=(6.5, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.2], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        axd = fig.add_subplot(gs[0, 1], sharey=ax)
        axd.plot(d["dos"], d["dos_f"], color=COLOR, lw=1.0)
        axd.fill_betweenx(d["dos_f"], 0.0, d["dos"], color=COLOR, alpha=0.25, lw=0)
        axd.axhline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
        axd.set_xlabel("DOS (arb.)")
        axd.set_xlim(left=0.0)
        axd.tick_params(axis="y", labelleft=False, length=0)
    else:
        fig, ax = plt.subplots(figsize=(5.2, 5))
    _draw_dispersion(ax, d, COLOR)
    ax.set_ylim(*ylim)
    ax.set_ylabel("frequency (THz)")
    ax.set_title(title, color=RED if imaginary else "black", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(steps, out, ylim, imag_tol):
    """Save a grid with the dispersion of every step to ``out``.

    Parameters
    ----------
    steps : list of tuple
        ``(label, data)`` per step in execution order, ``data`` from :func:`read_step`.
    out : str or Path
        PNG to write (150 dpi).
    ylim : tuple of float
        Shared frequency limits, THz.
    imag_tol : float
        A panel whose lowest frequency is below ``-imag_tol`` THz gets a red title.
    """
    import matplotlib.pyplot as plt

    n = len(steps)
    ncol = min(6, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.3 * nrow), sharey=True,
                             squeeze=False)
    for ax, (label, d) in zip(axes.ravel(), steps):
        imag = d["freq"].min() < -imag_tol
        _draw_dispersion(ax, d, RED if imag else COLOR)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{label}\nmin {d['freq'].min():.2f}  max {d['freq'].max():.2f} THz",
                     fontsize=7, color=RED if imag else "black")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("THz", fontsize=7)
    h = fig.get_figheight()
    fig.suptitle("Phonon dispersion of every step (red: imaginary modes)", fontsize=10,
                 y=1 - 0.08 / h)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.22 / h))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    """Command-line entry point: plot every step of ``run_dir`` and print a summary; exits 1
    with a message when no step has a dispersion."""
    p = argparse.ArgumentParser(description="Plot the phonon dispersion of every step of a WARP "
                                            "run, in execution order.")
    p.add_argument("run_dir", nargs="?", default=".",
                   help="run root: the directory WARP was launched in, holding iter_* "
                        "(default: current directory)")
    p.add_argument("-o", "--outdir", default="phonon_plots",
                   help="directory for the figures (default: phonon_plots)")
    p.add_argument("--overview-only", action="store_true",
                   help="write only overview.png, not one figure per step")
    p.add_argument("--band-yaml", action="store_true",
                   help="also write a phonopy band.yaml per step (warpt.band_yaml)")
    p.add_argument("--imag-tol", type=float, default=0.05,
                   help="a lowest frequency below -IMAG_TOL THz counts as imaginary "
                        "(default 0.05)")
    p.add_argument("--show", action="store_true",
                   help="open overview.png in a viewer after writing it")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    steps = []
    for order, (it, kind, idx, path) in enumerate(step_dirs(a.run_dir)):
        d = read_step(path)
        if d is None:
            continue
        rel = os.path.relpath(path, a.run_dir)
        steps.append((step_name(order, it, kind, idx), rel, path, d))
    if not steps:
        sys.exit(f"no step with {DISPERSION} or {DISPERSION_TXT} found under {a.run_dir}")

    lo = min(min(d["freq"].min(), d["dos_f"].min() if d["dos_f"] is not None else np.inf)
             for *_, d in steps)
    hi = max(max(d["freq"].max(), d["dos_f"].max() if d["dos_f"] is not None else -np.inf)
             for *_, d in steps)
    pad = 0.02 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{'step':34s} {'min freq (THz)':>15s} {'max freq (THz)':>15s}")
    for name, rel, path, d in steps:
        fmin, fmax = float(d["freq"].min()), float(d["freq"].max())
        imag = fmin < -a.imag_tol
        print(f"{rel:34s} {fmin:15.3f} {fmax:15.3f}{'   imaginary modes' if imag else ''}")
        if not a.overview_only:
            plot_step(d, f"{rel}   (min {fmin:.2f}, max {fmax:.2f} THz)", outdir / f"{name}.png",
                      ylim, imag)
        if a.band_yaml:
            from .band_yaml import write_band_yaml
            write_band_yaml(path, outdir / f"{name}.band.yaml")
    plot_overview([(rel, d) for _, rel, _, d in steps], outdir / "overview.png", ylim,
                  a.imag_tol)
    n_imag = sum(d["freq"].min() < -a.imag_tol for *_, d in steps)
    print(f"[plot_phonons_iter] {len(steps)} steps, {n_imag} with imaginary modes -> {outdir}/")
    if a.show:
        from .trajectory import show_file
        show_file(str(outdir / "overview.png"))


if __name__ == "__main__":
    main()
