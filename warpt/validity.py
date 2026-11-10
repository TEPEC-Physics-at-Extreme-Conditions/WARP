# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Stage-completion predicates for WARP's idempotent resume.

Each ``valid_*`` function answers whether one pipeline stage finished in the
current working directory by checking its output files (and, where cheap, their
internal consistency). :func:`first_missing_stage` walks the ordered
:data:`STAGES` and returns the first stage that is not complete.

Reads only files in the current working directory (``infile.*``,
``contcar_conf*``, ``<master>.samplesN.out``, ``outfile.*``); writes nothing.
The HDF5 checks require ``h5py``; without it the phonon stages always count as
missing.

Public functions: ``valid_setup``, ``valid_sampling``, ``valid_scf``,
``valid_collect``, ``valid_extract``, ``valid_ifc_pressure``, ``valid_phonons``,
``valid_gruneisen``, ``valid_lattice``, ``first_missing_stage``.
"""
import os
import re

from .runner import job_done

try:
    import h5py
    HAVE_H5PY = True
except Exception:
    HAVE_H5PY = False


STAGES = ("setup", "sampling", "scf", "collect", "extract", "ifc_pressure", "phonons", "lattice")


def _nonempty(path: str) -> bool:
    """True if ``path`` exists and has non-zero size."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def valid_setup() -> bool:
    """True if the TDEP unit-cell and supercell inputs are present."""
    return _nonempty("infile.ucposcar") and _nonempty("infile.ssposcar")


def valid_sampling(expected_ns: int | None = None) -> bool:
    """True if ``contcar_conf*`` configs exist (and reach ``expected_ns`` if given)."""
    confs = sorted([f for f in os.listdir(".") if f.startswith("contcar_conf")])
    if not confs:
        return False
    if expected_ns is not None:
        return len(confs) >= int(expected_ns)
    return True


def valid_scf(master_scf_basename: str) -> bool:
    """True if at least one ``<master>.samplesN`` input exists and every one has a
    ``.out`` file containing ``JOB DONE.``.
    """
    ins = [f for f in os.listdir(".") if re.match(rf"^{re.escape(master_scf_basename)}\.samples\d+$", f)]
    if not ins:
        return False
    for inp in ins:
        outp = inp + ".out"
        if not job_done(outp):
            return False
    return True


def _parse_meta(path="infile.meta"):
    """Parse ``(natoms, nsamples)`` from ``infile.meta``; ``None`` on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        nat = int(lines[0].split()[0])
        ns = int(lines[1].split()[0])
        return nat, ns
    except Exception:
        return None


def valid_collect() -> bool:
    """True if the collected TDEP inputs exist and line counts match the meta."""
    if not (_nonempty("infile.meta") and _nonempty("infile.stat") and _nonempty("infile.positions") and _nonempty("infile.forces")):
        return False
    meta = _parse_meta("infile.meta")
    if meta is None:
        return False
    nat, ns = meta
    try:
        n_stat = sum(1 for _ in open("infile.stat"))
        n_pos  = sum(1 for _ in open("infile.positions"))
        n_for  = sum(1 for _ in open("infile.forces"))
    except Exception:
        return False
    return (n_stat == ns) and (n_pos == nat * ns) and (n_for == nat * ns)


def valid_extract() -> bool:
    """True if ``extract_forceconstants`` produced ``outfile.forceconstant``."""
    return _nonempty("outfile.forceconstant")


def valid_ifc_pressure() -> bool:
    """True if ``ifc_pressure`` wrote a parseable 6-component ``outfile.stress_res``."""
    if not _nonempty("outfile.stress_res"):
        return False
    try:
        with open("outfile.stress_res") as f:
            for ln in f:
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    parts = ln.split()
                    if len(parts) == 6:
                        [float(x) for x in parts]
                        return True
        return False
    except Exception:
        return False


def _h5_has(path: str, keys: tuple[str, ...]) -> bool:
    """True if the HDF5 file at ``path`` exists and contains all ``keys``."""
    if not HAVE_H5PY:
        return False
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return all(k in f for k in keys)
    except Exception:
        return False


def valid_phonons() -> bool:
    """True if a phonon dispersion HDF5 with ``frequencies`` exists (DOS gate)."""
    return _h5_has("outfile.dispersion_relations.hdf5", ("frequencies",))


def valid_gruneisen() -> bool:
    """True if the grid-dispersion HDF5 has frequencies and Grüneisen data."""
    if not HAVE_H5PY or not os.path.exists("outfile.grid_dispersions.hdf5"):
        return False
    try:
        with h5py.File("outfile.grid_dispersions.hdf5", "r") as f:
            if "frequencies" not in f:
                return False
            if ("gruneisen_tensors" in f) or ("gruneisen_parameters" in f):
                return True
        return False
    except Exception:
        return False


def valid_lattice() -> bool:
    """True if the lattice update wrote ``outfile.new_vc_ucposcar``."""
    return _nonempty("outfile.new_vc_ucposcar")


def first_missing_stage(ctx, want_dos=False, want_gruneisen=False, want_lattice=False,
                        want_ifc=True, expected_ns=None):
    """Return the name of the first incomplete stage in the current directory.

    Walks the pipeline in order (setup -> sampling -> scf -> collect -> extract
    -> ifc_pressure -> phonons -> lattice) and returns the first stage whose
    validity check fails, or ``None`` if everything required is present. The
    ``want_*`` flags select which optional stages this particular step requires:
    :func:`~warpt.workflow_vc_relax.run_vc_relax` checks a position step with
    ``want_ifc=False`` and ``want_lattice=False``, whereas
    :func:`~warpt.restart_check.find_resume_plan` leaves ``want_ifc`` at its default
    and therefore reports a finished position step at stage ``ifc_pressure``.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Run context; only ``ctx.master_scf`` is used here.
    want_dos, want_gruneisen, want_lattice : bool
        Require the DOS-phonon, Grüneisen-grid, and lattice-update stages.
    want_ifc : bool
        Require ``ifc_pressure`` output (true for preconditioning/lattice steps,
        false for pure position steps).
    expected_ns : int or None
        Expected number of sampled configurations, if known.

    Returns
    -------
    str or None
        The first missing stage name, or ``None`` if complete.
    """
    master_base = os.path.basename(ctx.master_scf)

    if not valid_setup():
        return "setup"
    if not valid_sampling(expected_ns):
        return "sampling"
    if not valid_scf(master_base):
        return "scf"
    if not valid_collect():
        return "collect"
    if not valid_extract():
        return "extract"
    # Position steps do not run ifc_pressure (no outfile.stress_res); only
    # preconditioning geometries and the lattice step do. The want_ifc gate lets a
    # completed position step be recognised as complete.
    if want_ifc and (not valid_ifc_pressure()):
        return "ifc_pressure"
    if want_dos and (not valid_phonons()):
        return "phonons"
    if want_gruneisen and (not valid_gruneisen()):
        return "phonons"
    if want_lattice and (not valid_lattice()):
        return "lattice"
    return None