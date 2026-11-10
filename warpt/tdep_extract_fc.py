# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Force-constant extraction, Wyckoff projection and the structure-physicality gate.

Runs TDEP's ``extract_forceconstants`` (second order; third order and the
first-order structure update when requested), copies the results to the
``infile.*`` names the next stage reads, optionally projects the first-order
displacement onto the space-group Wyckoff degrees of freedom
(:mod:`warpt.Wyckoff_Projector`), and rejects a committed structure with atoms
closer than ``MIN_PHYSICAL_DIST_A``.

Reads, in the current working directory: the ``infile.*`` sampling files (through
``extract_forceconstants``), ``infile.ucposcar`` and ``outfile.uc_displacements``.
Writes: ``extract_forceconstants.log``, ``infile.forceconstant``,
``infile.forceconstant_thirdorder``, ``outfile.uc_projected_structure`` and
``outfile.new_ucposcar``.

Public functions: ``ensure_outfile_new_ucposcar``, ``min_interatomic_distance``,
``assert_structure_physical``, ``run_extract_fc``.
"""
import os
import shutil

import numpy as np

from .utilities import die, safe_run
from .io_poscar import read_poscar_frac
from .Wyckoff_Projector import generate_projected_structure

# Minimum interatomic distance (Å) below which assert_structure_physical aborts
# the run; stricter than the Wyckoff projector's 0.3 Å proximity guardrail.
MIN_PHYSICAL_DIST_A = 0.5


def ensure_outfile_new_ucposcar() -> None:
    """Create ``outfile.new_ucposcar`` from ``infile.ucposcar`` when TDEP did not write it.

    Copies ``infile.ucposcar`` to ``outfile.new_ucposcar`` and prints a warning if the
    former exists and the latter does not; otherwise does nothing. Returns None.
    """
    if (not os.path.exists("outfile.new_ucposcar")) and os.path.exists("infile.ucposcar"):
        shutil.copy("infile.ucposcar", "outfile.new_ucposcar")
        print("[warn] extract_forceconstants did not write outfile.new_ucposcar; using infile.ucposcar as fallback.", flush=True)


def min_interatomic_distance(poscar_path: str) -> float:
    """Return the smallest interatomic distance (Å) in a POSCAR.

    Parameters
    ----------
    poscar_path : str
        Path to a VASP5 POSCAR / ucposcar file (Direct or Cartesian).

    Returns
    -------
    float
        Minimum over all atom pairs of the distance after wrapping each fractional
        coordinate difference to [-0.5, 0.5] per axis; ``inf`` for fewer than two atoms.
    """
    frac, A, _ = read_poscar_frac(poscar_path)
    n = len(frac)
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = frac[j] - frac[i]
            d -= np.round(d)  # wrap each fractional component to [-0.5, 0.5]
            dmin = min(dmin, float(np.linalg.norm(d @ A)))
    return dmin


def assert_structure_physical(poscar_path: str, min_dist_angstrom: float = MIN_PHYSICAL_DIST_A) -> None:
    """Abort the run if a committed structure has atoms closer than ``min_dist_angstrom``.

    Applies to whatever structure is about to be propagated, whether written by
    ``extract_forceconstants --firstorder`` or by the Wyckoff projector. It runs after,
    and is stricter than, the projector's own proximity guardrail
    (``min_dist_angstrom`` = 0.3 Å in
    :func:`warpt.Wyckoff_Projector.generate_projected_structure`).

    Parameters
    ----------
    poscar_path : str
        Path to the structure to validate.
    min_dist_angstrom : float
        Rejection threshold in Å (default ``MIN_PHYSICAL_DIST_A``).

    Side effects
    ------------
    Prints the minimum distance when the check passes.

    Raises
    ------
    SystemExit
        Via :func:`die` when any pair is closer than ``min_dist_angstrom``.
    """
    d = min_interatomic_distance(poscar_path)
    if d < min_dist_angstrom:
        die(f"[sanity] {poscar_path} is unphysical: min interatomic distance "
            f"{d:.3f} A < {min_dist_angstrom:.2f} A -- aborting (soft-mode runaway "
            f"in the first-order structure update?).")
    print(f"[sanity] min interatomic distance = {d:.3f} A (>= {min_dist_angstrom:.2f}, ok)", flush=True)


def run_extract_fc(
    ctx,
    state,
    mode: str,
    *,
    do_wyckoff_project: bool = True,
    displacements_path: str = "outfile.uc_displacements",
    projected_out_path: str = "outfile.uc_projected_structure",
) -> None:
    """Run ``extract_forceconstants`` and commit the resulting structure.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Uses ``cutoffFC2`` (Å), ``cutoffFC3`` (Å or None), ``wz_supercell`` and
        ``wyckoff_damping``.
    state : context_state.WarpState
        Not read.
    mode : str
        ``"second"`` (FC2 only), ``"second_third"`` (FC2 and, if ``ctx.cutoffFC3`` is
        set, FC3) or ``"second_third_first"`` (additionally ``--firstorder``, the
        structure update).
    do_wyckoff_project : bool
        Project the first-order displacement onto the Wyckoff degrees of freedom (only
        in mode ``"second_third_first"`` and when ``displacements_path`` exists).
    displacements_path : str
        TDEP's first-order displacement file.
    projected_out_path : str
        Where the projected structure is written before it is copied to
        ``outfile.new_ucposcar``.

    Side effects
    ------------
    Writes ``extract_forceconstants.log``, ``infile.forceconstant``,
    ``infile.forceconstant_thirdorder`` (when produced), ``outfile.new_ucposcar`` (from
    TDEP, from the projection, or copied from ``infile.ucposcar`` when TDEP wrote none)
    and ``projected_out_path``. Returns None.

    Raises
    ------
    SystemExit
        Via :func:`die` for an unknown mode, a failed extraction with no
        ``outfile.forceconstant``, any projection failure, or a committed structure that
        fails :func:`assert_structure_physical`.
    """
    if mode not in ("second", "second_third", "second_third_first"):
        die(f"Unknown extract mode: {mode}")

    # Build the command.
    #
    # --wz_supercell passes TDEP's undocumented ``-wz`` option, which replaces the
    # spherical second-order cutoff with the Voronoi (super Wigner-Seitz) cell of
    # the supercell; ``-rc2`` is then not passed at all. It opens many more pair
    # shells for the same sample count, so it is skipped while the working
    # directory path contains "preconditioning", where the sample sets are too
    # small to determine them.
    use_wz = bool(getattr(ctx, "wz_supercell", None)) and "preconditioning" not in os.getcwd()
    if getattr(ctx, "wz_supercell", None) and not use_wz:
        print("[wz][info] --wz_supercell set but SKIPPED during preconditioning "
              "(underdetermined bootstrap; see tdep_extract_fc.py).", flush=True)
    if use_wz:
        print("[wz][EXPERIMENTAL] Using TDEP's hidden -wz "
              f"{ctx.wz_supercell} (Voronoi cutoff) INSTEAD of -rc2 "
              f"{ctx.cutoffFC2}. This option is UNSUPPORTED and may break the "
              "run. Do not use it for production without validating against a "
              "spherical-cutoff run on the same configurations.", flush=True)
        cmd = ["extract_forceconstants", "-wz"] + str(ctx.wz_supercell).split()
    else:
        cmd = ["extract_forceconstants", "-rc2", str(ctx.cutoffFC2)]

    if ctx.cutoffFC3 is not None and mode != "second":
        cmd += ["-rc3", str(ctx.cutoffFC3)]
    else:
        print("[info] no --rc3 or mode 'second'; skipping third-order extraction.", flush=True)

    if mode == "second_third_first":
        cmd.append("--firstorder")

    cmd_str = " ".join(cmd)
    print(f" Running: {cmd_str}", flush=True)

    rc, out, err = safe_run(cmd_str)

    with open("extract_forceconstants.log", "w") as logf:
        logf.write(out)
        if err:
            logf.write("\n--- STDERR ---\n")
            logf.write(err)

    # extract_forceconstants can exit with a non-zero code (rc=4) because of a
    # benign IEEE_OVERFLOW flag that the gfortran runtime turns into an ERROR STOP
    # at termination, after the fit has completed and the output files have been
    # written. A non-zero code is therefore fatal only when outfile.forceconstant is
    # missing or empty; otherwise a warning is printed and the finished fit is kept.
    fc_ok = os.path.exists("outfile.forceconstant") and os.path.getsize("outfile.forceconstant") > 0
    if rc != 0 and not fc_ok:
        die(f"extract_forceconstants failed (rc={rc}) with no outfile.forceconstant. "
            f"See extract_forceconstants.log")
    if rc != 0:
        print(f"[warn] extract_forceconstants exited rc={rc} but wrote a valid "
              f"outfile.forceconstant (likely a benign IEEE_OVERFLOW at exit); "
              f"continuing.", flush=True)
    if not fc_ok:
        die("extract_forceconstants failed (missing/empty outfile.forceconstant)")

    # Copy the results to the infile.* names read by the next stage.
    try:
        shutil.copy("outfile.forceconstant", "infile.forceconstant")
    except Exception:
        pass

    if os.path.exists("outfile.forceconstant_thirdorder"):
        shutil.copy("outfile.forceconstant_thirdorder", "infile.forceconstant_thirdorder")

    ensure_outfile_new_ucposcar()

    # Wyckoff projection of the first-order structure update.
    if do_wyckoff_project and mode == "second_third_first":
        if os.path.exists(displacements_path):
            try:
                damping = float(getattr(ctx, "wyckoff_damping", 0.8))
                print(f"[wyckoff] projecting displacement onto symmetry (damping={damping:g})",
                      flush=True)
                out_path, _ = generate_projected_structure(
                    refined_ucposcar_path="infile.ucposcar",
                    displacements_path=displacements_path,
                    out_path=projected_out_path,
                    symprec=1e-5,
                    damping=damping,
                )
                shutil.copy(out_path, "outfile.new_ucposcar")
                print("[wyckoff] applied projected update -> outfile.new_ucposcar", flush=True)
            except Exception as exc:
                # A "[guardrail:" message means the projector rejected the structure
                # (too-large displacement, too-close pair or changed space group).
                if "[guardrail:" in str(exc):
                    die(f"[wyckoff] symmetry guardrail violated -> aborting:\n{exc}")
                # Any other projection failure also aborts; TDEP's own first-order structure
                # update is not used as a fallback. The projector's orbit match is
                # setting-dependent: Wyckoff_Projector.load_spacegroup takes the table
                # record of spglib's default setting and generate_projected_structure tries
                # every record of the group, but no transformation_matrix/origin_shift is
                # applied, so a cell whose setting is absent from the table fails here.
                die(f"[wyckoff] projection failed -> aborting (no silent fallback):\n{exc}\n"
                    f"Hint: the orbit match is setting-dependent and a cell in a setting absent "
                    f"from the Wyckoff table fails; re-standardise the starting cell (e.g. with "
                    f"refine_structure) and start again.")
        else:
            print("[wyckoff] no displacement file -> skipping projection", flush=True)

    # Physicality gate on the committed structure, whether it came from TDEP's
    # --firstorder update or from the projection.
    if mode == "second_third_first" and os.path.exists("outfile.new_ucposcar"):
        assert_structure_physical("outfile.new_ucposcar")
