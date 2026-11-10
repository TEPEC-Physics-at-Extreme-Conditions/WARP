# This file is part of WARP: https://github.com/TEPEC-Physics-at-Extreme-Conditions/WARP
# Copyright (C) 2026 The WARP developers (see the NOTICE file)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# WARP is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. WARP is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; see the LICENSE file for details.
"""Thermal sampling and calculator dispatch for one work directory.

Generates canonical (optionally quantum) configurations with TDEP's
``canonical_configuration``, converts each to a Quantum ESPRESSO input with
:mod:`warpt.structconv`, patches the ``&CONTROL`` block (per-sample ``outdir``,
``disk_io = 'none'``) and runs the QE or MLIP jobs through
:func:`warpt.runner.run_jobs_parallel`. Sample outputs that already contain
``JOB DONE.`` are kept and not recomputed.

Reads, in the current working directory: ``contcar_conf*`` (if present),
``infile.ucposcar``, ``infile.ssposcar`` and the master QE input deck named by
``ctx.master_scf`` (copied in when absent).
Writes: ``infile.isotopes`` (when ``ctx.masses`` is set; ``canonical_configuration`` then
runs with ``--readiso``), ``contcar_conf*``, ``Clean_<n>.pos``, ``<master>.samples<n>``,
``<master>.samples<n>.out``, the directories ``outdir_sample_<n>/`` and, for
chunked MLIP batches, ``mlip_batch.<k>.list`` and ``mlip_batch.<k>.list.log``.

Environment variables: ``MAX_CONC`` (QE concurrency, default 4), ``MAX_CONC_MLIP``
(MLIP concurrency, default ``MAX_CONC`` or 8), ``WARPT_MLIP_BATCH`` (``0``,
``false`` or ``no`` disables MLIP batching), ``WARPT_MLIP_PROC_GB`` (per-process
base memory in GB for the batch cap), and ``SLURM_MEM_PER_NODE``,
``SLURM_MEM_PER_CPU``, ``SLURM_CPUS_PER_TASK``, ``SLURM_NTASKS`` (memory budget).

Public functions: ``do_sampling_and_run``.
"""
import os
import glob
import shutil
import subprocess
from typing import Tuple, List

from .utilities import die
from .io_poscar import normalize_poscar_to_direct_scale1
from .io_workflow import write_isotopes
from .runner import build_cmd, run_jobs_parallel, job_done
from .timing import stage

from . import structconv as _structconv


# ---------------------------------------------------------------------------
# Memory budget for chunked MLIP batching.
#
# Each chunk is a separate interpreter that loads the full checkpoint, so node
# memory scales with the number of chunks, not with the core count. The peak is
# during the checkpoint load (a training checkpoint also holds optimizer state,
# so the file is larger than the resident model); chunks start together, so the
# peaks add.
_MLIP_BASE_BYTES   = 1.5 * 1024 ** 3   # interpreter + torch + neighbour lists
_MLIP_CKPT_FACTOR  = 1.3               # load transient relative to on-disk size
_MLIP_MEM_HEADROOM = 0.85              # leave the allocation some slack


def _slurm_mem_budget_bytes():
    """Memory budget in bytes: ``SLURM_MEM_PER_NODE`` (MB), else ``SLURM_MEM_PER_CPU`` (MB) times
    ``SLURM_CPUS_PER_TASK`` or ``SLURM_NTASKS``; None when neither is set or parsable."""
    mb = os.environ.get("SLURM_MEM_PER_NODE")
    if mb:
        try:
            return int(float(mb)) * 1024 ** 2
        except ValueError:
            pass
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    ncpu = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
    if per_cpu and ncpu:
        try:
            return int(float(per_cpu)) * int(ncpu) * 1024 ** 2
        except ValueError:
            pass
    return None


def _model_ckpt_bytes(pwscf_run):
    """On-disk size of the checkpoint named by --ml_params; 0 if not resolvable."""
    toks = (pwscf_run or "").split()
    for flag in ("--ml_params", "-ml_params"):
        if flag in toks:
            i = toks.index(flag) + 1
            if i < len(toks):
                try:
                    return os.path.getsize(toks[i])
                except OSError:
                    return 0
    return 0


def _mlip_chunk_cap(pwscf_run, requested):
    """Cap the number of concurrent MLIP processes by the Slurm memory budget.

    Parameters
    ----------
    pwscf_run : str
        Calculator command line; the path after ``--ml_params`` (or ``-ml_params``) gives
        the checkpoint size.
    requested : int
        Concurrency asked for.

    Returns
    -------
    (cap, note) : tuple[int, str]
        ``cap`` is ``min(requested, floor(0.85 * budget / per_process))`` with
        ``per_process = base + 1.3 * checkpoint_bytes``, where ``base`` is 1.5 GB or
        ``WARPT_MLIP_PROC_GB`` (GB) when set, and never below 1. ``requested`` is returned
        unchanged when no Slurm budget is visible or the checkpoint size cannot be read.
        ``note`` is the line the caller logs.
    """
    budget = _slurm_mem_budget_bytes()
    if not budget:
        return requested, "no Slurm memory limit visible -- concurrency left at %d" % requested

    ckpt = _model_ckpt_bytes(pwscf_run)
    if not ckpt:
        return requested, ("checkpoint size unknown (--ml_params unreadable) -- "
                           "concurrency left at %d, memory NOT verified" % requested)

    # WARPT_MLIP_PROC_GB (GB) overrides the fixed per-process part of the estimate;
    # the default 1.5 GB covers interpreter, torch and neighbour lists for small
    # supercells; raise it for large cells.
    base = _MLIP_BASE_BYTES
    override = os.environ.get("WARPT_MLIP_PROC_GB")
    if override:
        try:
            base = float(override) * 1024 ** 3
        except ValueError:
            pass
    per_proc = base + _MLIP_CKPT_FACTOR * ckpt
    cap = max(1, int((budget * _MLIP_MEM_HEADROOM) // per_proc))
    g = 1024 ** 3
    note = ("memory budget %.0f GB, ~%.1f GB per process (%.1f GB checkpoint) "
            "-> at most %d concurrent" % (budget / g, per_proc / g, ckpt / g, cap))
    if cap < requested:
        note += "; REQUESTED %d, CAPPED TO %d" % (requested, cap)
    return min(cap, requested), note



def _ensure_local_master(master_scf: str) -> str:
    """Copy the master QE input deck into the current working directory unless a file of
    its basename exists there. Returns the basename used locally.
    """
    local_master = os.path.basename(master_scf)
    if not os.path.exists(local_master):
        shutil.copy(master_scf, local_master)
    return local_master


def _patch_control_block(lines: List[str], outdir_path: str) -> List[str]:
    """Rewrite ``outdir`` and ``disk_io`` inside the ``&CONTROL`` namelist.

    Existing ``outdir``/``disk_io`` lines in ``&CONTROL`` are replaced by
    ``outdir = '<outdir_path>'`` and ``disk_io = 'none'``; missing ones are inserted
    before the namelist's closing ``/``. Every other line is returned unchanged.
    Returns the new list of lines.
    """
    new_lines = []
    inside_control = False
    inserted_outdir = False
    inserted_diskio = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("&control"):
            inside_control = True
            new_lines.append(line)
            continue

        if inside_control and stripped == "/":
            if not inserted_outdir:
                new_lines.append(f" outdir = '{outdir_path}'\n")
            if not inserted_diskio:
                new_lines.append(" disk_io = 'none'\n")
            inside_control = False
            new_lines.append(line)
            continue

        if inside_control and stripped.lower().startswith("outdir"):
            new_lines.append(f" outdir = '{outdir_path}'\n")
            inserted_outdir = True
        elif inside_control and stripped.lower().startswith("disk_io"):
            new_lines.append(" disk_io = 'none'\n")
            inserted_diskio = True
        else:
            new_lines.append(line)

    return new_lines


@stage("sampling + forces")
def do_sampling_and_run(ctx, state, ns: int, *, force_regenerate: bool = False) -> Tuple[int, int]:
    """Generate ``ns`` thermal configurations and run the calculator on each.

    Parameters
    ----------
    ctx : context_state.WarpContext
        Uses ``temp`` (K), ``quantum``, ``max_freq`` (THz) and ``T_debye`` (K) (the last two
        only while ``state.use_fake_fc`` is set), ``master_scf`` and ``pwscf_run``.
    state : context_state.WarpState
        ``current_dir`` is set to the working directory; ``use_fake_fc`` selects the
        Debye-model options of ``canonical_configuration``.
    ns : int
        Number of configurations.
    force_regenerate : bool
        Only changes the wording of the resume log line. A complete set of
        ``contcar_conf*`` files (at least ``ns``) is always reused; a partial set is
        deleted and regenerated.

    Returns
    -------
    (n_configs, n_jobs) : tuple[int, int]
        Number of ``contcar_conf*`` files found and number of calculator jobs queued
        (outputs already containing ``JOB DONE.`` are not queued).

    Side effects
    ------------
    Runs in the current working directory. Normalises ``infile.ucposcar`` and
    ``infile.ssposcar`` and runs ``canonical_configuration`` when configurations are
    generated. Writes ``Clean_<n>.pos``, ``<master>.samples<n>`` and
    ``outdir_sample_<n>/`` for every configuration, deletes incomplete
    ``<master>.samples<n>.out`` files, and launches the calculator jobs (with MLIP
    batching, one process per ``mlip_batch.<k>.list`` chunk).

    Raises
    ------
    SystemExit
        Via :func:`die` when ``canonical_configuration`` fails or no ``contcar_conf*``
        files exist afterwards.
    RuntimeError
        From :func:`warpt.runner.run_jobs_parallel` when a job ends without ``JOB DONE.``.
    """
    try:
        state.current_dir = os.getcwd()
    except Exception:
        pass

    stat = "Bose-Einstein" if bool(ctx.quantum) else "Boltzmann"
    print(f" Using {stat} statistics", flush=True)

    existing_conts = sorted(glob.glob("contcar_conf*"))
    # Resume-safe sampling. A complete set of contcar_conf* (at least ns) means the
    # configurations of this step already exist, either from finished sampling or
    # from an interrupted run of the same geometry or step. Each step has its own
    # directory, so the configurations always match the current force constants and
    # geometry. The set is reused, even when force_regenerate is set, so that
    # calculations that already finished are not repeated. Only an absent or
    # incomplete set is (re)generated; a partial set is deleted and regenerated from
    # scratch, because canonical_configuration writes the whole set in one call.
    if len(existing_conts) >= ns:
        why = "force_regenerate set but complete set present" if force_regenerate else "complete set present"
        print(f"[resume] {len(existing_conts)} contcar_conf* (>= ns={ns}); reusing ({why}).", flush=True)
    else:
        if existing_conts:
            print(f"[sampling] partial contcar set ({len(existing_conts)} < ns={ns}); clearing and regenerating.", flush=True)
            for f in existing_conts:
                try:
                    os.remove(f)
                except Exception:
                    pass
        # Rewrite both POSCARs as scale-1 Direct files before canonical_configuration.
        normalize_poscar_to_direct_scale1("infile.ucposcar", "infile.ucposcar")
        normalize_poscar_to_direct_scale1("infile.ssposcar", "infile.ssposcar")

        cmd = ["canonical_configuration", "-t", str(ctx.temp), "-n", str(ns)]
        if getattr(state, "use_fake_fc", False):
            if getattr(ctx, "max_freq", None):
                cmd += ["--maximum_frequency", str(ctx.max_freq)]
            if getattr(ctx, "T_debye", None):
                cmd += ["--debye_temperature", str(ctx.T_debye)]
        if bool(ctx.quantum):
            cmd.append("--quantum")
        if getattr(ctx, "masses", None):
            write_isotopes(ctx.masses)
            cmd.append("--readiso")

        print(f"[canonical] {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, check=False)
        if rc.returncode != 0:
            die(f"canonical_configuration failed (rc={rc.returncode})")

    conts = sorted(glob.glob("contcar_conf*"))
    if not conts:
        die("No contcar_conf* files found after canonical_configuration/resume.")

    local_master = _ensure_local_master(ctx.master_scf)

    jobs = []

    # Concurrency is set separately for QE and MLIP.
    #   QE   -- pw.x parallelises internally; MAX_CONC (default 4) is the number of
    #           pw.x processes allowed to run at once.
    #   MLIP -- process startup (interpreter, import, checkpoint load) dominates a
    #           single evaluation, so configurations are grouped into batches that
    #           each reuse one calculator (WARPT_MLIP_BATCH, default on).
    #
    # Environment variables:
    #   MAX_CONC          QE concurrency (default 4)
    #   MAX_CONC_MLIP     MLIP concurrency: the requested number of batch chunks when
    #                     batching is on (before the memory cap of _mlip_chunk_cap),
    #                     or of simultaneous single-configuration processes when it
    #                     is off (default: MAX_CONC, else 8)
    #   WARPT_MLIP_BATCH  "0", "false" or "no" runs one process per configuration;
    #                     any other value (default) runs chunked batches
    is_mlip = "MLIP_interface" in (ctx.pwscf_run or "")
    if is_mlip:
        max_concurrent = int(os.environ.get("MAX_CONC_MLIP",
                                            os.environ.get("MAX_CONC", "8")))
        use_batch = os.environ.get("WARPT_MLIP_BATCH", "1") not in ("0", "false", "no")
    else:
        max_concurrent = int(os.environ.get("MAX_CONC", "4"))
        use_batch = False

    for idx, cont in enumerate(conts, start=1):
        scf_in = f"{os.path.basename(ctx.master_scf)}.samples{idx}"
        out_path = scf_in + ".out"

        # --- Step 0: write a clean POSCAR (header and the first nat coordinates only) ---
        with open(cont, "r") as fin:
            lines = fin.readlines()

        species_counts = list(map(int, lines[6].split()))
        nat = sum(species_counts)

        clean = f"Clean_{idx}.pos"
        with open(clean, "w") as fout:
            fout.writelines(lines[:7])          # title, scale, lattice, species, counts
            fout.write(lines[7])                # Direct/Cartesian line
            fout.writelines(lines[8:8 + nat])   # first nat coordinates only

        # --- Step 1: structconv poscar -> pwscf (QE input), in-process; atom order is preserved ---
        _structconv.poscar_to_pwscf(clean, local_master, scf_in)

        # --- Step 2: patch &CONTROL outdir + disk_io ---
        outdir_path = f"./outdir_sample_{idx}/"
        os.makedirs(outdir_path, exist_ok=True)

        with open(scf_in, "r") as f:
            qe_lines = f.readlines()

        new_lines = _patch_control_block(qe_lines, outdir_path)
        with open(scf_in, "w") as f:
            f.writelines(new_lines)

        # --- Step 3: skip if already completed successfully ---
        if job_done(out_path):
            continue
        if os.path.exists(out_path) and (not job_done(out_path)):
            try:
                os.remove(out_path)
            except Exception:
                pass

        # --- Step 4: queue job ---
        cmd = build_cmd(ctx.pwscf_run, scf_in)  # adds "-input scf_in"
        jobs.append((cmd, out_path))

    # --- Step 5: run queued jobs ---
    if jobs:
        if use_batch and len(jobs) > 1:
            # Chunked batches: the queued jobs are split round-robin into up to `cap`
            # chunks (never more than there are jobs) and each chunk is run by one
            # MLIP_interface process (`--batch <listfile>`), which loads the
            # checkpoint once. Each list line is "<input> <output>"; the chunk's own
            # stdout goes to <listfile>.log. Every structure still writes its own
            # output ending in JOB DONE., so resume and the collect barrier work as for
            # single jobs; a chunk that fails part-way leaves the outputs it finished
            # valid and the rest absent.
            cap, memnote = _mlip_chunk_cap(ctx.pwscf_run, int(max_concurrent))
            print("[warpt] MLIP batching: %s" % memnote, flush=True)
            nchunk = max(1, min(cap, len(jobs)))
            chunks = [jobs[i::nchunk] for i in range(nchunk)]
            chunks = [c for c in chunks if c]
            base = [c for c in (ctx.pwscf_run or "").split()]
            batch_jobs = []
            for k, chunk in enumerate(chunks):
                listfile = os.path.join(os.getcwd(), f"mlip_batch.{k}.list")
                with open(listfile, "w") as fh:
                    for cmd, out_path in chunk:
                        fh.write(f"{cmd[cmd.index('-input') + 1]} {out_path}\n")
                batch_jobs.append((base + ["--batch", listfile], listfile + ".log"))
            run_jobs_parallel(batch_jobs, max_concurrent=len(batch_jobs), poll_s=10.0)
        else:
            run_jobs_parallel(jobs, max_concurrent=max_concurrent, poll_s=10.0)

    return (len(conts), len(jobs))