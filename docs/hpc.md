# Running on a cluster

WARP is one long-lived Python process that launches many short ones. This page describes what it
detects about a batch allocation, how it spreads force evaluations across it, and how to carry a
relaxation that outlives a single job through a chain of jobs. Everything here is generic: WARP has
no knowledge of any particular machine or scheduler beyond the environment variables named below.

## The shape of a job

The driver itself is cheap: it waits on subprocesses almost all the time. What consumes the
allocation is the force evaluations, and to a lesser extent the TDEP tools. A batch script
therefore looks like an ordinary job that happens to run Python:

```bash
#!/bin/bash -l
#SBATCH --job-name=warp
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
#SBATCH --output=warpt_run.log
#SBATCH --open-mode=append

module load <modules providing pw.x>
source <python environment with warpt>/bin/activate
export PATH="<tdep installation>/bin:$PATH"

export OMP_NUM_THREADS=1
export MAX_CONC=4
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

python3 -m warpt \
    --calculator QE -pwx "srun -n 32 pw.x" \
    -m master.scf.in -ucp start.vasp \
    --mode vc-relax -t 300 --press_ext 0.0 \
    -sc 2 2 2 -n 128 -rc2 5.0 --rc3 3.5 -pct 0.05 --quantum
```

Here four `pw.x` runs of 32 MPI ranks each fill the 128 tasks of the node. Several `srun` steps
then run side by side inside one allocation; depending on the Slurm version and the site
configuration, each may need `srun --exact` to be confined to its own share of the cores.

`PYTHONUNBUFFERED=1` is recommended: without it the driver's output reaches the job log in large,
late blocks, which makes a running job hard to follow.

## What WARP detects

Only one thing, and only for the machine-learned backend: **the memory of the allocation**. It is
read from `SLURM_MEM_PER_NODE`, or from `SLURM_MEM_PER_CPU` multiplied by `SLURM_CPUS_PER_TASK` or
`SLURM_NTASKS`. Nothing else about the allocation is inspected — not the core count, not the
walltime, not the node list.

Everything else is set in the job script:

| What | How |
| --- | --- |
| how many force evaluations run at once | `MAX_CONC`, or `MAX_CONC_MLIP` for the potential backend |
| how each evaluation is parallelised | the `--pwx` (or `--vasp_cmd`) command: `srun -n ...`, `mpirun -n ...`, a wrapper |
| threads for everything that is not pinned | `OMP_NUM_THREADS` |

See [Environment variables](environment.md) for the full list.

## Batching force evaluations

Quantum ESPRESSO and VASP samples are run one process per configuration, at most `MAX_CONC` at a
time. The job pool starts new processes as old ones finish, and polls with an adaptive back-off —
from 0.05 s, growing by half each idle poll, capped at 10 s, and reset whenever a process finishes —
so a step of many quick evaluations is not dominated by polling and a step of long ones costs
almost nothing to watch. A process that exits without writing `JOB DONE.` stops the run at once;
WARP does not terminate the processes still running.

The machine-learned backend is different, because loading the model costs more than evaluating one
configuration. The queued evaluations are split round-robin into at most `MAX_CONC_MLIP` chunks and
each chunk is handed as a list file to one backend process, which loads the model once and walks
its whole list. Each structure still writes its own output ending in `JOB DONE.`, so resume and the
collection barrier are unchanged, and a chunk that dies leaves its finished outputs valid.

That is where the memory budget matters. Every chunk is a separate interpreter holding the whole
model, so memory scales with the number of chunks rather than with the core count, and the chunks
start together so their peaks coincide. WARP caps the chunk count at 85 % of the visible budget
divided by its per-process estimate, and prints what it decided. Request memory explicitly in the
job script: without a value to read, nothing caps the concurrency and a large model will swap the
node.

## Threads

Two stages are pinned rather than left to the environment: `extract_forceconstants` is always run
with `OMP_NUM_THREADS=1`, and the phonon calculations with `OMP_NUM_THREADS=16`. The previous value
is restored afterwards, including when the call fails, so the setting never leaks into the rest of
the run. The calculator processes inherit the exported value, so for `pw.x` with one MPI rank per
core keep it at 1.

## Time limits

WARP has no notion of a walltime. It runs until it converges or until it is killed, and it is
designed for the second case: every stage writes its outputs as it finishes them, and a restarted
run re-derives what is complete from those files. A killed job loses the force evaluations that
had not finished and the post-processing (collection, fit, stress, phonons) of the step in flight;
every finished force evaluation is kept. With a DFT calculator that post-processing is cheap next
to the force evaluations; with the machine-learned backend it can be the larger share of a step.

Two habits make that reliable:

- **Append to the log, never overwrite it.** Use `--output` with `--open-mode=append`, or pipe
  through `tee -a`, so that the history of the earlier attempts survives.
- **Read the timing tables.** From the second iteration on, WARP prints two tables at the top of
  every iteration — wall time per external command and per workflow stage — so a job that is
  killed still leaves a breakdown of where its hours went, up to its last completed iteration. Use
  them to size the next job.

## Chaining jobs

Because the resume is a plain re-run, a relaxation longer than one job is a dependency chain of
identical jobs:

```bash
jid=$(sbatch --parsable warp.job.sh)
for i in $(seq 1 5); do
    jid=$(sbatch --parsable --dependency=afterany:$jid warp.job.sh)
done
```

`afterany` rather than `afterok`: the point of the chain is to continue after a job that ran out of
time, which is a failure as far as the scheduler is concerned.

Every job in the chain runs the **same command in the same directory**. A job that starts on a
converged tree walks it, prints `[lattice][resume] iteration already converged here.` and returns;
that is the normal end of a chain.

!!! warning "Guard the chain, not the run"
    WARP does not stop a chain, so the chain keeps relaunching a finished run — cheaply, since every
    stage is recognised as complete, but not for free. Only the converged lattice step contains
    `final_structure.vasp`, so a test placed after the `cd` in the job script ends the remaining
    jobs early:

    ```bash
    if ls iter_*/lattice_/final_structure.vasp >/dev/null 2>&1; then exit 0; fi
    ```

    Alternatively, cancel the remaining jobs when the file appears.

!!! warning "Do not change options between links of a chain"
    Resume reads the files, not the command line. Every job in the chain must carry an identical
    command, or the finished stages of one setting will be reused under another.

## Sizing a run

Three numbers dominate the cost, and they multiply:

- `-n`, the configurations per step — the ensemble size, and the quantity the statistical error of
  both the residual force and the stress depends on;
- the number of position steps per iteration, which falls as the structure settles;
- the number of iterations, i.e. lattice steps, which is set by how far the starting cell is from
  the finite-temperature one.

The preconditioning schedule adds its own configurations once, at the start. Choose the sample
counts as multiples of the concurrency, so that no process sits idle at the end of a step. Every
position and lattice step draws `-n` new configurations; a step whose structure update was small
also fits on the previous step's fitted sample set, which already includes anything that step had
itself merged, so the ensemble can grow over several consecutive small steps at no extra cost.
