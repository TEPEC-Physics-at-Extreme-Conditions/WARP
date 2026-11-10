# Environment variables

A few settings are read from the environment rather than from the command line, because they
describe the machine rather than the physics: how many force evaluations may run at once, how the
machine-learned backend is batched, how much memory the allocation provides and how much timing
output is printed. This page lists all of them, with their defaults and their effect.

## WARP's own variables

| Variable | Default | Effect |
| --- | --- | --- |
| `WARPT_MLIP_BATCH` | `1` (on) | Controls how the machine-learned backend is launched. Set it to `0`, `false` or `no` to run one process per configuration; any other value keeps the default chunked batches, in which each process loads the model once and walks a list of configurations. Only consulted when the calculator command is the backend (`warpt.MLIP_interface`), and batches are formed only when a step has more than one evaluation to run. |
| `WARPT_MLIP_PROC_GB` | 1.5 GB | Overrides, in GB, the fixed per-process part of the memory estimate that caps the number of concurrent backend chunks. The default covers interpreter, torch and neighbour lists for a small supercell; raise it for large cells. The model checkpoint's own size is added on top automatically. |
| `WARPT_VASP_CMD` | unset | The command that runs VASP, read by `warpt.VASP_interface` when it is called without `--vasp_cmd`. A run with `--calculator VASP` sets it from `--vasp_cmd`, replacing any exported value, so that the sample processes inherit a command containing spaces intact. Export it only when running `python3 -m warpt.VASP_interface` directly. |
| `WARPT_TIMING` | `1` | Set it to `0` to switch off every timing record, all `[time]` output and the timing tables. |
| `WARPT_TIME_MIN_S` | `1.0` | Minimum duration, in seconds, for a single command to get its own `[time]` line. Shorter calls are still counted in the summary tables; they are just not printed one by one. |

!!! note "The two timing variables are read at import"
    They are read once, when the module is first imported, so export them before starting the run.
    Changing them mid-run has no effect.

## Concurrency

| Variable | Default | Effect |
| --- | --- | --- |
| `MAX_CONC` | `4`; `1` with `--calculator VASP` | How many Quantum ESPRESSO or VASP processes run at once. With `--calculator VASP`, WARP sets it to 1 when it is not exported, because each VASP run is already parallel. `pw.x` parallelises internally as well, so the product of this and the `--pwx` process count is what has to fit the allocation. |
| `MAX_CONC_MLIP` | `MAX_CONC` if set, otherwise `8` | How many machine-learned backend processes run at once. With batching on it is the number of chunks requested (never more than there are evaluations), which the memory cap below may reduce; with batching off it is the number of simultaneous single-configuration processes, and no memory cap applies. |

Sample counts that are multiples of the concurrency avoid idle processes at the end of a step;
choose `-n` and `--precond_schedule` accordingly.

## Memory budget

When the backend is batched, each chunk is a separate interpreter holding the whole model, so the
memory needed scales with the number of chunks and the peaks coincide. WARP reads the allocation
from the scheduler and caps the chunk count accordingly:

| Variable | Used as |
| --- | --- |
| `SLURM_MEM_PER_NODE` | the budget, in MB; preferred when set |
| `SLURM_MEM_PER_CPU` | the budget, in MB per CPU, multiplied by the count below |
| `SLURM_CPUS_PER_TASK` | the CPU count for that product |
| `SLURM_NTASKS` | the CPU count when `SLURM_CPUS_PER_TASK` is unset |

The cap is 85 % of the budget divided by the per-process estimate — `WARPT_MLIP_PROC_GB`, or 1.5
GB, plus 1.3 times the on-disk size of the checkpoint given by `--ml_params` — and is never below 1.
It is announced on every batched sampling step, in one line:

```text
[warpt] MLIP batching: memory budget ... GB, ~... GB per process (... GB checkpoint) -> at most N concurrent; REQUESTED M, CAPPED TO N
```

The `REQUESTED ..., CAPPED TO ...` part appears only when the cap is below the request. If no
budget is visible, or the checkpoint size cannot be read, the concurrency is left at the requested value
and the line says so — including the words `memory NOT verified` in the second case. Outside Slurm
nothing caps the concurrency, so set `MAX_CONC_MLIP` by hand.

## Threads

| Variable | Effect |
| --- | --- |
| `OMP_NUM_THREADS` | Read **and written**. WARP pins it to 1 for the duration of each `extract_forceconstants` call and to 16 for the phonon calculations (`phonon_dispersion_relations`), restoring the previous value — or unsetting it again — afterwards, even if the call fails. The exported value applies to everything else, including `ifc_pressure` and the calculator processes, which inherit the environment. |

## Setting them

```bash
export OMP_NUM_THREADS=8
export MAX_CONC=1
export MAX_CONC_MLIP=1        # one backend process; one model in memory
export WARPT_TIME_MIN_S=5     # quieter timing output

python3 -m warpt ... 2>&1 | tee -a warpt_run.log
```

WARP reads no other environment variables: everything else that changes a run is a command-line
option. [Running on a cluster](hpc.md) puts these into the context of a batch job.
