# Workspaces, the workbench, local calculations and registered runs

How hypothesis agents work beyond requesting calculations: what they read and write, the
commands they run, the calculations that run locally on an interactive node, and the earlier
runs they bring into the record.

| | What it is | Approved by the researcher | Evidence |
|---|---|---|---|
| Read access | Any file the researcher can read, except the files listed below | — | no |
| Workspace | A writable directory per hypothesis | — | no |
| Workbench | Shell commands and scripts in the workspace, in a sandbox | no; `/workbench off` turns it off | no |
| Registered run | A finished FHI-aims run computed outside CEDAR, brought into the record | no; nothing runs | yes, marked as computed outside CEDAR |
| Local calculation | An examined calculation run on the interactive node CEDAR runs on | yes: `/approve submit` | yes, as for a Slurm run |

## What agents read and write

A hypothesis agent's file tools read the whole filesystem with the researcher's Unix
permissions, except:

- secrets: CEDAR's configuration directory (API keys, profiles) and well-known credential
  stores (`~/.ssh`, `~/.gnupg`, `~/.netrc`, `~/.aws`, `~/.config/gcloud`, `~/.kube`,
  `~/.docker`);
- other hypotheses' work: CEDAR's data directory (event log, stored files, transcripts,
  other workspaces) and the compute profile's runs directory, except the run directories of
  the agent's own calculations;
- `/proc`, `/sys` and `/dev`, which are not files.

The agent's own workspace, `<data>/workspaces/<investigation>/<hypothesis>/`, and the
directories its own and the shared registered runs came from stay readable. Reads are
streamed: a large file is read only up to the requested lines, and a search stops after
2 GiB (`agents/toolkits/files.py`).

An agent writes only in its workspace: `write_file`, `delete_file`, and its workbench
commands. `request_calculation` takes a structure or input file from the workspace by path,
and `run_analysis` can copy files a script wrote into `analyses/<analysis_id>/` there.

## The workbench

`run_command` runs a shell command with bash in the agent's workspace
(`engine/workbench.py`, `engine/sandbox.py`). Agents use it to check things: build or
inspect a structure with ASE, test a script on a sample file, search outputs, run tests.

With bubblewrap, a command:

- sees the filesystem read-only, with the paths above hidden (empty), and the profile's DFT
  binaries and the MPI and Slurm launchers hidden too; the agent's run directories are shown
  again inside the hidden runs directory;
- writes only in the workspace and a private `/tmp` (the machine's own `/tmp` entries are
  visible, read-only); its `$HOME` is `.home` in the workspace;
- has no network, so no web, ssh or scheduler;
- gets a clean environment without credentials, CEDAR's Python first on its search path,
  and thread variables matching its cores;
- runs in its own process tree, pinned to its cores, and is killed at its time limit or
  when its agent's work is halted.

Without bubblewrap, only `python <script>.py` runs, under the analysis guard: reads limited
the same way, writes only in the workspace, no programs, no sockets.

| | Login node | Interactive node |
|---|---|---|
| Cores per command | 1 | what the allocation has free |
| Commands at once | 2, across all agents | as cores allow |
| Time per command | 10 minutes by default, at most 30 | 10 by default, at most 60, never past the allocation's end |
| Memory, disk | no limit | no limit |

The numbers are defaults of the compute profile's optional `local` section
(`LocalExecution`), which may also name further Python environments for commands and
further programs to hide.

Every command is recorded as a `workbench_run_recorded` event: the command, purpose,
directory, sandbox, host and allocation, cores, time limit, outcome, exit code, duration,
what it printed (stored by hash; the agent sees its start and end, and the whole of it is in
`workbench-output/<run>.txt` in the workspace), and the content hash of every workspace file
it created or changed. That is context, never evidence. The researcher turns the workbench
off and on with `/workbench off` and `/workbench on` (`workbench_policy_changed`).

## Where CEDAR runs

CEDAR counts as being on an interactive node only when `SLURM_JOB_ID` is set, Slurm reports
that job as running, and this machine is one of its nodes (`hpc/node.py`). The allocation's
cores on this node (`SLURM_CPUS_ON_NODE`) and its time left (`squeue -o %L`) then bound
local work. Anything else is a login node. The agents' overview and the Chief Scientist's
`investigation_state` say which, and whether calculations can run locally.

Local calculations start where the dispatcher runs. The dispatcher records its host in
`dispatcher.host`; `cedar dispatch --stop` refuses to signal a dispatcher on another
machine, and the chat says when the dispatcher runs elsewhere. Run the chat and the
dispatcher on the same node: the data directory is often on a shared filesystem.

## Local calculations

A task with `execution: local` runs on the interactive node CEDAR runs on, inside the
researcher's allocation, under `fhiaims.run` 0.2:

- its script, `local.sh`, is the approved job template without its `#SBATCH` lines, plus one
  `#CEDAR --ntasks=N --time=T` line; the examiner checks it like a job script, and its task
  count and time limit against the allocation's cores and time left (with five minutes to
  spare);
- on a login node the request is rejected at once: "DFT cannot run locally on login nodes";
- submission needs the researcher's `/approve submit <calc>` (or the approver, in full
  automation), as for Slurm.

The local backend (`hpc/local.py`) has the scheduler adapter's interface, so the journaled
submission, reconciliation by job name, the job monitor, run checks and agent wake-ups are
the same as for Slurm. It records each job under `<runs_root>/.cedar-local/`, starts it once
the allocation's cores are free, and runs it through a detached wrapper
(`hpc/local_runner.py`) that pins it to its cores, stops it at its time limit and records its
exit code. A run whose allocation ended before it recorded an exit code is FAILED; one on
another node whose allocation still runs is UNKNOWN. Evidence from a local run records where
it ran (`execution`, `host`, `allocation`).

A calculation's identity does not include where it runs: requesting on Slurm a calculation
identical to a healthy local one links the existing run instead of computing it again.

## Registered runs

`register_existing_run` (hypothesis agents, for their own hypothesis; the Chief Scientist,
shared or for one hypothesis) and `/register <dir> <note>` (the researcher, shared with every
agent) bring a finished FHI-aims run into the record (`engine/registration.py`). Nothing runs:

- `control.in`, `geometry.in` and the output are stored by hash;
- the output's health is checked with the protocol's markers (normal end, SCF and geometry
  convergence); there is no scheduler state, and the code version is recorded, not required;
- the settings the inputs used become the run's task, classified as a task declares them:
  physical and unknown keywords as controls, numerical ones as numerical settings, and the
  species blocks by the species-defaults level they equal (or a hash);
- a `run_registered` event creates a calculation already `QUALITY_ASSESSED`.

Readings of a registered run are `archived_computation` evidence; analyses of it are analysis
evidence carrying the limitation that it was computed outside CEDAR. Both are VALIDATED only
if its health checks passed. A registered run cannot be predicted: its outputs existed before
any prediction, so its evidence establishes the discrepancy or checks a hypothesis rather
than testing a prediction. An agent's registered runs are its hypothesis's alone; runs the
researcher or the Chief Scientist registers for no hypothesis are shared by every agent, but
each agent's readings of them stay its own.
