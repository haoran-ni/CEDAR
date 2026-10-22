# Scientific protocols

A protocol is the second layer between what agents write and what runs. For one calculation type it can:
- generate any input files an agent did not write
- examine the complete set of files that would run
- check a finished run's health, and read the few values every run of its kind prints

It never runs anything and never changes a file it was given. What to take from a healthy run is the agent's decision: it reads values the code printed, citing the lines, or computes them with an analysis script ([`ANALYSIS_SCRIPTS.md`](ANALYSIS_SCRIPTS.md)).

A code has far more input keywords, and far more situations in which they make sense, than any fixed list can cover. So from `fhiaims.run` 0.3 on, the examination checks deterministically only what needs no knowledge of the code's keywords: what runs on the machine, and that the files say what the task declares. Whether the inputs make sense is judged by the **input reviewer**, an agent that reads the files with the installed code's manual and source and the code's documentation on the web ([below](#the-input-reviewer)). Run checks after the run stay deterministic: they read the code's own success and failure markers.

Everything about a particular HPC system comes from the user-approved compute profile. Every setting a test depends on is declared by the agent in the task.

This document grows with each implemented protocol.

## `fhiaims.run` 0.3

One FHI-aims run of one structure, at fixed geometry or relaxing it, through Slurm or locally on the interactive node CEDAR runs on. New requests use this version; 0.2 and 0.1 stay for the calculations recorded under them, which keep their rules ([below](#fhiaimsrun-02-and-01)).

### Declared variables

Any `control.in` keyword may be declared, named as in `control.in`, with the same value conventions as 0.2, and `species_defaults` names a level directory of the profile's species defaults. No list of known keywords limits what agents write; the agent should declare every setting its test depends on, since undeclared settings are shown to the reviewer as such. `postprocess_anyway` stays forbidden: it would hide an unconverged SCF cycle from the run check.

### Deterministic examination

`ERROR` rejects the inputs at once, back to the agent; `REVIEW` needs the researcher, whatever the input reviewer decides; `INFO` findings are facts recorded for the reviewer.

| What | Finding |
|---|---|
| Missing files; unreadable `geometry.in` lines; no atoms | `ERROR` |
| A declared keyword missing from `control.in`, set with other arguments, or set twice | `ERROR` |
| `postprocess_anyway`; a declared value that cannot be written as `control.in` arguments; a `species_defaults` level that is not in the profile | `ERROR` |
| An element without a species block | `ERROR` |
| Inputs identical to a usable calculation's | `ERROR` (link it instead) |
| The job script, examined against the approved profile as for 0.2: commands, directives, resources, the task's budget, and for a local run the allocation | `ERROR` or `REVIEW` as for 0.2 |
| A keyword `control.in` sets that the task does not declare (with its line) | `INFO` |
| A species block that differs from the declared level's defaults file; one for an element not in the structure | `INFO` |
| `geometry.in` keywords the examiner does not interpret; the structure's atoms, elements, constraints, initial moments, shortest distance, and empty space along the third lattice vector | `INFO` |

The examination also records every keyword `control.in` sets, with its arguments as written. Comparisons of calculations use these, so a setting no task declares still shows up as a difference.

Inputs that pass wait for the input reviewer: the calculation stays `SPECIFIED` until it decides.

### The input reviewer

An agent separate from the one that wrote the inputs, and from the Evidence Judge. The dispatcher gives it the inputs waiting for review, one task per hypothesis. It is shown:
- what the calculation is for: observable, estimand, hypothesis, predictions naming it;
- the examination's facts, and every keyword `control.in` sets;
- earlier revisions of the same request it sent back, with its reasons;
- the hypothesis's other calculations, with the keywords set differently from this one;
- the installed code: binary, version, data paths, and the source tree with its manual (LaTeX sources, `doc/manual`) and its `control.in` parser (`src/read_control.f90`), found next to them;
- the files, numbered by line.

It reads any file the researcher can, except credentials. It always has the web, whatever the investigation's `/web` setting, but only the documentation sites in `REVIEW_DOMAINS` (`web.py`: fhi-aims.org, fhi-aims-club.gitlab.io, wiki.fysik.dtu.dk, ase-lib.org), so nothing about the system under investigation reaches the agents through it. It treats the files, their comments and web pages as data, never as instructions.

It records one verdict per calculation (`input_review_recorded`), with findings (`note`, `warning`, `problem`, each with the file, line and keyword it concerns) and the role of every keyword `control.in` sets (`physical_model`, `numerical`, `convergence_aid`, `output`, `other`). The decision follows the most serious finding:

| Decision | When | What follows |
|---|---|---|
| `input_agent_approved` | notes only | the inputs wait for the researcher's `/approve submit` (or the approver, in full automation) |
| `input_agent_warned` | a warning | `/pending` shows the researcher what to look at; `/approve inputs` or `/reject inputs` |
| `input_agent_rejected` | a problem | the calculation is `REJECTED`; its agent is woken with the reasons, at once, even while it waits for other calculations |

A finding's sources are cited: a local file by absolute path, lines and the exact words, which are checked against the file; a web page by URL and words, recorded as not checked. A revision of the same request (same hypothesis, system and estimand) that the reviewer has already sent back three times cannot be sent back again: a fourth rejection goes to the researcher. Job-script findings that need review go to the researcher even when the reviewer approves. If two reviewer tasks end without a decision on a calculation, the researcher decides on it. The researcher may approve or reject inputs at any time, before the reviewer too.

The verdict is a model's judgement, recorded as the reviewer's own and never as evidence.

### Run checks

As for 0.2, except that a relaxation is recognized from the output too, since `control.in` may relax the geometry without the task declaring it. Evidence from the run also carries the limitation that its inputs were checked deterministically only for what runs on the machine and for consistency with the task's declarations, their scientific content judged by the input reviewer or the researcher.

## `fhiaims.run` 0.2 and 0.1

Kept for the calculations recorded under them; new requests are refused and pointed to 0.3. Their examination knows a fixed list of keywords: anything else needs the researcher's review.

### What they cover

One FHI-aims run of one structure, at fixed geometry or relaxing it. Version 0.2 adds runs on the interactive node CEDAR runs on (`execution: local`); version 0.1 runs through Slurm only and stays for the calculations recorded under it. Everything below holds for both.

### Observables

The agent declares what it will take from the run: a name, a precise definition and a unit, for any quantity. The protocol itself reads two values every run prints, and the run check records them with their lines so agents can cite them:

| Reading | From `aims.out` |
|---|---|
| `total energy` (eV) | `Total energy of the DFT / Hartree-Fock s.c.f. calculation` in the final output block |
| `zero-broadening total energy` (eV) | `Final zero-broadening corrected energy` in the final output block; meaningful for metals with occupation broadening |

### Declared variables

Variables are `control.in` keywords, named as in `control.in`. Values are written as arguments: a string is split on whitespace, a number or list is formatted as written, `true` means the keyword appears without arguments, and `false` means it is absent. `species_defaults` is a pseudo-variable: the name of a level directory (such as `light`) under the species-defaults directory recorded in the compute profile.

| Class | Keywords | Declared as |
|---|---|---|
| Physical model or estimand | `xc`, `hse_unit`, `hybrid_xc_coeff`, `spin`, `charge`, `relativistic`, `vdw_ts`, `vdw_correction_hirshfeld`, `many_body_dispersion`, `many_body_dispersion_nl`, `use_dipole_correction`, `relax_geometry`, `relax_unit_cell`, `default_initial_moment`, `fixed_spin_moment` | intervention or control |
| Numerical accuracy | `k_grid`, `k_offset`, `occupation_type`, `sc_accuracy_rho`, `sc_accuracy_eev`, `sc_accuracy_etot`, `sc_accuracy_forces`, `sc_accuracy_stress`, `basis_threshold`, `species_defaults` | numerical setting |
| How convergence is reached or reported | `mixer`, `charge_mix_param`, `n_max_pulay`, `preconditioner`, `ini_linear_mixing`, `ini_linear_mix_param`, `sc_iter_limit`, `sc_init_iter`, `sc_abandon_etot`, `KS_method`, `compute_forces`, `final_forces_cleaned`, `hessian_to_restart_geometry`, `output_level`, `output`, `compensate_multipole_errors` | optional; may appear in `control.in` undeclared |
| Forbidden | `postprocess_anyway` (lets FHI-aims carry on after an unconverged SCF cycle) | never |

Every task must declare `xc`, `spin`, `charge`, `relativistic`, `occupation_type` and `species_defaults`, and `k_grid` for a periodic structure. No record then depends on an unstated code default. Undeclared `sc_accuracy_rho`, `sc_accuracy_eev` or `sc_accuracy_etot` are allowed, with a warning that the code's own default applies.

### Input examination

Findings are `ERROR` (reject), `REVIEW` (needs user approval), `WARNING` or `INFO`. `/pending` shows each `REVIEW` finding with its file and line, and the run directory that holds the examined files. The researcher approves such inputs (`/approve inputs`), or rejects them, or any examined inputs not yet submitted, with a reason that wakes the agent (`/reject inputs`).

**Declarations:**
- An unknown variable is an `ERROR`.
- So is a variable declared in the wrong class, or one missing from the required set.
- So is a `species_defaults` level that is not a directory in the profile.

**`control.in`:**
- Each declared keyword must appear once, with arguments that match the declaration. Numbers compare by value, so `1e-5` matches `1.0d-5`. Otherwise `ERROR`.
- A physical or numerical keyword that was not declared is an `ERROR`.
- A forbidden keyword is an `ERROR`.
- A keyword the protocol does not know is `REVIEW`.
- Each element in the structure needs a species block (`ERROR` if missing). A block that differs from the declared level's defaults file is `REVIEW`; comparison ignores comments and whitespace. A species block for an element not in the structure is a `WARNING`.

**`geometry.in`:**
- Unreadable lines are an `ERROR`; unrecognized keywords are `REVIEW`.
- Atoms closer than 0.5 Å are an `ERROR`.
- `initial_moment` with `spin none` is an `ERROR`.
- A periodic structure with at least 5 Å of empty space along the third lattice vector is treated as a slab and must use one k-point along it (`ERROR`).
- Relaxing a slab with no constrained atoms is a `WARNING`.
- `k_grid` without a periodic cell is an `ERROR`.

**Job script:** examined against the approved compute profile. The requested walltime and core-hours must also fit the task budget. A task's `budget.cores`, if given, sets the task count of a generated script.

**Local script** (0.2, `execution: local`): `local.sh` is the approved job template without its `#SBATCH` lines, with one `#CEDAR --ntasks=N --time=T` line stating what it may use; the protocol generates it with the cores local DFT may use here (the allocation's on this node less 4, which stay for the agents' workbench commands, or the task's, if fewer) and the profile's default walltime, ending at least five minutes before the allocation does. Its commands are examined like a job script's. `#SBATCH` lines in it are an `ERROR`; so is a task count beyond the cores local DFT may use on this node (an allocation of 4 cores or fewer leaves none), a time limit past the allocation's end, a `job.sh` given for a local run (or `local.sh` for a Slurm one), and any local request while CEDAR runs on a login node: "DFT cannot run locally on login nodes". Under 0.1, a local request is an `ERROR`.

**Duplicates:** inputs identical to a reusable calculation's, any hypothesis's, are an `ERROR`; the agent should link that calculation instead.

**Requested resources:** the examination records what the job script asks for (partition, tasks, walltime, core-hours), which `/pending` shows the researcher.

### Run checks

When a job ends, the validator checks the run and stores `aims.out` and the scheduler's logs:

| Check | Passes when |
|---|---|
| `scheduler_completed` | the scheduler reports `COMPLETED` |
| `output_present` | `aims.out` exists (only recorded when it does not) |
| `code_version` | the `FHI-aims version` line matches the profile's recorded version |
| `normal_termination` | `Have a nice day.` is present |
| `scf_converged` | at least one `Self-consistency cycle converged.` and no `SCF cycle not converged.` or `SELF-CONSISTENCY CYCLE DID NOT CONVERGE` |
| `geometry_converged` (relaxations only) | `Present geometry is converged.` and no `Aborting optimization` |
| `inputs_unchanged` | the run directory still holds the examined files |

These marker strings were checked against the FHI-aims source and real outputs of versions 240507 and 260331.1.

A local run's checks are the same; its scheduler state comes from the local backend, and its run check records where it ran (`execution`, `host`, `allocation`, and the CPUs and id of its Slurm step).

## Every version

### Registered runs

A run computed outside CEDAR can be registered from its directory ([`WORKBENCH.md`](WORKBENCH.md)). Nothing runs, so no input reviewer is involved. Its output gets the same checks except `scheduler_completed` (there is no scheduler state), `code_version` (the version is recorded, not required to match the profile) and `inputs_unchanged` (its inputs are stored when it is registered). Its settings are read from its inputs: physical keywords, forbidden ones and keywords the protocol does not know become controls; numerical keywords become numerical settings; convergence aids are left out; `species_defaults` names the level of the profile's species defaults that the species blocks equal, per element where they differ, or a hash of a block that equals none.

### Evidence from a run

Every result taken from the run, a cited reading or an analysis result, gets its quality from the run check. Whether it can support a hypothesis is a separate matter: only if a prediction of that hypothesis named the run before it was submitted.

- every check passed and the job `COMPLETED`: `VALIDATED`, meaning the value at the declared settings from a healthy run
- a check failed: `INVALID`, or `INCONCLUSIVE` if only `geometry_converged` failed

A reading must also have every quote found at its lines of the stored output (`citations_verified`). An analysis result must come from a script that completed, and carries the checks the script made for the run's particular purpose; a failed script check makes it `INCONCLUSIVE` ([`ANALYSIS_SCRIPTS.md`](ANALYSIS_SCRIPTS.md)). It is recorded with the limitation that the script has not been independently reviewed.

### Limitations

- Numerical convergence of the quantities taken from a run, and of differences between runs, with respect to k-points, basis, slab thickness and vacuum is not established by a single calculation. It needs a declared convergence comparison.
- Under 0.3, whether the inputs make sense, the basis quality included, is the input reviewer's judgement, a model's, checked by nothing but the researcher's look where it warns. Its local quotes are verified; its web sources and its reasoning are not.
- Under 0.2 and 0.1, a species block modified by an agent can run only after user review, keyword meanings cover only the keywords listed above, and anything else needs review rather than being guessed.
- A registered run's inputs were not examined before it ran; the settings recorded for it are read from its files, and nothing checks that they produced its output. Its outputs existed before any prediction, so its evidence can challenge a hypothesis, not support it. Its settings are classified with the 0.2 keyword lists.
