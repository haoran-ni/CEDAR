# Scientific protocols

A protocol is the second layer between what agents write and what runs. For one calculation type it can:
- generate any input files an agent did not write
- examine the complete set of files against the declared task and its own policy
- check a finished run's health, and read the few values every run of its kind prints

It never runs anything and never changes a file it was given. What to take from a healthy run is the agent's decision: it reads values the code printed, citing the lines, or computes them with an analysis script ([`ANALYSIS_SCRIPTS.md`](ANALYSIS_SCRIPTS.md)).

Protocols contain only knowledge that holds for the code everywhere, such as what each input keyword means and which output lines mark success or failure. Everything about a particular HPC system comes from the user-approved compute profile. Every scientific and numerical setting is declared by the agent in the task.

This document grows with each implemented protocol.

## `fhiaims.run` 0.2 (and 0.1)

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

A local run's checks are the same; its scheduler state comes from the local backend, and its run check records where it ran (`execution`, `host`, `allocation`).

### Registered runs

A run computed outside CEDAR can be registered from its directory ([`WORKBENCH.md`](WORKBENCH.md)). Its output gets the same checks except `scheduler_completed` (there is no scheduler state), `code_version` (the version is recorded, not required to match the profile) and `inputs_unchanged` (its inputs are stored when it is registered). Its settings are read from its inputs: physical keywords, forbidden ones and keywords the protocol does not know become controls; numerical keywords become numerical settings; convergence aids are left out; `species_defaults` names the level of the profile's species defaults that the species blocks equal, per element where they differ, or a hash of a block that equals none.

### Evidence from a run

Every result taken from the run, a cited reading or an analysis result, gets its quality from the run check. Whether it can support a hypothesis is a separate matter: only if a prediction of that hypothesis named the run before it was submitted.

- every check passed and the job `COMPLETED`: `VALIDATED`, meaning the value at the declared settings from a healthy run
- a check failed: `INVALID`, or `INCONCLUSIVE` if only `geometry_converged` failed

A reading must also have every quote found at its lines of the stored output (`citations_verified`). An analysis result must come from a script that completed, and carries the checks the script made for the run's particular purpose; a failed script check makes it `INCONCLUSIVE` ([`ANALYSIS_SCRIPTS.md`](ANALYSIS_SCRIPTS.md)). It is recorded with the limitation that the script has not been independently reviewed.

### Limitations

- Numerical convergence of the quantities taken from a run, and of differences between runs, with respect to k-points, basis, slab thickness and vacuum is not established by a single calculation. It needs a declared convergence comparison.
- A species block modified by an agent can run only after user review. The examiner does not judge basis quality.
- A registered run's inputs were not examined before it ran; the settings recorded for it are read from its files, and nothing checks that they produced its output. Its outputs existed before any prediction, so its evidence can challenge a hypothesis, not support it.
- Keyword meanings cover the keywords listed above. Anything else needs review rather than being guessed.
