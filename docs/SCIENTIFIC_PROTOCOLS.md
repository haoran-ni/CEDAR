# Scientific protocols

A protocol is the second layer between what agents write and what runs. For one calculation type it can:
- generate any input files an agent did not write
- examine the complete set of files that would run
- check a finished run's health, and read the few values every run of its kind prints

It never runs anything and never changes a file it was given. What to take from a healthy run is the agent's decision: it reads values the code printed, citing the lines, or computes them with an analysis script ([`ANALYSIS_SCRIPTS.md`](ANALYSIS_SCRIPTS.md)).

A code has far more input keywords, and far more situations in which they make sense, than any fixed list can cover, and a fixed list raises false alarms on inputs that are fine. So from `fhiaims.run` 0.4 on, and for `vasp.run`, the examination checks deterministically only what runs on the machine: that the files are there, and the job script against the approved template, its resources and the budget. Nothing about the DFT settings is enforced by rules. The **input reviewer**, an agent, checks that the code will run the inputs as the calculation the task describes and that nothing in them is harmful, with the installed code's manual and source and the code's documentation on the web where it is unsure ([below](#the-input-reviewer)); the scientific choices are the agents' and the researcher's. Run checks after the run stay deterministic: they read the code's own success and failure markers.

Everything about a particular HPC system comes from the user-approved compute profile. The agent describes in the task what the calculation is and writes the inputs that make it so.

This document grows with each implemented protocol: `fhiaims.run` (FHI-aims) and `vasp.run` (VASP). A compute profile hosts a protocol when it has an installation of its code (`fhi-aims`, `vasp`).

## `fhiaims.run` 0.4

One FHI-aims run of one structure, at fixed geometry or relaxing it, through Slurm or locally on the interactive node CEDAR runs on. New requests use this version; 0.3, 0.2 and 0.1 stay for the calculations recorded under them, which keep their rules ([below](#fhiaimsrun-03)).

### Inputs and declared settings

The agent writes `control.in` and `geometry.in`, typically by copying the researcher's files and changing what its test changes; only the job script is generated, from the approved template, if the agent does not write it. The agent may also give any other file the run reads, such as a restart file, under any plain file name but the output's (`aims.out`). CEDAR stores such files by hash, writes them into the run directory with the rest and includes them in the calculation's identity, but does not read them: the input reviewer checks them. Every input file is read-only while the run runs and must be unchanged after it (`inputs_unchanged`), so inputs that make FHI-aims rewrite one of them fail the run. The task declares what the calculation is, in whatever terms describe it: what it varies (interventions), physical conditions it holds fixed (controls) and accuracy settings (numerical settings), e.g. `{"dispersion": "MBD-NL"}`, `{"adsorption_site": "top", "n_layers": 4}` or `{"k_grid": [4, 4, 1]}`. The declarations are the agent's description, for the record, the input reviewer and comparisons; they are not checked against the files (`declarations_checked` is false in the examination), and what `control.in` actually sets is recorded from the file.

### Deterministic examination

`ERROR` rejects the inputs at once, back to the agent; `REVIEW` needs the researcher's look, whatever the input reviewer decides; `INFO` findings are facts recorded for the reviewer.

| What | Finding |
|---|---|
| A missing `control.in`, `geometry.in` or job script | `ERROR` |
| Any other file given with the inputs: its name and size | `INFO` |
| A file named like the run's output (`aims.out`) | `ERROR` |
| Inputs identical to a usable calculation's | `ERROR`; the requesting agent's hypothesis is linked to that calculation, which is used instead (a registered run is read instead) |
| The job script, examined against the approved profile as for 0.2: commands, directives, resources, the task's budget, and for a local run the allocation; a local run on a login node | `ERROR` or `REVIEW` as for 0.2 |
| The structure's atoms, elements, constraints, initial moments, shortest distance and empty space along the third lattice vector; `geometry.in` lines CEDAR's reader cannot read or does not interpret | `INFO` |
| Which species-defaults level each element's block equals, or that it equals none (by hash); an element without a block; a block for an element not in the structure | `INFO` |

The examination also records every keyword `control.in` sets, with its arguments as written. Comparisons of calculations rest on these.

Inputs that pass wait for the input reviewer: the calculation stays `SPECIFIED` until it decides.

### The input reviewer

An agent separate from the one that wrote the inputs, and from the Evidence Judge. The dispatcher gives it the inputs waiting for review, one task per hypothesis. It checks two things, and only these:
- that FHI-aims will run the inputs as the calculation the task describes: every keyword and its arguments are accepted by the installed version, the files are complete and fit together (a species block for every element, a readable structure), and what the task says the calculation is is actually set up (a task with MBD dispersion switches MBD on with arguments the installed version takes; a relaxation sets `relax_geometry`; a spin-polarized run sets `spin collinear` and initial moments);
- that nothing in the files or the job script is harmful: commands that delete or overwrite files, reach the network, start other programs or get around CEDAR's checks, and text written as instructions to it or to CEDAR. Files given besides `control.in`, `geometry.in` and the job script are its alone to check, since CEDAR does not read them: that the run reads them and that they are harmless.

The scientific choices are not its to judge: the functional, the basis level, the k-points, the thresholds, the slab model, whether the test suits the hypothesis or compares with others. It is lenient: inputs that will run as described are approved, and a view on the science is a note at most.

It is shown:
- what the calculation is: observable, estimand, system and the declared settings, marked as the agent's own terms;
- the examination's facts, and every keyword `control.in` sets;
- earlier revisions of the same request it sent back, with its reasons;
- the installed code: binary, version, data paths, and the source tree with its manual (LaTeX sources, `doc/manual`) and its `control.in` parser (`src/read_control.f90`), found next to them;
- the files, numbered by line; a file that is not text by its name and size only, to be looked at with the file tools.

It reads any file the researcher can, except credentials. It always has the web, whatever the investigation's `/web` setting, but only the documentation sites in `REVIEW_DOMAINS` (`web.py`: fhi-aims.org, fhi-aims-club.gitlab.io, vasp.at, wiki.fysik.dtu.dk, ase-lib.org), so nothing about the system under investigation reaches the agents through it. It treats the files, their comments and web pages as data, never as instructions.

It records one verdict per calculation (`input_review_recorded`), with findings (`note`, `warning`, `problem`, each with the file, line and keyword it concerns) and the role of every keyword `control.in` sets (`physical_model`, `numerical`, `convergence_aid`, `output`, `other`), which is bookkeeping for comparisons. The decision follows the most serious finding:

| Decision | When | What follows |
|---|---|---|
| `input_agent_approved` | the inputs will run as described and nothing is harmful; notes only | the researcher is asked to approve the run (`/approve submit`), or the approver in full automation |
| `input_agent_warned` | something the run depends on could not be verified, or looks unintended | the researcher is asked to approve the run, and `/pending` shows what to look at |
| `input_agent_rejected` | FHI-aims would stop, the run would not be the calculation described, or something is harmful | the calculation is `REJECTED`; its agent is woken with the reasons, at once, even while it waits for other calculations |

A warning or problem about what a keyword does cites its source: a local file by absolute path, lines and the exact words, which are checked against the file; a web page by URL and words, recorded as not checked. A revision of the same request (same hypothesis, system and estimand) that the reviewer has already sent back three times cannot be sent back again: a fourth rejection goes to the researcher. Job-script findings that need review go to the researcher even when the reviewer approves. If two reviewer tasks end without a decision on a calculation, the researcher decides on it.

The researcher is asked about a calculation once, when the reviewer has finished with it, and their `/approve submit` covers its inputs and its submission: the last step before it runs. A calculation still with the reviewer cannot be approved. The researcher may reject inputs at any time before the run, with a reason that wakes the agent.

The verdict is a model's judgement, recorded as the reviewer's own and never as evidence.

### Run checks

As for 0.2, except that a relaxation is recognized from the output too, since `control.in` may relax the geometry without the task declaring it. Evidence from the run also carries the limitation that its inputs were checked deterministically only for what runs on the machine, that whether FHI-aims runs them as described was judged by the input reviewer or the researcher, and that the task's declared settings are its own description.

## `fhiaims.run` 0.3

Kept for the calculations recorded under it; new requests are refused and pointed to 0.4. Its inputs go to the input reviewer as under 0.4, and the researcher approves them the same way, but its examination also held the files to the task's declarations, which raised false alarms (a keyword named in another case, a flag declared as `true` against `.true.` in the file, a structural fact declared as a setting):
- any `control.in` keyword could be declared, named as in `control.in`, and a declared keyword missing from `control.in`, set with other arguments, or set twice was an `ERROR`; so were a declared value that cannot be written as `control.in` arguments, and a `species_defaults` level not in the profile;
- `postprocess_anyway` was forbidden, unreadable `geometry.in` lines, a structure without atoms and an element without a species block were `ERROR`s;
- a missing `control.in` was generated from the declared keywords and `species_defaults`;
- keywords set but not declared, and species blocks differing from the declared level, were `INFO`.

## `fhiaims.run` 0.2 and 0.1

Kept for the calculations recorded under them; new requests are refused and pointed to 0.4. Their examination knows a fixed list of keywords: anything else needs the researcher's review.

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

Findings are `ERROR` (reject), `REVIEW` (needs user approval), `WARNING` or `INFO`. `/pending` shows each `REVIEW` finding with its file and line, and the run directory that holds the examined files. The researcher approves such inputs with the run (`/approve submit`), or rejects them, or any examined inputs not yet submitted, with a reason that wakes the agent (`/reject inputs`).

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

## `vasp.run` 0.1

One VASP run of one structure, at fixed geometry or relaxing it, through Slurm or locally on the interactive node CEDAR runs on. It works as `fhiaims.run` 0.4 does: the agent writes the inputs, the examination checks what runs on the machine and records facts, the input reviewer checks that VASP will run the inputs as described and that nothing in them is harmful, and the researcher approves the run.

### The installation

The compute profile's `vasp` installation names:
- `binary`: the standard build (`vasp_std`), and `version`: the version as OUTCAR's first line prints it after `vasp.` (e.g. `5.4.4.18Apr17-6-g9f103f2a35`, `6.4.2`);
- `variants`: the other builds of the same version, `gam` (Γ only) and `ncl` (non-collinear and spin-orbit); a job script may launch any of them;
- `data_paths`: `potcars`, the PAW library (e.g. `potpaw_PBE`), one directory per dataset, named as the dataset's `TITEL` names it (`Pb_d`), with that dataset alone in its `POTCAR`; and optionally `source`, VASP's `src` directory, if it is not next to the binaries' directory (`<install>/src` beside `<install>/bin`);
- `job_template`, when VASP's jobs need a setup the profile's template does not have (other modules or environment scripts). A code's own template follows the same rules as the profile's, and its jobs, generated or the agent's, are rendered and examined against it.

### Inputs and declared settings

The agent writes `INCAR`, gives `POSCAR` as the structure (naming its species, VASP 5 format) and writes `KPOINTS`, unless INCAR's `KSPACING` is to set the k-points. It gives `POTCAR` by path, or declares `potcar`, the PAW datasets of POSCAR's species in its order (e.g. `{"potcar": ["Te", "Pb_d"]}`), and CEDAR writes POTCAR from the profile's library, the datasets one after another as they are in the library. The job script is generated from the approved template when the agent does not write one, launching the build the inputs need: `ncl` when INCAR sets `LNONCOLLINEAR` or `LSORBIT`, the standard build otherwise; the `gam` build only in a job script the agent writes. The run's standard output goes to `vasp.out`.

Other files the run reads, such as `WAVECAR` or `CHGCAR` to restart from, may be given under any plain file name but the outputs' (`vasp.out`, `OUTCAR`, `OSZICAR`, `CONTCAR`, `vasprun.xml`), as for `fhiaims.run` 0.4. Every input file is read-only while the run runs and must be unchanged after it: VASP writes `WAVECAR` and `CHGCAR` at the end of a run unless INCAR sets `LWAVE = .FALSE.` and `LCHARG = .FALSE.`. The declared settings are the agent's description, not checked against the files (`potcar` only names what to write).

### Deterministic examination

Besides what runs on the machine, two things about the inputs are rules, because VASP would otherwise stop at once or compute something else without a word: POTCAR must match POSCAR's species, and a non-collinear INCAR must run on the `ncl` build.

| What | Finding |
|---|---|
| A missing `INCAR`, `POSCAR`, `POTCAR` (not given, and no `potcar` declared) or job script | `ERROR` |
| A `potcar` dataset the library does not have (the library's datasets for that element are named) | `ERROR` |
| POTCAR holds a different number of datasets than POSCAR has species, or a dataset for another element than POSCAR names for that species | `ERROR` |
| INCAR sets `LNONCOLLINEAR` or `LSORBIT` and the job script launches another build than the profile's `ncl` | `ERROR` |
| The same, when the profile records no `ncl` build | `WARNING` |
| The `gam` build with a KPOINTS that is not Γ alone | `WARNING` |
| An INCAR tag named nowhere in the installed VASP's source (where the source is found): VASP silently ignores tags it does not read | `WARNING` |
| An INCAR line that sets no tag (VASP ignores it): a heading; a `WARNING` when it starts with a tag VASP reads, as when `=` was left out | `INFO` or `WARNING` |
| `WAVECAR` or `CHGCAR` given while INCAR does not set `LWAVE` or `LCHARG` to `.FALSE.` | `WARNING` |
| Any other file given with the inputs; a file named like an output | `INFO`; `ERROR` |
| The job script, its resources, the budget and, for a local run, the allocation | as for `fhiaims.run` 0.4 |
| The k-point set KPOINTS gives (or that KSPACING sets it); a tag set more than once; the build the job script launches | `INFO` |
| Each POTCAR dataset (`TITEL`, `ZVAL`) and whether it equals the library's; a POSCAR naming no species (VASP takes them from POTCAR) | `INFO` |
| The structure: atoms per species, directions fixed by selective dynamics, shortest distance, empty space along the third lattice vector; POSCAR lines CEDAR's reader cannot read | `INFO` |

The examination records every tag INCAR sets, as written, the k-point set (`KPOINTS`) and the datasets (`POTCAR`); comparisons rest on these, and the input reviewer classifies each. It is shown the installed VASP (its builds, the PAW library and the source, where every INCAR tag it reads appears by name in quotes) and the VASP wiki.

### Run checks

When a job ends, the validator checks the run and stores `vasp.out`, `OUTCAR`, `OSZICAR`, `CONTCAR` and `vasprun.xml` (those that exist) and the scheduler's logs:

| Check | Passes when |
|---|---|
| `scheduler_completed` | the scheduler reports `COMPLETED` |
| `output_present` | `OUTCAR` exists (only recorded when it does not; the first fatal error in `vasp.out` is quoted) |
| `code_version` | OUTCAR's `vasp.` line reports the profile's recorded version |
| `normal_termination` | `General timing and accounting informations for this job` is present (if not, the first fatal error in `vasp.out` is quoted) |
| `scf_converged` | every ionic step's electronic loop ends with `aborting loop because EDIFF is reached`; VASP 5 just goes on after `NELM` steps, VASP 6 says `EDIFF was not reached (unconverged)` |
| `geometry_converged` (when OUTCAR's `IBRION` is 1, 2 or 3 and `NSW` > 0) | `reached required accuracy - stopping structural energy minimisation` |
| `inputs_unchanged` | the run directory still holds the examined files |

The protocol reads `free  energy   TOTEN` and `energy(sigma->0)` of the last ionic step from OUTCAR's ionic-step block, not from the lines of the electronic steps, which are spaced differently. These markers were checked against real output of VASP 5.4.4 and VASP 6's formats. `test_vasp_formats.py` checks them against real runs when `CEDAR_REAL_VASP_RUNS` names a directory of them.

## Every protocol

### Submission

Once the researcher (or, in full automation, the approver) has approved the run, the job manager submits it: to Slurm, or to the local backend. A scheduler that refuses the job (`sbatch` answers with an error) is asked again on the dispatcher's next passes, each refusal recorded (`submission_refused`), since a refusal can be temporary. After three refusals the calculation is `SUBMISSION_FAILED`, with what the scheduler said; a local run that no longer fits (more tasks than local DFT may use here, or a time limit past the allocation's end, which can happen while the inputs wait for review and approval) is refused once and for all. Before each verdict the job is looked up by name, since a refused `sbatch` can still have queued it. The calculation's agent is woken at once with the scheduler's words, to request the calculation again with the cause fixed, usually in the job script; the same DFT inputs may be given again, and the new request is reviewed and approved like any other. An `sbatch` that does not answer leaves the calculation waiting to be submitted; the next pass looks for the job by name before submitting again.

### Registered runs

A run computed outside CEDAR can be registered from its directory ([`WORKBENCH.md`](WORKBENCH.md)); the code is recognized from its inputs. Nothing runs, so no input reviewer is involved. Its output gets the same checks except `scheduler_completed` (there is no scheduler state), `code_version` (the version is recorded, not required to match the profile) and `inputs_unchanged` (its inputs are stored when it is registered). Its settings are read from its inputs:
- FHI-aims: physical keywords, forbidden ones and keywords the protocol does not know become controls; numerical keywords become numerical settings; convergence aids are left out; `species_defaults` names the level of the profile's species defaults that the species blocks equal, per element where they differ, or a hash of a block that equals none.
- VASP: tags that change the physical model or what is computed (e.g. `GGA`, `IVDW`, `ISPIN`, `LDAUU`, `IBRION`), and tags the protocol does not classify, become controls; accuracy tags (e.g. `ENCUT`, `PREC`, `EDIFF`, `ISMEAR`, `SIGMA`) numerical settings; tags that change only how a result is reached, run or written (e.g. `ALGO`, `NCORE`, `LWAVE`) are left out. `KPOINTS` (the k-point set) and `POTCAR` (the datasets, from POTCAR or, when it was not kept, from OUTCAR's `POTCAR:` lines) are numerical settings. Its inputs are recorded under `vasp.run` 0.1, so a request with the same inputs and code version is found identical to it.

### Evidence from a run

Every result taken from the run, a cited reading or an analysis result, gets its quality from the run check. Whether it can support a hypothesis is a separate matter: only if a prediction of that hypothesis named the run before it was submitted.

- every check passed and the job `COMPLETED`: `VALIDATED`, meaning the value at the declared settings from a healthy run
- a check failed: `INVALID`, or `INCONCLUSIVE` if only `geometry_converged` failed

A reading must also have every quote found at its lines of the stored output (`citations_verified`). An analysis result must come from a script that completed, and carries the checks the script made for the run's particular purpose; a failed script check makes it `INCONCLUSIVE` ([`ANALYSIS_SCRIPTS.md`](ANALYSIS_SCRIPTS.md)). It is recorded with the limitation that the script has not been independently reviewed.

### Limitations

- Numerical convergence of the quantities taken from a run, and of differences between runs, with respect to k-points, basis (FHI-aims) or plane-wave cutoff (VASP), slab thickness and vacuum is not established by a single calculation. It needs a declared convergence comparison.
- Under 0.4, whether FHI-aims runs the inputs as the task describes is the input reviewer's judgement, a model's, checked by nothing but the researcher's look before they approve the run. Its local quotes are verified; its web sources and its reasoning are not. Nothing checks the scientific choices in the inputs, the basis quality included: they are the agent's, and the researcher's when they approve. The declared settings are the agent's description; comparisons rest on what `control.in` sets.
- Under `vasp.run` 0.1 the same holds for whether VASP runs the inputs as described. The check for INCAR tags VASP does not read rests on the installed source, when it is found next to the binaries or named in the profile; without it, only the input reviewer looks. POTCAR is matched to POSCAR by element, not by which dataset of the element suits the calculation, which is the agent's choice.
- Under 0.3 the input reviewer also judged whether the inputs made sense for the test.
- Under 0.2 and 0.1, a species block modified by an agent can run only after user review, keyword meanings cover only the keywords listed above, and anything else needs review rather than being guessed.
- A registered run's inputs were not examined before it ran; the settings recorded for it are read from its files, and nothing checks that they produced its output. Its outputs existed before any prediction, so its evidence can challenge a hypothesis, not support it. Its settings are classified with the 0.2 keyword lists.
