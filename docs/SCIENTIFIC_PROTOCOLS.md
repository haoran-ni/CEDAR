# Scientific protocols

A protocol is the second layer between what agents write and what runs. For one calculation type it can:
- generate any input files an agent did not write
- examine the complete set of files against the declared task and its own policy
- check a finished run's health and read the requested observable

It never runs anything and never changes a file it was given.

Protocols contain only knowledge that holds for the code everywhere, such as what each input keyword means and which output lines mark success or failure. Everything about a particular HPC system comes from the user-approved compute profile. Every scientific and numerical setting is declared by the agent in the task.

This document grows with each implemented protocol.

## `fhiaims.total_energy` 0.1

Total energy of one structure from FHI-aims, at fixed geometry or after relaxation.

### Observables

| Observable | Read from `aims.out` |
|---|---|
| `total energy` (eV) | `Total energy of the DFT / Hartree-Fock s.c.f. calculation` in the final output block |
| `zero-broadening total energy` (eV) | `Final zero-broadening corrected energy` in the final output block; meaningful for metals with occupation broadening |

A task must request one of these exactly.

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

Findings are `ERROR` (reject), `REVIEW` (needs user approval), `WARNING` or `INFO`.

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

**Job script:** examined against the approved compute profile (see `DECISIONS.md`). The requested walltime and core-hours must also fit the task budget.

**Duplicates:** inputs identical to a reusable calculation's are an `ERROR`; the agent should link that calculation instead.

### Run checks

| Check | Passes when |
|---|---|
| `scheduler_completed` | the scheduler reports `COMPLETED` |
| `output_present` | `aims.out` exists (only recorded when it does not) |
| `code_version` | the `FHI-aims version` line matches the profile's recorded version |
| `normal_termination` | `Have a nice day.` is present |
| `scf_converged` | at least one `Self-consistency cycle converged.` and no `SCF cycle not converged.` or `SELF-CONSISTENCY CYCLE DID NOT CONVERGE` |
| `geometry_converged` (relaxations only) | `Present geometry is converged.` and no `Aborting optimization` |
| `inputs_unchanged` | the run directory still holds the examined files |
| `citations_verified` | every cited quote was found at its lines in the stored output |
| `agrees_with_protocol_reading` | the agent's value equals the protocol's own reading, when both exist |

These marker strings were checked against the FHI-aims source and real outputs of versions 240507 and 260331.1.

A failed health check makes the evidence `INVALID`. If only `geometry_converged` failed, it is `INCONCLUSIVE` instead. If the agent and the protocol read different values, or nobody read a value, the evidence is `INCONCLUSIVE`. Otherwise it is `VALIDATED`, meaning a converged total energy at the declared settings.

### Limitations

- Numerical convergence of energy differences with respect to k-points, basis, slab thickness and vacuum is not established by a single calculation. It needs a declared convergence comparison.
- A species block modified by an agent can run only after user review. The examiner does not judge basis quality.
- Keyword meanings cover the keywords listed above. Anything else needs review rather than being guessed.
