# AGENTS.md — Sim2Real Scientist

> **Status:** Design and engineering specification, not a claim of completed functionality.
> **Audience:** Human contributors and coding agents implementing this repository.
> **Mission:** Build an interactive, auditable, hypothesis-driven scientific agent swarm that explains and constrains discrepancies between simulations and experimental observations, validates scientific claims using dependable tools, and recommends the next most informative action.

This file is the **primary repository-wide implementation contract**. Read the supporting documents before changing the architecture, data contracts, scientific workflows, or demonstration methodology:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): components, lifecycle, concurrency, orchestration, and TUI.
- [`docs/SCIENTIFIC_PROTOCOLS.md`](docs/SCIENTIFIC_PROTOCOLS.md): observable alignment, DFT best practices, numerical tests, uncertainty, and scientific validity.
- [`docs/EVIDENCE_AND_PROVENANCE.md`](docs/EVIDENCE_AND_PROVENANCE.md): durable data structures, event sourcing, human-readable logs, and traceability.
- [`docs/ANALYSIS_SCRIPTS.md`](docs/ANALYSIS_SCRIPTS.md): analysis scripts agents write: how they run, their guards, correctness requirements, and planned independent review.
- [`docs/DEMO_AND_EVALUATION.md`](docs/DEMO_AND_EVALUATION.md): initial cases, offline blind evaluation, baselines, and presentation requirements.
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md): vertical-slice milestones and acceptance criteria.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): settled requirements versus provisional choices.

## 1. Non-negotiable project goals

1. **Investigate simulation–experiment discrepancies.** Given simulation outputs, experimental measurements, uncertainties, sample/measurement conditions, and a research question, identify the *actual quantities being compared*. Diagnose definition mismatches before attributing residual differences to physics.
2. **Generate multiple causal explanations.** Construct a set of competing or compatible, physically motivated, *falsifiable* hypotheses. Do not assume that one explanation must win or that additive contributions are identifiable.
3. **Instantiate a dynamic hypothesis swarm.** Allocate investigation agents by hypothesis; their count, specialization, tools, and lifetimes depend on the question and budget. No predefined XC/temperature/coverage swarm is mandatory. The Chief Scientist decides which agents to create, at most one per hypothesis, and the role each plays; the researcher sees the proposal and can reply before the agents start, and if they do not reply within the review window (10 minutes) the agents start.
4. **Run independent, parallel investigations.** Agents propose predictions and tests, gather evidence, challenge their own claims, and return structured reports. They work at the same time and cannot see each other's work. The Chief Scientist collects and analyzes their results; deliberate cross-examination goes through it.
5. **Examine everything agents write before it runs.** Agents may write DFT inputs and job scripts themselves, or ask protocol tools to generate them. Nothing runs unexamined: deterministic protocol tools check every input file and job script against the declared task and the protocol's policy before submission, and anything the tools cannot verify requires explicit user approval. Agents may also read raw output files directly to interpret results, and write their own analysis scripts to compute any quantity from them. A value printed in an output must cite the file and lines it came from; a computed value must come from an analysis script that the system ran on hashed copies of the outputs, on the login node under guards (it can never start programs, so never DFT), and recorded with its inputs. Either becomes validated evidence only after protocol tools have checked the run's health and verified the citation or recorded the analysis. Analysis scripts must be correct and safe; independent review of them by other agents is a planned second layer.
6. **Support durable asynchronous computation.** Hypothesis agents should relinquish their runtime after submitting long-running jobs. An independent monitor and event dispatcher resume work when jobs finish; an open chat or terminal is NOT required for jobs to progress.
7. **Keep a continuously accessible Chief Scientist conversation.** The Chief Scientist plans, supervises, explains, accepts real-time user instructions, and coordinates research. It does not personally execute DFT, phonons, MLIP, shell commands, or other scientific calculations.
8. **Select next actions intelligently.** Recommend or launch permitted computations/measurements that best discriminate among hypotheses, respecting information quality, resource cost, feasibility, and authorization; do not fabricate information-gain scores.
9. **Preserve scientific reasoning traces.** Record concise, human-readable scientific justifications, predictions, actions, observations, uncertainties, interventions, and revisions at every iteration. Save references to the actual underlying evidence. Do not request or expose a model's private/internal chain of thought.
10. **Provide a terminal-native interface.** Use a fixed/live swarm and HPC-job status panel above a scrollable Chief Scientist conversation and a fixed input area. Do not block the conversation while other agents or HPC tasks run.
11. **Show a credible end-to-end example.** Demonstrate real scientific evidence and a completed inspectable workflow—not simply a chat simulation, literature summary, or submitted job awaiting completion.

## 2. Scientific scope and problem representation

Initial scope: atomistic materials, molecules on surfaces, organic/metal-oxide and organic/metal interfaces, adsorption geometry/energetics, electronic states, work functions, relevant spectra, structural and thermodynamic discrepancies. Possible extensions include catalysis, phases, defects, dynamics, and other scientific domains. Do not bake one material, one observable, or one correction into the core architecture.

A scientific investigation must express:

- **Scientific question and target claim:** what is allegedly inconsistent, and what claim would resolution support?
- **Simulation provenance:** structure IDs, charge/spin, coverage, boundary conditions, electronic-structure method, XC, dispersion, basis, k mesh, numerical tolerances, sample conditions, code and version.
- **Experiment provenance:** measured observable (not simply a label like “gap”), technique, units, calibration/reference, uncertainty, temperature, pressure, preparation history, composition, coverage, environment, accessible raw data and citations.
- **Observable alignment:** an explicit forward/measurement model, if needed, that maps simulated quantities to experimental signals; a Kohn–Sham gap is not directly an optical gap, atom-projected DOS is not UPS intensity, a static 0 K geometry is not a finite-T structural distribution.
- **Discrepancy:** numerical size *only after* compatible definitions, references, units and conditions are established; otherwise label `NOT_COMPARABLE` and plan a comparison protocol.
- **Constraints:** available tools, validated protocols, accessible clusters, computation budgets, deadlines, external retrieval policy, approval rules, and permitted data.

One useful conceptual model (not a universally identifiable decomposition) is

\[
y_{\mathrm{exp}} = \mathcal M[f_{\mathrm{sim}}(x,\theta),\phi] + \delta(x) + \epsilon\ ,\]

where \(\mathcal M\) maps a model result to a measured observable, \(\theta\) describes model/physical variables, \(\phi\) describes measurement settings, \(\delta\) is model discrepancy, and \(\epsilon\) is observational noise. These terms may be correlated or non-identifiable. Never present a clean percentage breakdown without defensible data and assumptions.

## 3. Agent roles and boundaries

### 3.1 Chief Scientist (top-level supervisor, **not** a compute agent)

Responsibilities:

- Hold a natural, real-time conversation with the researcher at any point in the investigation.
- Inspect research state and explain which hypotheses have been generated, which jobs are pending/running/completed/failed, which evidence has passed validation, and what remains unresolved.
- Create/revise the high-level research plan; coordinate hypothesis generation, screening, priority, dependency graphs, budgets, checkpoints, approvals, and phase transitions.
- Delegate to specialist agents or orchestration tools. Propose and handle new, paused, combined, discarded, or revived hypotheses in response to evidence or user direction.
- Propose the hypothesis agents and their roles for the researcher's review; collect, compare and analyze what they report.
- Consult Evidence Judge for scientific synthesis and Research Recorder for traceable history.
- Accept user intervention without resetting unrelated work, e.g., “pause new coverage tests, prioritize XC” while active calculations remain unless explicitly cancelled.

Hard constraints: no direct DFT execution or shell execution, no job submission that bypasses input examination, no fabricated job status, no unilateral promotion of a user's preference to scientific evidence, and no closing of an investigation: the Chief Scientist proposes closing, with its reasons, and only the researcher closes it. Chief Scientist can **request** computation (and may draft inputs) through task orchestration, but only the job manager executes it, and only after examination. A continuously accessible chat **does not imply** a continuously running LLM inference process.

### 3.2 Hypothesis generation / review

Before allocating agents:

1. Align observables and check baseline evidence/conditions.
2. Produce distinct candidate mechanisms with physical rationale.
3. For each, specify predicted discriminating observations, possible falsifiers, known dependencies, required data/protocols, and costs.
4. Merge duplicates; recognize interacting mechanisms; prioritize feasible and informative tasks.
5. Instantiate as many hypothesis-specific agents as warranted—not a fixed roster: at most one agent per hypothesis, and not necessarily one for a low-value hypothesis. The Chief Scientist proposes them; the researcher reviews the proposal, and any reply holds it until the Chief Scientist revises it.

Specializations (XC, thermal, coverage, surface-defect, numerics, kinetics, quasiparticle, excitonic, vibrational, spectroscopy, etc.) are **examples**, not permanently resident workers. A generic typed `HypothesisAgent` may be configured dynamically with a hypothesis, scope, allowed tools, budget, and constraints. New interactions may merit additional agents after evidence appears.

### 3.3 Hypothesis Agent contract

For each investigation cycle:

1. **Formulate:** state one precise, falsifiable hypothesis with rationale and assumptions.
2. **Predict:** commit expected observations and conditions **before** seeing test results (immutable preregistration).
3. **Plan:** choose a discriminating, safe, scientifically comparable test, including controls and cost.
4. **Execute or delegate:** read data and raw outputs directly, write and run analysis scripts, and request computation, either by writing the DFT inputs and job script itself or by asking protocol tools to generate them. Everything goes through input examination before submission; the agent never submits jobs or runs commands itself, and its analysis scripts run only through the guarded analysis runner. Long tasks return identifiers rather than blocking.
5. **Interpret:** read the raw outputs that came back, citing file and line ranges for every value read from them, and compute other quantities with recorded analysis scripts, so every number can be traced and checked.
6. **Challenge:** identify counterevidence, alternative hypotheses, confounders and failure of method applicability.
7. **Report:** emit a typed, evidence-linked report with uncertainty, conclusion status, limitations, and next recommendations.

Agents investigate independently and cannot see each other's work. The Chief Scientist collects their reports and compares them across hypotheses; it exposes confounders, contradictions, or mechanisms that interact, and passes on to an agent another's finding when its test needs it. A structured critique round, if added, also goes through the Chief Scientist. All useful updates persist. Agents must not simply repeat their prior textual conclusions as evidence.

### 3.4 Evidence Judge / synthesis service

The Judge:

- Compares *compatible, validated* evidence rather than collecting agent votes.
- Distinguishes direct computational observations, inference under assumptions, published experimental claims, indirect constraints, missing evidence, and model limits.
- Tests common conditions, numerical comparability, independence, references, uncertainty and confounding; prevents mixing incompatible geometries/methods/coverages/observables as though controlled.
- Allows **several causes to coexist**; creates interaction follow-ups where justified.
- Separates `SUPPORTED`, `CHALLENGED`, `INCONCLUSIVE`, `NOT_TESTED`, and `NOT_COMPARABLE` conclusions from the **quality of each computational evidence object**.
- Reports residual discrepancies and estimates only what is actually identifiable. No fabricated mechanism probabilities, percentages, uncertainty bars, or literature citations.

### 3.5 Next-best-action planner

Evaluate candidate next calculations/measurements by expected discriminatory power/information gain, scientific relevance, resource cost, likelihood of completion, and available protocols. Bayesian experimental design or decision-theoretic methods may be used **only if predictive outcome distributions and priors are actually defined**. Otherwise label ranking as a qualitative, justified priority rather than numeric expected information gain. The system can suggest an experimental measurement without pretending to execute a laboratory experiment.

## 4. Verified Scientific Computing Engine

Agents may write calculation inputs and job scripts, but a **deterministic, versioned protocol library** examines them before any production calculation runs. Every request declares its intent in a scientific task specification (observable, system, variables, comparisons, fidelity/accuracy goal, budget) and either attaches agent-written files or asks the protocol's generator to write them. The examination checks that the files actually implement the declared intent and follow the protocol's policy; generated files go through the same examination.

Minimum workflow:

1. **Protocol resolution:** ensure a supported protocol exists and inputs, physical assumptions and execution rights are adequate; unsupported methods return a structured rejection or research gap.
2. **Input authoring:** the agent writes the code inputs and job script, or the protocol generates them from tested templates/policies. Generators use structure-aware sampling/convergence settings rather than universal fixed integers.
3. **Input examination (second layer):** deterministic tools parse the files and check them against the declared task and the protocol's policy: structures, geometry, stoichiometry, cell/PBC, vacuum, charge, spin, magnetism, occupation/smearing, dispersion, XC, basis, k sampling, units, energy/reference definitions, supported code/version, control variables, resource allocations, and the job script's directives and commands. The verdict is `APPROVED`, `REJECTED` (contradicts the declared task or violates policy), or `NEEDS_REVIEW` (content the tools cannot verify, such as an unknown keyword or shell line), which needs explicit user approval before submission.
4. **Execution:** submit only examined files, through a constrained job manager, to Slurm; persist calculation identity, input hashes, manifest, job ID and task linkage atomically as far as the backend permits.
5. **Runtime checks and controlled recovery:** inspect scheduler state and scientific code outputs separately. Retry only known recoverable failures under specified policy, and always record changes; changes that alter the scientific question/observable require fresh approval or an appropriately defined comparison.
6. **Result reading and checks:** protocol tools check run health (normal termination, SCF and geometry convergence) and store the outputs by hash. Agents then read raw outputs directly, and the tools verify that every value an agent quotes appears at the cited lines of the hashed output; or agents compute quantities with analysis scripts that run on the login node under guards and are recorded with their inputs. Target property convergence, reference consistency and physically meaningful result selection are checked by declared comparisons.
7. **Scientific validation:** mark as `VALIDATED`, `PROVISIONAL`, `INCONCLUSIVE`, or `INVALID` against explicit protocol conditions; output a structured Evidence Object with limitations.

### 4.1 Scientific-variable isolation

Classify inputs as (a) **intervention/scientific variables** deliberately changed (e.g., XC, coverage, defects, temperature), (b) **numerical settings** controlled to target a defensible accuracy (e.g., k sampling, basis, tolerance, slab/vacuum convergence), and (c) **physical conditions/controls** (e.g., termination, spin, charge, environment). "Same numerical accuracy" does not always mean "same integer settings": a different surface supercell can require a different k mesh. Every comparative task must explicitly state its *estimand* and controls.

Example XC investigation:

- Fixed-geometry comparisons isolate electronic-energy differences at common structures.
- Independently relaxed comparisons include both functional-dependent electronic energies and structural relaxation.
- Never silently mix those two estimands in a causal attribution.
- For two configurations with **identical composition and supercell**, relative total energies can be compared directly; do not add unnecessary reference species.

### 4.2 Numerical convergence and theory limits

Prioritize convergence of the **target difference/observable** rather than only total energy or SCF thresholds. Test relevant k meshes, basis, slab thickness, vacuum, spin solutions and other conditions; convergence is an empirical test, not a promise from a `tight` preset. Scoping/accuracy tiers (`screening`, `research`, `reference`) specify evidence expectations, not magic parameter bundles. Theories with different approximations may disagree *after* all numerical gates pass; do not label theory error as a convergence problem.

### 4.3 Portability: no site-specific settings in code

Sim2Real must run on any HPC system. Nothing about a particular cluster belongs in the code: code binaries and versions, species or pseudopotential locations, environment setup, launch commands, schedulers, partitions, resource limits, or "usual" numerical settings taken from one group's past runs.

- An **environment-setup agent** (the Chief Scientist or a dedicated subagent) works out how to run each code on the current system, by reading local files (existing job scripts, inputs, module files, installation directories) or by asking the user. It proposes a **compute profile**: the code installations, job-script template, approved environment-setup lines, scheduler type, allowed partitions and limits. Each value records where it came from (file and content hash, or the user's answer).
- Deterministic checks verify the proposed profile (paths exist, the binary is executable, the template is well formed). The profile takes effect only after the user approves it, and changing it later needs another approval.
- Generated job scripts are rendered from the approved profile's template. Agent-written job scripts are examined against the approved profile, not against fixed patterns in code.
- Scheduler access goes through adapters; Slurm is implemented first.
- Protocols encode only code-level knowledge that holds everywhere, such as what each input keyword means and which output lines signal success or failure. Scientific and numerical settings are declared explicitly in each task by the agent, which may learn typical values from the user's previous inputs or ask the user.

### 4.4 FHI-aims implementation notes

Initial scientific backend: **FHI-aims**, with Slurm as the first scheduler adapter (ASE where useful). Future adapters can add VASP, Quantum ESPRESSO, phonon codes, MD and MLIPs. Existing ecosystems (pymatgen, atomate2, custodian, AiiDA) are useful references/components where compatible, but do not prematurely combine all of them into a two-day prototype.

- FHI-aims species defaults (e.g., `light`, `intermediate`, `tight`, `really_tight`) are only starting points; verify the protocol with target-specific convergence tests and current manual/version documentation.
- FHI-aims has atom-centered DFT+U facilities. **Do not equate this with the molecular-orbital DFT+U used in CO/metal adsorption-site literature**; placing a Hubbard U on Cu 3d does *not* reproduce a correction targeted at CO's frontier molecular orbitals. Verify syntax and force/gradient support against the installed FHI-aims build before implementing.
- Standard MLIPs (including UMA tasks) cannot by themselves provide reliable UPS spectra, electronic band gaps, or work functions; pair structural sampling with an appropriate electronic/measurement model and validate transferability to the tested surface chemistry.
- Slurm `COMPLETED` does **not** mean SCF/relaxation or target-property convergence; never conflate scheduler success with scientific success.

More detail: [`docs/SCIENTIFIC_PROTOCOLS.md`](docs/SCIENTIFIC_PROTOCOLS.md).

## 5. Asynchronous Slurm and durable agent execution

Implement **event-driven fan-out/fan-in**:

- Each Hypothesis Agent requests a scientific task and receives a durable `calculation_id` / `job_id` and a `WAITING_FOR_JOB` state. It checkpoints and **ends its current LLM runtime**; do not consume tokens while idle.
- A separate Job Monitor queries the scheduler through its adapter (for Slurm: `squeue`/`sacct` or approved APIs) and reacts to changes. Treat unknown/unavailable scheduler information as unknown, not completed. Respect cluster polling and login-node policies.
- Result validation happens *before* job output becomes trusted scientific evidence; failures create explicit terminal or recoverable failure events.
- An Event Dispatcher links events to awaiting investigations, restores checkpointed state, and invokes only the appropriate agents/nodes. The Chief Scientist conversation stays available throughout.
- Workers and simulations run independently; fast hypotheses can finish while other DFT tasks remain queued/running. Judge may publish provisional synthesis before final evidence; require clear completeness status.
- Persist job/task state independently of the terminal UI. Closing and reopening the TUI must not interrupt active jobs.
- Support `QUEUED`, `RUNNING`, `WAITING_FOR_JOB`, `ANALYZING`, `COMPLETED`, `FAILED`, `PAUSED`, `CANCELLED`, plus explicit error/retry statuses. Distinguish agent/hypothesis state from Slurm job state and scientific validation state.
- Use idempotent task identifiers and duplicate-submission safeguards, event deduplication, optimistic concurrency or transactional state updates, timeouts, retry limits, approval checks and cancellation semantics. Preserve outputs from jobs launched under an earlier research plan.

**Checkpoint caution:** In LangGraph, resuming an interrupted node can re-execute code before the interruption point. Prefer separating submit/wait/analyze nodes and idempotent submission; never place an unguarded `sbatch` before an interruption that could replay. Persist model-independent state, not Python stack frames.

Detailed lifecycle and state transitions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 6. Chief Scientist interaction, human authority, and live UI

The researcher may at any time ask for status, rationale, alternate tests, budget, evidence provenance, changes in priority, pauses, or additional hypotheses. Requests update a versioned **Research Plan**; concurrent workers see relevant version changes at safe checkpoints. Never silently cancel a running HPC job because a priority changed. Confirm or apply the configured authorization rule before disruptive or expensive actions. User preferences influence plan priorities **but do not alter the evidence**, retroactively rewrite earlier predictions, or bypass validation gates.

### Terminal UI required behavior

A terminal-native, Claude Code–style console with:

1. **Top: pinned/live Agent Swarm dashboard** with each agent's hypothesis, status, task, and associated Slurm `PENDING`/`RUNNING`/`COMPLETED`/`FAILED` statuses. Clearly separate “job finished” from “hypothesis verified”. Indicate task ages/progress if known, never invent percentages.
2. **Middle: scrollable Chief Scientist chat**, readable Markdown, concise messages and optional collapsible activity summaries; status refresh must not flood the chat history.
3. **Bottom: pinned input composer**, accepting normal conversational prompts and explicit control commands; input remains responsive while background jobs continue.
4. **Expandable details** on agents/jobs/evidence, research timeline and reports; link/open associated filesystem artifacts.
5. **Reconnect support**: restarting client reconnects to the persistent state and live event updates, without restarting scientific investigations.
6. **Accessible fallback** for terminals without rich capabilities; reasonable keyboard navigation and small terminal sizes.

Recommended initial UI framework: Python **Textual**; possible `DataTable`, `RichLog`, `Markdown`, `TextArea`, command actions. Backend communication through events or simple polling for MVP. UI is *not* the orchestrator or sole store of truth. Optional commands: `/status`, `/agents`, `/jobs`, `/report`, `/pause`, `/resume` (require specification for exact semantics). Do not claim these commands exist until implemented.

## 7. Evidence, uncertainty, and epistemic constraints

The system should classify proposed mechanisms as:

- **Directly quantifiable** with a supported theoretical/protocol method.
- **Approximately quantifiable** with known approximation range and uncertainties.
- **Constrained indirectly** through external measurements/parameters rather than direct simulation.
- **Currently non-identifiable or unsupported**, requiring additional evidence.

**Finite temperature is not a generic number to add to 0 K DFT.** Different observables require different finite-T models (e.g., QHA/vibrational free energies for thermodynamic stability, electron–phonon and thermal-expansion treatment for electronic levels, MD/snapshot electronic calculations for structural fluctuation effects on work function). Clearly separate configuration averaging, electronic thermal occupations, anharmonicity, preparation kinetics, and actual measurement response. Do not claim that averaging snapshot work functions necessarily reproduces a measured UPS signal.

If computing a mechanism's contribution, specify reference state, method, controlled variables and estimator. Coupled mechanisms may require interaction terms or joint models. Statistical sampling noise, numerical truncation, model bias and experimental uncertainty are distinct and potentially correlated. Propagate covariance where known; do not assign uncalibrated LLM “confidence” as a probability. If only one theory value and one measured value are available, a detailed percentage split is usually not identifiable.

Evidence must link to raw code outputs or a verifiable external source, plus method/condition information, not merely an LLM assertion. A value an agent read from an output cites the file and lines it came from. Source credibility and access are part of provenance; blind/offline evaluation may intentionally withhold external literature, and this must be enforced through tool permissions and data access.

## 8. Scientific Research Recorder and human-readable documentation

Implement **two synchronized layers**:

- **Durable machine-readable source of truth:** append-only or append-oriented event stream and typed objects in SQLite/PostgreSQL/JSONL, covering hypothesis creation, predictions, decisions, tool intents, approvals, protocol choices, submissions, scheduler updates, validation, evidence, critiques, plan changes, and user interventions. Events have IDs, timestamps, actor, investigation, round, plan version, and references.
- **Human-readable research notebook:** automatically build compact Markdown research summaries per investigation, per hypothesis and per iteration. Each round answers: *What did we hypothesize? Why? What did we predict? What did we test? What did we observe? What changed our assessment? What remains unknown? What is next?*

Document significant decisions and their *stated scientific justification*, not hidden internal model reasoning. Preregister predictions before consuming outcomes; make them immutable. Attach Evidence IDs and calculation manifests to every important quantitative claim. Present unverified hypotheses as hypotheses, not facts. Record exact human interventions and changes to research plan priorities separately from empirical evidence. Generate exportable investigation reports suitable for later presentation or scientific writing, with disclosed limitations; do not silently alter raw logs.

See [`docs/EVIDENCE_AND_PROVENANCE.md`](docs/EVIDENCE_AND_PROVENANCE.md) and [`templates/ROUND_REPORT.md`](templates/ROUND_REPORT.md).

## 9. Evaluation and reproducibility requirements

### Primary demonstration (proposed)

**CO/Cu(111) low-coverage adsorption-site preference.** A classic discrepancy involves semilocal DFT predictions that may favor hollow and experimental observations favoring atop. Use actual FHI-aims results rather than assuming a specific method/geometry produces a desired outcome. Track coverage, slab structure, numerical convergence, experiment preparation, and thermodynamic conditions. Distinguish atom-centered FHI-aims DFT+U from the published molecular-orbital CO correction; do not claim direct reproduction. The purpose is evaluating a **scientific process**, not merely reciting the known literature explanation.

### Secondary demonstration (proposed)

**TCNE/Au interface structure:** potentially planar predictions versus experimentally reported tilted/upright ordered arrangements. Check the precise surface, phase, coverage, preparation and reported observations from source materials; experiment need not show the global equilibrium minimum of a pristine ideal slab. Treat it as an open investigation or precomputed-data case if new DFT runs exceed the demonstration window. Do not claim a previously published hypothesis (e.g., adatoms) is a novel discovery.

Other reference cases: CO/Pt(111), CO/Rh(111), O2/Ag(111), benzene/Cu(111), vibrational/spectroscopic mismatches. These require per-case dataset checks before use, not canned claims of precise results.

### Scientific blind tests

- Compare **retrieval disabled** versus normal evidence-assisted modes honestly. A cloud LLM API without web browsing is *not physically air-gapped*; offline evidence restrictions do **not** eliminate pretraining contamination.
- Hide reference solutions and selected simulation results from the agents. Require timestamped preregistered hypotheses, predictions, chosen discriminatory tests, and method constraints before revealing withheld evidence.
- Add counterfactual/control cases (e.g., deliberately under-converged numerics, changed coverage, other substrates), clearly marked as synthetic when synthetic. An agent must not mechanically blame XC for every CO adsorption anomaly.
- Benchmark against a vanilla LLM and a single tool-using agent **with matched data/tool access and comparable resource budgets**. Evaluate valid hypotheses, evidence quality, number of unsupported claims, result-update correctness, test-selection quality, runtime/cost, and compliance with protocols.
- Never prepopulate fabricated results, status indicators, confidence scores, novelty claims, live job IDs, or benchmark metrics.

For a short presentation, show a genuine discrepancy, live hypotheses, actual computations or independent withheld-data analyses, validated outcomes, evidence-based revisions, next-best-action, and the auditable timeline. A submitted Slurm job alone is not an adequate scientific result.

## 10. Suggested technology stack (choices, not obligations)

- **Language/runtime:** Python; typed Pydantic-style contracts.
- **Orchestration:** LangGraph for durable, dynamic, iterative workflows; simple asyncio and/or an Agents SDK can bootstrap a narrow prototype. Avoid multiple competing orchestration frameworks without need.
- **Compute:** ASE, FHI-aims first, optionally VASP/QE, Phonopy, suitable MLIPs (e.g., UMA only where validated).
- **HPC:** scheduler adapters (Slurm first), a submission manager and an independently running monitor; site details come from the user-approved compute profile, never from code; respect individual cluster policies.
- **Storage:** SQLite for prototype; database abstraction with PostgreSQL/queue upgrades for scale; artifact filesystem with content hashes and immutable provenance.
- **Statistics:** NumPy/SciPy, and probabilistic libraries only when an actual statistical model is justified.
- **TUI:** Textual with fixed upper dashboard, conversation, and input.
- **References:** tools such as AiiDA, atomate2, pymatgen and custodian offer tested ideas for workflows/provenance/handlers, but first implementation should be intentionally narrow.

Architectural decisions are provisional where marked in [`docs/DECISIONS.md`](docs/DECISIONS.md). Choose tested APIs according to actual installed versions; no dependency version or code-specific feature should be invented.

## 11. Development instructions for coding agents

- **Build a vertical slice, not mock scientific theater.** Start from one well-scoped adsorption problem, typed hypothesis/test objects, a single working protocol, real Slurm or a clearly labeled test-double, a parser/validator, checkpoint/resume, and a round summary. Broaden only after the path works end to end.
- **Never execute anything unexamined.** Agents may write inputs and job scripts, but only files that passed input examination (or that the user explicitly approved after a `NEEDS_REVIEW` verdict) may reach HPC, and they run through the job manager, never through an agent's own shell. Agent-written analysis scripts are examined and run by the analysis runner on the login node, under guards that keep them to their staged inputs and stop them from starting programs, so they can never run DFT. Validate arguments, forbid path traversal, cap budgets, use least privileges and preserve cluster/user data.
- **Never silently change a scientific parameter** while recovering a computation. Distinguish numerical recovery from changing the physical model. Store exact versions and diffs.
- **Never identify scheduler `COMPLETED` as validated scientific evidence** before parsing/checks.
- **Do not let conversational context serve as sole state.** All important status/evidence must survive agent shutdown, process restart, UI exit, and research-plan revision.
- **Avoid duplicate calculations.** Use canonical task identity and hashes, idempotent submission/consumer operations, explicit cache reuse and concurrency-safe writes.
- **Honor user intervention** with explicit authorization for cancellation/spending, preserve running job IDs and prior evidence, and record who changed the plan and why.
- **Distinguish facts, inferences, recommendations, and unknowns** in UI and generated reports. Label citations, conditions, and uncertainties. No invented scientific numerical results.
- **Do not confuse public reasoning summaries with hidden chain of thought.** Persist concise scientific decision records, observation-based rationales and validation steps, not model-private deliberation.
- **Do not add fake installation or usage claims to README** until working commands have been exercised and tested. Do not claim novelty, awards, final publication, or validated scientific performance without supporting evidence.
- **Provide tests:** protocol schema and invalid-input cases; input examination of agent-written files (approved, rejected, needs review); citation checks for agent readings of outputs; intended-control vs uncontrolled-change cases; parser convergence/failure checks; resume/idempotency; replayed events; user plan versions; cancellation; UI reconnect; blinded evaluation leak checks; unit/reference consistency; record/source linking.
- **Document implementation changes** in the relevant docs and update [`docs/DECISIONS.md`](docs/DECISIONS.md) when a provisional design becomes a concrete decision.

## 12. Definition of done — minimum credible prototype

A minimal, honest end-to-end prototype should demonstrate all of the following:

- A user can submit a concrete simulation–experiment mismatch and keep chatting with the Chief Scientist.
- The Chief Scientist delegates to **dynamic** hypothesis agents, which register distinct falsifiable predictions before testing.
- At least one registered scientific protocol creates executable scientific inputs, runs or integrates a real calculation, and validates the relevant observable or energy difference.
- A long-running submitted task can be checkpointed; the agent releases runtime and resumes from a job-result event without duplicate submission. Other tasks and user conversation remain responsive.
- Evidence has IDs, links to raw outputs, methods, units, uncertainty/quality status, and clear non-comparability conditions.
- The Judge updates one assessment *in response to new evidence* and can explicitly return inconclusive results.
- The system records each investigation round, major decision, user intervention, and changed conclusion in readable Markdown and durable events.
- The terminal UI displays real per-agent and HPC status, with a pinned top panel and independent chat input.
- A demo/evaluation uses real or disclosed precomputed data, includes at least one baseline/control, and clearly identifies what remains unimplemented or unvalidated.

**Core maxim:** *The researcher directs the science. The Chief Scientist coordinates the investigation. The hypothesis swarm investigates, writes its own inputs and reads its own outputs. Verified protocols examine what the agents write and read before it counts. Evidence determines the claims.*
