# Sim2Real Scientist

**A hypothesis-driven scientific agent swarm for understanding and reducing discrepancies between simulations and experiments.**

Computational models and experimental measurements often disagree. The disagreement may arise from an approximation in the electronic-structure method, an idealized atomic structure, finite-temperature or environmental effects, an incorrect comparison of observables, numerical inaccuracies, or several interacting causes. A plausible explanation is not necessarily a verified explanation.

**Sim2Real Scientist** is a proposed, human-guided autonomous research system that investigates these discrepancies as scientific questions. A conversational **Chief Scientist** coordinates dynamically instantiated hypothesis agents. Each agent develops falsifiable predictions, requests validated computational analyses or evaluates experimental evidence, and reports what its results do—and do not—establish. An evidence-synthesis layer reconciles the findings and recommends the next computation or experiment expected to be most informative.

## Design principles

- **Hypotheses first:** Investigations create specialized agents for the actual scientific problem, not a fixed collection of agents for every run.
- **Human in control:** Researchers can converse with the Chief Scientist and change priorities while independent investigations continue.
- **Validated computation:** Agents specify *what* to test; versioned scientific protocols control *how* calculations are prepared, run, and checked.
- **Asynchronous HPC:** Long-running DFT and related calculations are submitted to schedulers such as Slurm. Agents release their runtime while jobs execute and resume when results arrive.
- **Evidence before conclusions:** Reports distinguish quantified contributions, model-dependent estimates, unresolved mechanisms, and non-identifiable explanations. The system can conclude that the evidence is insufficient.
- **Research provenance:** Every meaningful hypothesis, planned test, executed action, result, revision, and human intervention is recorded in a concise, auditable research notebook.
- **Terminal-native:** A planned Claude Code–style terminal interface combines an always-accessible Chief Scientist conversation with a persistent live agent/job status panel.

## Initial research demonstrations

The first proposed benchmark is the **CO/Cu(111) adsorption-site discrepancy**, where common semilocal DFT treatments can favor a hollow adsorption site while low-coverage experiments favor atop adsorption. The goal is not to rediscover a known literature result through retrieval, but to test whether agents can propose discriminating checks, carry them out with reliable computational protocols, and revise conclusions based on independently revealed evidence. A more challenging example from ongoing research is **TCNE/Au**, involving a possible disagreement between computed planar configurations and experimentally inferred upright/tilted arrangements; its exact experimental conditions and comparison must be established before use.

The project is being designed as a **general-purpose investigation framework**, rather than a hard-coded solution to either example. Its initial scope emphasizes atomistic materials science, surfaces, interfaces, DFT, MLIPs, and spectroscopy, with extensions to other computational sciences envisaged.

## Project status

**Design specification / pre-implementation.** The architecture and demonstration strategy are documented, but this repository description does not claim that the agent runtime, DFT workflows, terminal interface, or benchmarks have been implemented or validated yet. Installation and usage instructions will be added when working software exists.

For detailed implementation guidance, start with [`AGENTS.md`](AGENTS.md). Supporting design documents are in [`docs/`](docs/).
