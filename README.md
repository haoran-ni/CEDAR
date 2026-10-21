# Sim2Real Scientist

**A hypothesis-driven scientific agent swarm for understanding and reducing discrepancies between simulations and experiments.**

Computational models and experimental measurements often disagree. The disagreement may arise from an approximation in the electronic-structure method, an idealized atomic structure, finite-temperature or environmental effects, an incorrect comparison of observables, numerical inaccuracies, or several interacting causes. A plausible explanation is not necessarily a verified explanation.

**Sim2Real Scientist** is a proposed, human-guided autonomous research system that investigates these discrepancies as scientific questions. A conversational **Chief Scientist** coordinates dynamically instantiated hypothesis agents. Each agent develops falsifiable predictions, requests validated computational analyses or evaluates experimental evidence, and reports what its results do—and do not—establish. An evidence-synthesis layer reconciles the findings and recommends the next computation or experiment expected to be most informative.

## Design principles

- **Hypotheses first:** Investigations create specialized agents for the actual scientific problem, not a fixed collection of agents for every run.
- **Human in control:** Researchers can converse with the Chief Scientist and change priorities while independent investigations continue.
- **Examined computation:** Agents may write their own DFT inputs and job scripts and read raw outputs directly. Versioned protocol tools examine every input and job script before it runs, and verify the values agents read from outputs before they count as evidence.
- **Asynchronous HPC:** Long-running DFT and related calculations are submitted to schedulers such as Slurm. Agents release their runtime while jobs execute and resume when results arrive.
- **Evidence before conclusions:** Reports distinguish quantified contributions, model-dependent estimates, unresolved mechanisms, and non-identifiable explanations. The system can conclude that the evidence is insufficient.
- **Research provenance:** Every meaningful hypothesis, planned test, executed action, result, revision, and human intervention is recorded in a concise, auditable research notebook.
- **Terminal-native:** A planned Claude Code–style terminal interface combines an always-accessible Chief Scientist conversation with a persistent live agent/job status panel.

## Initial research demonstrations

The first proposed benchmark is the **CO/Cu(111) adsorption-site discrepancy**, where common semilocal DFT treatments can favor a hollow adsorption site while low-coverage experiments favor atop adsorption. The goal is not to rediscover a known literature result through retrieval, but to test whether agents can propose discriminating checks, carry them out with reliable computational protocols, and revise conclusions based on independently revealed evidence. A more challenging example from ongoing research is **TCNE/Au**, involving a possible disagreement between computed planar configurations and experimentally inferred upright/tilted arrangements; its exact experimental conditions and comparison must be established before use.

The project is being designed as a **general-purpose investigation framework**, rather than a hard-coded solution to either example. Its initial scope emphasizes atomistic materials science, surfaces, interfaces, DFT, MLIPs, and spectroscopy, with extensions to other computational sciences envisaged.

## Project status

**Early prototype.** The path from a described discrepancy to recorded evidence is built:
- The Chief Scientist opens an investigation and delegates work to hypothesis agents.
- Agents write FHI-aims inputs, which protocol tools examine.
- Researcher-authorized jobs run under Slurm.
- Agents read the outputs and compute whatever quantities they need with their own analysis scripts, which run under guards on the login node. Cited readings and script results are verified or recorded before they become evidence.
- The Evidence Judge assesses hypotheses.
- A research notebook in Markdown, generated from the event log, records each round: what was hypothesized and why, predicted, tested and observed, what changed, what remains unknown and what is next.
- A full-screen terminal interface keeps a live panel of hypotheses, agents, jobs, evidence and assessments above the conversation, and opens notebook pages and cited output lines in place.

The full path passes automated tests that use stand-ins for the models and the scheduler. On one Slurm cluster, with FHI-aims and Claude models, each step except the Evidence Judge has also run for real, though not yet together in one uninterrupted investigation. The terminal interface has also been used with the real Chief Scientist, and a real hypothesis agent has analysed a real FHI-aims output with its own scripts, both on test investigations.

Not yet built:
- protocols beyond FHI-aims total energies
- schedulers other than Slurm
- model providers other than Anthropic
- parallel agents
- blind evaluation

Nothing here has been validated as producing correct scientific conclusions.

## Getting started

Requirements: Python 3.12, a Slurm cluster with FHI-aims, and an Anthropic API key.

```bash
pip install -e ".[dev]"                 # in a conda env or venv; puts `sim2real` in its bin/
sim2real auth add anthropic             # stores the key in ~/.config/sim2real (mode 600)
sim2real models init                    # choose models per agent role in ~/.config/sim2real/models.toml
sim2real models check                   # one tiny request to confirm the key works
sim2real setup --name mycluster         # an agent works out how FHI-aims runs here and asks you questions
sim2real profile show mycluster
sim2real profile approve mycluster      # nothing runs on the cluster until you approve a profile
sim2real dispatch                       # keep running (e.g. in tmux): submits what you authorize, runs agents
sim2real start                          # live panel and chat with the Chief Scientist (--plain for line by line); /help lists commands
sim2real report                         # write the research notebook (the dispatcher also keeps it current)
```

Calculations are submitted only after you type `/approve submit <calculation>` in the chat. Each investigation's notebook is in the data directory, by default `~/.local/share/sim2real/notebooks/<investigation>/index.md`.

For detailed implementation guidance, start with [`AGENTS.md`](AGENTS.md). Design documents and the record of decisions are in [`docs/`](docs/).
