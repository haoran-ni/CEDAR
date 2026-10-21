# CEDAR

**Autonomous Scientific Agent Swarm for Computation-Experiment Discrepancies**

Computational models and experimental measurements often disagree. The disagreement may arise from an approximation in the electronic-structure method, an idealized atomic structure, finite-temperature or environmental effects, an incorrect comparison of observables, numerical inaccuracies, or several interacting causes. A plausible explanation is not necessarily a verified explanation.

**CEDAR** is a proposed, human-guided autonomous research system that investigates these discrepancies as scientific questions. A conversational **Chief Scientist** coordinates dynamically instantiated hypothesis agents. Each agent develops falsifiable predictions, requests validated computational analyses or evaluates experimental evidence, and reports what its results do—and do not—establish. An evidence-synthesis layer reconciles the findings and recommends the next computation or experiment expected to be most informative.

## Design principles

- **Hypotheses first:** Investigations create specialized agents for the actual scientific problem, not a fixed collection of agents for every run.
- **Human in control:** Researchers can converse with the Chief Scientist and change priorities while independent investigations continue.
- **Examined computation:** Agents may write their own DFT inputs and job scripts and read raw outputs directly. Versioned protocol tools examine every input and job script before it runs, and verify the values agents read from outputs before they count as evidence. Agents check things with sandboxed commands in their own workspaces, and need a calculation only when a hypothesis needs new DFT.
- **Asynchronous HPC:** Long-running DFT and related calculations are submitted to schedulers such as Slurm. Agents release their runtime while jobs execute and resume when results arrive.
- **Evidence before conclusions:** Reports distinguish quantified contributions, model-dependent estimates, unresolved mechanisms, and non-identifiable explanations. The system can conclude that the evidence is insufficient.
- **Research provenance:** Every meaningful hypothesis, planned test, executed action, result, revision, and human intervention is recorded in a concise, auditable research notebook.
- **Terminal-native:** A planned Claude Code–style terminal interface combines an always-accessible Chief Scientist conversation with a persistent live agent/job status panel.

## Initial research demonstrations

The first proposed benchmark is the **CO/Cu(111) adsorption-site discrepancy**, where common semilocal DFT treatments can favor a hollow adsorption site while low-coverage experiments favor atop adsorption. The goal is not to rediscover a known literature result through retrieval, but to test whether agents can propose discriminating checks, carry them out with reliable computational protocols, and revise conclusions based on independently revealed evidence. A more challenging example from ongoing research is **TCNE/Au**, involving a possible disagreement between computed planar configurations and experimentally inferred upright/tilted arrangements; its exact experimental conditions and comparison must be established before use.

The project is being designed as a **general-purpose investigation framework**, rather than a hard-coded solution to either example. Its initial scope emphasizes atomistic materials science, surfaces, interfaces, DFT, MLIPs, and spectroscopy, with extensions to other computational sciences envisaged.

## Project status

**Early prototype.** The path from a described discrepancy to recorded evidence is built:
- The Chief Scientist opens an investigation and proposes hypothesis agents: at most one per hypothesis, each with the role its work needs. The researcher sees the proposal; the agents start after 10 minutes unless the researcher replies.
- Agents work at the same time and independently: none sees another's work. When they have reported, the Chief Scientist collects and compares their results and reports to the researcher.
- Agents write FHI-aims inputs, which protocol tools examine.
- Agents read any file you can, except secrets and CEDAR's own records (which hold the other agents' work), and write only in their own workspaces. There they run shell commands and scripts to check things, in a sandbox: no network, no scheduler, no DFT code, a time limit, every command recorded; what they compute this way is context, not evidence.
- Your earlier FHI-aims runs can be registered (`/register`, or by the agents and the Chief Scientist): their files are stored by hash and their health checked, and evidence from them is marked as computed outside CEDAR.
- When CEDAR runs on an interactive node, inside your Slurm allocation, calculations can also run there (locally) instead of as Slurm jobs; on a login node they cannot. Local runs need your `/approve submit` like Slurm jobs.
- Agents can look at images and PDFs: files you attach with `/attach`, which every agent of the investigation can view, and images or PDFs in their own run directories.
- With `/web on` (or `/web only <domains>`), the Chief Scientist and the hypothesis agents can search and read the web. It is off by default, and the Evidence Judge and the approver never use it.
- Researcher-authorized jobs run under Slurm.
- Agents read the outputs and compute whatever quantities they need with their own analysis scripts, which run under guards on the login node. Cited readings and script results are verified or recorded before they become evidence.
- The Evidence Judge assesses hypotheses: when the Chief Scientist asks, or, in full automation, every round.
- In full automation, an approver agent takes your routine decisions (starting agents, submitting calculations) within core-hour and token limits you set; the panel shows the mode.
- When the Chief Scientist decides what to do next, it records the candidate steps it weighed (what each hypothesis predicts each would show, cost, feasibility, priority) and which it takes, and why. You see this in the chat and the notebook.
- A research notebook in Markdown, generated from the event log, records each round: what was hypothesized and why, predicted, tested and observed, what changed, what remains unknown and what is next.
- A full-screen terminal interface keeps a live panel of hypotheses, agents, jobs, evidence and assessments above the conversation, and opens notebook pages and cited output lines in place.

The full path passes automated tests that use stand-ins for the models and the scheduler. The workbench's sandbox (bubblewrap) has been tested on one cluster's login node; registering has been tried on a real FHI-aims run; local calculations have run only with a stand-in for FHI-aims and a simulated allocation, not yet with FHI-aims on an interactive node. Web search and fetch, images and PDF pages have also been checked through CEDAR's model adapter and agent loop with a real Claude model, though not yet in a real investigation. On one Slurm cluster, with FHI-aims and Claude models, each step except the Evidence Judge has also run for real, though not yet together in one uninterrupted investigation. The terminal interface has also been used with the real Chief Scientist, and a real hypothesis agent has analysed a real FHI-aims output with its own scripts, both on test investigations. On another test investigation, without calculations, the real Chief Scientist proposed two agents and revised the proposal after a researcher reply; the two agents ran at the same time, and the Chief Scientist then reviewed their reports.

Not yet built:
- protocols for codes other than FHI-aims
- schedulers other than Slurm
- model providers other than Anthropic
- retries of failed agent work
- blind evaluation

Nothing here has been validated as producing correct scientific conclusions.

## Installation

You need:
- access to a Linux cluster that runs Slurm and has FHI-aims installed (for calculations)
- conda (Miniconda or Anaconda), to create a Python 3.12 environment
- an Anthropic API key

**1. Get the code** and go into its folder:

```bash
git clone <repository-url> cedar
cd cedar
```

**2. Create a conda environment named `cedar`** with Python 3.12, and activate it:

```bash
conda create -n cedar python=3.12
conda activate cedar
```

If your cluster provides conda as a module, load it first (for example with `module load`). If `conda activate` says your shell is not set up for it, run `conda init bash` once and open a new terminal.

**3. Install CEDAR** into the environment, from the repository folder:

```bash
pip install -e ".[dev]"
cedar --version                         # prints: cedar 0.0.1
```

`-e` installs it in editable mode: changes to the code take effect without reinstalling, but if you move the repository folder, run `pip install -e ".[dev]"` again. `[dev]` adds pytest, so `pytest` runs the tests.

The `cedar` command lives in the environment's `bin/` folder and works without activating the environment. To use it from any shell, link it into a folder on your `PATH`, for example `ln -s "$(which cedar)" ~/.local/bin/cedar` (run this while the environment is active).

**4. Store your API key:**

```bash
cedar auth add anthropic                # asks for the key without showing it
```

The key is kept in `~/.config/cedar/credentials.toml`, readable only by you. If the `ANTHROPIC_API_KEY` environment variable is set, it is used instead.

**5. Choose the models:**

```bash
cedar models init                       # writes ~/.config/cedar/models.toml
cedar models check                      # one tiny request to confirm the key and model work
```

`models.toml` suggests Claude Opus 5.5 for every agent; edit it to choose a model per agent role.

**6. Describe your cluster** in a compute profile. Nothing runs on the cluster until you approve one:

```bash
cedar setup --name mycluster            # an agent reads your existing job scripts and asks you questions
cedar profile show mycluster            # review what it proposes
cedar profile approve mycluster
```

**7. Start CEDAR:**

```bash
cedar start                             # live panel and chat with the Chief Scientist; /help lists commands
```

`cedar start --plain` gives a line-by-line chat instead of the full-screen interface, and `cedar report` writes the research notebook (the dispatcher also keeps it current).

Configuration and keys live in `~/.config/cedar`; research data (event log, stored files, notebooks) in `~/.local/share/cedar`. Set `CEDAR_CONFIG_DIR` or `CEDAR_DATA_DIR` to put them elsewhere.

## Using CEDAR

The chat starts a dispatcher in the background whenever there is work for it: it starts the agents, submits what you authorize and follows the jobs, keeps going after you close the chat, and stops by itself after 30 idle minutes (`cedar dispatch --stop` stops it now; `cedar start --no-dispatch` leaves it to you, e.g. `cedar dispatch --agents N` in tmux). Proposed agents start after a 10-minute review window unless you reply; `/agents` shows them and `/approve agents` starts them at once. Calculations are submitted only after you type `/approve submit <calculation>` in the chat. When you will be away, `/mode full-automation <core-hours> <tokens>` lets an approver agent decide on agents and submissions, and the Evidence Judge assess new evidence every round, within those limits; `/mode human-in-the-loop` takes the decisions back, and `/approve inputs`, `/approve close`, `/cancel` and `/halt` always stay yours. `/attach <path> <note>` gives the agents a figure, paper or data file to view, `/register <dir> <note>` brings a finished FHI-aims run of yours into the record for every agent, `/web on`, `/web only <domains>` or `/web off` sets whether they may use the web, and `/workbench off` or `/workbench on` whether they may run commands in their workspaces (on by default). To run calculations locally, start CEDAR on an interactive node, inside a Slurm allocation, and run the chat and the dispatcher there. There is no token limit unless you set one: `/budget 5M` limits the tokens all agents of the investigation may process in total, the Chief Scientist's answers included, and `/budget off` removes it. All three also work before the investigation exists. An investigation is closed only by you: the Chief Scientist proposes closing it, and `/approve close` closes it for good. `/halt` stops an investigation at once: it cancels its jobs, retires its agents and pauses it; `/halt all` does so for every investigation and stops the background dispatcher. Each investigation's notebook is in the data directory, by default `~/.local/share/cedar/notebooks/<investigation>/index.md`.

For detailed implementation guidance, start with [`AGENTS.md`](AGENTS.md). Design documents are in [`docs/`](docs/).
