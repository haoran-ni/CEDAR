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
- Agents work at the same time, one hypothesis each, numbered A1, A2, ...: each can see the others' work, read their files and use their calculations, but records only for its own hypothesis. When they have reported, the Chief Scientist compares their results across hypotheses, records that comparison (where each stands, how much of the discrepancy it accounts for, what remains unexplained), proposes new hypotheses from it, and reports to the researcher.
- Every task of an agent starts a new conversation with a briefing CEDAR builds from the record, so long investigations do not outgrow the model's context; agents can read their earlier conversations and the notebook. There is no limit on model calls. Agents end each task with a structured report (findings with evidence ids, open questions, requests to you); an agent is woken whenever one of its calculations ends; a run cut off by a temporary model-service error is continued.
- Agents write FHI-aims or VASP inputs, and may add any other file a run reads, such as a restart file; for VASP, CEDAR writes POTCAR from the PAW datasets the agent names, and the job script launches the build the inputs need (`ncl` for spin-orbit). Deterministic checks cover only what runs on the machine: the job script against your approved template, its limits and the budget; for VASP also that POTCAR matches POSCAR's species. No rule holds the DFT settings to a list or to what the agent declared. An input-reviewer agent then checks that the code will run the inputs as the calculation the agent describes, and that nothing in them is harmful, the files CEDAR does not read included, using the installed code's manual or source, and documentation sites on the web, where it is unsure. It leaves the scientific choices to the agents and to you. It approves the inputs, warns you to look first, or sends them back to the agent with its reasons. Only then are you asked to approve the run, the last step before it starts. You can also reject inputs, with a reason the agent is woken to address.
- Every agent reads any file you can, except credentials (CEDAR's configuration directory and the usual key stores); hypothesis agents write only in their own workspaces. There they run shell commands and scripts to check things, in a sandbox: no network, no scheduler, no DFT code, a time limit, every command recorded; what they compute this way is context, not evidence.
- Experimental knowledge, even a single sentence from you such as "the measured adsorption height is 2.5 Å", becomes evidence when an agent records it with its quoted source; quotes from your messages, files and PDFs are checked, and values read off figures or web pages stay provisional.
- Your earlier FHI-aims or VASP runs can be registered (`/register`, or by the agents and the Chief Scientist): their files are stored by hash and their health checked, and evidence from them is marked as computed outside CEDAR.
- When CEDAR runs on an interactive node, inside your Slurm allocation, calculations can also run there (locally) instead of as Slurm jobs, on all but 4 of the allocation's cores, which stay for the agents' commands; on a login node they cannot. Local runs need your `/approve submit` like Slurm jobs. Each runs as its own Slurm step on cores no other run uses, whatever MPI it uses; one that does not fit waits until enough cores are free, and the panel says what each waits for ("waiting for 60 free cores") or where it runs ("running on CPUs 0-59").
- Agents can look at images and PDFs: files you attach with `/attach`, which every agent of the investigation can view, and images or PDFs in their own run directories.
- With `/web on` (or `/web only <domains>`), the Chief Scientist and the hypothesis agents can search and read the web. It is off by default, and the Evidence Judge and the approver never use it. The input reviewer always reads code documentation sites, and nothing else.
- Researcher-authorized jobs run under Slurm. A job the scheduler refuses is tried again on the next passes; if it is still refused, its agent is woken with what the scheduler said, to fix the cause and request it again.
- Agents read the outputs and compute whatever quantities they need with their own analysis scripts, which run on the login node isolated with bubblewrap, seeing only the files staged for them (run outputs, and materials or files such as measured data). Cited readings and script results are verified or recorded before they become evidence.
- The Evidence Judge assesses hypotheses: when the Chief Scientist asks, or, in full automation, every round. A hypothesis can be supported only by a calculation it predicted before the calculation was submitted; earlier results, such as your registered runs and measurements, can challenge it but not support it.
- In full automation, an approver agent takes your routine decisions (starting agents, submitting calculations) within core-hour and dollar limits you set; the panel shows the mode.
- Budgets are in dollars, at the price of each kind of token (cached input costs far less than fresh input or output): every investigation may spend $25 on models unless you set another amount with `/budget`. The size of the Chief Scientist's conversation with you is a separate, token limit: past 800,000 tokens it continues in a new conversation, briefed from the record.
- When the Chief Scientist decides what to do next, it records the candidate steps it weighed (what each hypothesis predicts each would show, cost, feasibility, priority) and which it takes, and why. You see this in the chat and the notebook.
- A research notebook in Markdown, generated from the event log, records each round: what was hypothesized and why, predicted, tested and observed, what changed, what remains unknown and what is next.
- A full-screen terminal interface keeps a live panel of hypotheses, agents, jobs, evidence and assessments above the conversation, and opens notebook pages and cited output lines in place.

The full path passes automated tests that use stand-ins for the models and the scheduler. The workbench's sandbox (bubblewrap) has been tested on one cluster's login node; registering has been tried on a real FHI-aims run; the VASP protocol has been tried only on the files of one real VASP 5.4.4 run, outside CEDAR's dispatcher (its inputs examined, its POTCAR rebuilt identically from the library, its OUTCAR checked), and no VASP calculation has yet run through CEDAR; local calculations have run through CEDAR only with a stand-in for FHI-aims and a simulated allocation; on an interactive node, real FHI-aims runs (and Intel MPI and Open MPI test programs) started the way CEDAR starts them ran side by side as Slurm steps on separate cores, but not yet from within CEDAR. Web search and fetch, images and PDF pages have also been checked through CEDAR's model adapter and agent loop with a real Claude model, though not yet in a real investigation; with OpenAI's GPT-5.6 Sol, so have tool calls in a resumed conversation, images and PDF pages in a tool's output, web search within one site and a reply cut off at its output limit (`tests/test_openai_live.py`), but no agent has yet run on it in an investigation. On one Slurm cluster, with FHI-aims and Claude models, each step except the Evidence Judge has also run for real, though not yet together in one uninterrupted investigation. The terminal interface has also been used with the real Chief Scientist, and a real hypothesis agent has analysed a real FHI-aims output with its own scripts, both on test investigations. On another test investigation, without calculations, the real Chief Scientist proposed two agents and revised the proposal after a researcher reply; the two agents ran at the same time, and the Chief Scientist then reviewed their reports.

Not yet built:
- protocols for codes other than FHI-aims and VASP
- schedulers other than Slurm
- model providers other than Anthropic and OpenAI
- retries of agent work that failed for reasons other than a temporary model-service error or an interruption
- blind evaluation

The input reviewer has run only with stand-in models in tests, not yet with a real model on real inputs.

Nothing here has been validated as producing correct scientific conclusions.

## Installation

You need:
- access to a Linux cluster that runs Slurm and has FHI-aims or VASP installed (for calculations)
- conda (Miniconda or Anaconda), to create a Python 3.12 environment
- an Anthropic or OpenAI API key, or both

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
cedar auth add openai                   # the same for OpenAI, if you use its models
```

The keys are kept in `~/.config/cedar/credentials.toml`, readable only by you. If the `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable is set, it is used instead.

**5. Choose the models:**

```bash
cedar models init                       # writes ~/.config/cedar/models.toml
cedar models check                      # one tiny request to confirm the key and model work
```

`models.toml` suggests Claude Opus 5.5 for every agent; edit it to choose a model per agent role, from Anthropic or OpenAI (its comments show GPT-5.6 Sol for one role). With `cedar models check --role <role>`, the request goes to that role's model.

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

**Starting over:**

```bash
cedar reset                             # delete every investigation and its history; keep profiles, models and keys
cedar purge                             # delete everything CEDAR stored, the setup and API keys included
```

Each lists what it deletes and asks you to type its name (`--yes` skips that). Run directories, with the calculations' inputs and outputs, stay unless you add `--runs`. Neither deletes while a dispatcher is running or a calculation CEDAR started has not ended, as its scheduler reports: wait for it, or add `--cancel-jobs`. Only files CEDAR writes are deleted. An API key set in your environment stays set. `pip uninstall cedar` removes the program itself.

## Using CEDAR

The chat starts a dispatcher in the background whenever there is work for it: it starts the agents, submits what you authorize and follows the jobs, keeps going after you close the chat, and stops by itself after 30 idle minutes (`cedar dispatch --stop` stops it now; `cedar start --no-dispatch` leaves it to you, e.g. `cedar dispatch --agents N` in tmux). Proposed agents start after a 10-minute review window unless you reply; `/agents` shows them and `/approve agents` starts them at once. Every calculation's inputs go to the input reviewer first. Once it has finished, the chat asks you to approve the calculation, and your approval is the last step before it runs: `/approve submit <calculation>`, `/approve submit A2` for everything agent A2 waits to run, or `/approve submit all`. One approval covers the inputs and the submission. `/pending` shows who asks for each calculation, what it computes, where it runs, its size in tasks, minutes and core-hours, and the directory holding its examined inputs. Where the reviewer warned or could not decide, it also says what to look at, at which line of which file; `/approve submit all` takes those only once you have been shown them. It names each calculation by the first 8 characters of its id, which the commands accept. A calculation still with the reviewer cannot be approved yet. `/reject inputs <calculation> <reason>` refuses inputs before they run: the agent is woken with your reason. When you will be away, `/mode full-automation <core-hours> <dollars>` lets an approver agent decide on agents and submissions, and the Evidence Judge assess new evidence every round, within those limits; `/mode human-in-the-loop` takes the decisions back. Calculations whose inputs need your look, `/reject inputs`, `/approve close`, `/cancel` and `/halt` always stay yours. `/attach <path> <note>` gives the agents a figure, paper or data file to view, `/register <dir> <note>` brings a finished FHI-aims run of yours into the record for every agent, `/web on`, `/web only <domains>` or `/web off` sets whether they may use the web, and `/workbench off` or `/workbench on` whether they may run commands in their workspaces (on by default). To run calculations locally, start CEDAR on an interactive node, inside a Slurm allocation, and run the chat and the dispatcher there. Every investigation's agents may spend $25 on models in total, the Chief Scientist's answers included, unless you set another amount: `/budget 40` sets $40, `/budget` shows what is spent, and `/budget off` removes the limit. Running agents share the budget, checking what all of them have spent before every model call. All three also work before the investigation exists. CEDAR knows the published prices of the current Claude models and of GPT-5.6 Sol (`cedar models show` lists them); for another model, give its prices in `models.toml`. An investigation is closed only by you: the Chief Scientist proposes closing it, and `/approve close` closes it for good. `/halt` stops an investigation at once: it cancels its jobs, retires its agents and pauses it; `/halt all` does so for every investigation and stops the background dispatcher. Each investigation's notebook is in the data directory, by default `~/.local/share/cedar/notebooks/<investigation>/index.md`.

For detailed implementation guidance, start with [`AGENTS.md`](AGENTS.md). Design documents are in [`docs/`](docs/).
