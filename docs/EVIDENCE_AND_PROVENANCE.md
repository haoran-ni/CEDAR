# Evidence and provenance

How CEDAR records what happened in an investigation, and how a claim in a
report can be traced back to the files it rests on.

## Two layers

| Layer | What it is | Who writes it |
|---|---|---|
| Event log | The source of truth: typed, append-only events in SQLite | Only `ResearchLedger`, after `projection.apply` accepts each event |
| Research notebook | Markdown pages generated from the event log | Only `notebook.update_notebook`; never edited by hand |

Research state (hypotheses, calculations, evidence, plans) is whatever replaying the log
produces. The notebook is a view of that log. Deleting the notebook loses nothing: it is
regenerated from the log.

## What lives where

Under the data directory (`$CEDAR_DATA_DIR`, else `$XDG_DATA_HOME/cedar`, else
`~/.local/share/cedar`):

| Path | Contents |
|---|---|
| `events.sqlite` | The event log of every investigation |
| `artifacts/objects/<aa>/<sha256>` | Stored files: structures, examined inputs and job scripts, raw outputs, analysis scripts with what they printed and wrote, and the materials the researcher attached. Written once, read-only, addressed by content hash |
| `analyses/` | Working directories of analysis scripts while they run; each is removed when its script ends |
| `workspaces/<investigation>/<hypothesis>/` | Each hypothesis agent's workspace, the only place it writes; what its workbench commands printed is in `workbench-output/` there |
| `transcripts/` | Each agent conversation, one JSON line per message (mode 600), including the images and PDF pages an agent viewed and the web results the provider returned: one per task of a hypothesis agent, a review, the Evidence Judge or the approver, and the chat with the researcher, continued in a new one when it grows large. Every task opens with a briefing built from the record that names the agent's earlier conversations, which agents may read. Not the research record |
| `notebooks/<investigation>/` | The research notebook |
| `dispatcher.lock`, `dispatcher.heartbeat`, `dispatcher.pid`, `dispatcher.host`, `dispatcher.log` | The dispatcher's lock, last-pass time, process id and machine, and the output of one the chat started |
| `activity/` | One file per agent run in progress, for the live panel; removed when the run ends |

Run directories, where jobs actually execute, are under the compute profile's `runs_root`; the local backend's records of local runs are in `.cedar-local/` there.
The compute profile itself and the API keys live in the configuration directory, never in
the data directory.

## The event log

Each event is an envelope around one typed payload. The envelope records:

- `event_id`, `investigation_id`, and `stream_version` (contiguous from 1 per investigation)
- `occurred_at` (UTC) and the `actor`: its kind (user, Chief Scientist, hypothesis agent, Evidence Judge, approver, input reviewer, protocol engine, job manager, job monitor, validator, analysis runner, workbench, system) and id. A hypothesis agent's id, such as `agent-hyp-xc`, names the one hypothesis it works on
- `round` and `plan_version` at the time of recording
- an optional `idempotency_key`, so a retried write is recorded once
- an optional `causation_id`

Payloads cover the whole research process:

| Area | Events |
|---|---|
| Problem | `investigation_created`, `investigation_status_changed` (closing carries the researcher's authorization), `closing_proposed` (by the Chief Scientist; only the researcher closes), `observable_alignment_recorded` |
| Direction | `user_intervention_recorded` (the researcher's words, verbatim, with any authorization), `plan_revised` (with the message that prompted it), `round_started`, `synthesis_recorded` (the Chief Scientist's comparison across hypotheses at a review: where each open hypothesis stands and how much of the discrepancy it accounts for, citing evidence, their interactions, what remains unexplained, what cannot be separated, and the new hypotheses it suggests; closing needs one newer than the latest evidence), `next_actions_recommended` (the Chief Scientist's candidate next steps, compared, and its choice; delegations and agent proposals cite it), `mode_changed` (human-in-the-loop or full automation, with its limits), `approver_decision` (in full automation, the approver's approval or refusal of starting agents or a submission, with its reason), `retrieval_policy_changed` (the researcher's setting of the agents' web access: off, on, or only some domains), `workbench_policy_changed` (the researcher's setting of the agents' workbench: on or off), `budget_changed` (what the investigation's agents may spend on models in total, in dollars, or no limit; an investigation opens with $15) |
| Materials | `material_attached` (a file the researcher gave the agents: its stored copy, name and note; context, not evidence) |
| Hypotheses | `hypothesis_proposed`, `hypothesis_status_changed`, `prediction_registered` (naming the calculations it predicts, any of the investigation's, before they are submitted) |
| Computation | `calculation_requested` (with where it runs: Slurm or local), `calculation_linked`, `calculation_inputs_examined` (the files, findings, every setting the main input file makes, whether the input reviewer checks the content, and whether the examination held the files to the task's declared settings, as `fhiaims.run` 0.1 to 0.3 did), `input_review_recorded` (the input reviewer's verdict on one calculation's inputs: approved, warned or rejected, with findings, their cited sources, and the role of every keyword; a model's judgement, never evidence), `calculation_state_changed` (a rejection or cancellation carries its reason; a rejection of inputs, also the researcher's message or the input reviewer's review that made it; a local run's move to queued, running or ended carries the backend's note, e.g. "waiting for 60 free cores"), `scheduler_note_recorded` (a queued or running local run's new note when its state stays, e.g. "waiting for Slurm to start its step (Requested nodes are busy)"), `run_checked` (the run's health checks and stored outputs), `run_registered` (a run computed outside CEDAR: who registered it and from where, its stored inputs and outputs, the settings its inputs used, its health checks; for one hypothesis or shared by every agent) |
| Results | `analysis_recorded` (a script an agent ran: the script, its hashed inputs from runs and other staged data, how it was isolated, output and results), `evidence_recorded`, `assessment_recorded` |
| Workbench | `workbench_run_recorded` (a command an agent ran in its workspace: what, where, with which resources, how it ended, what it printed, and the hash of each file it wrote; context, not evidence) |
| Agents | `agents_proposed` (by the Chief Scientist: per hypothesis a role and a first task), `agents_started` (by the researcher's command or after the review window; the agents are numbered A1, A2, ... in this order), `proposal_withdrawn`, `agent_retired`, `agent_task_queued` (work for an agent, the Judge, the input reviewer (naming the calculations it reviews), or a Chief Scientist review; a continuation names the task whose run was cut off), `agent_run_recorded` (model, tokens, web searches and fetches, what it cost at the model's prices, outcome, the event its briefing showed, and the report: a hypothesis agent's structured report, its final message, or, when the run ended without one, CEDAR's account of what it did, built from the record) |

Events are never updated or deleted; database triggers refuse both. A correction is a new
event. `cedar reset` and `cedar purge` delete the whole log, with everything else in the
data directory; nothing deletes part of it.

## Evidence objects

An `evidence_recorded` event carries an `EvidenceObject`:

- **kind**: direct computation (a reading of one run CEDAR made), archived computation (a reading of one registered run, computed outside CEDAR), analysis (a result of a recorded analysis script), published experiment (a measured value, recorded with its quoted source: a researcher's message, the investigation's recorded measurement, an attached material, a file or a web page), inference, or indirect constraint (a fact about the experiment, such as its coverage or temperature)
- **observable**, with its definition and unit, and the **value** with unit, reference state and uncertainty components
- **quality**: `VALIDATED`, `PROVISIONAL`, `INCONCLUSIVE` or `INVALID`, with every **quality check** that produced it
- **sources**: stored artifacts (raw outputs, inputs, analysis scripts) or external citations
- **citations**: for each value read from an output, the stored file, the line range and the quoted text
- **calculation_id**, or **calculation_ids**, **analysis_id** and **analysis_result**: the run a reading came from, or the analysis, the result of it, and the runs it used
- **interpreted_by**: the agent whose reading it is, if an agent read it
- **limitations**, **assumptions**, and **non-comparability** conditions
- the **method**: the code, protocol and declared settings (from `fhiaims.run` 0.4, the agent's description; what the inputs set is in the calculation's examination)

A run gives evidence only after its run check; a registered run, after the health check made when it was registered. Experimental evidence is recorded by the validator for the agent that cites it: a quote from a message, a text file or a PDF is checked against the stored text, and a value must appear in its quote; values read off a figure or a web page are `PROVISIONAL`. A value read from an output cites the lines
it came from, checked verbatim against the stored, hashed file. A value computed by an
analysis script is taken from the analysis record, which holds the script and every input
by hash ([`ANALYSIS_SCRIPTS.md`](ANALYSIS_SCRIPTS.md)). Either way the evidence is
`VALIDATED` only if every run it rests on passed its checks.

## Tracing a claim

Starting from a statement in a report, for example "hyp-xc is INCONCLUSIVE":

1. The **assessment** names the evidence it cites and gives the judge's rationale.
2. Each **evidence object** names its calculation and quality checks, and either the cited output lines with the artifact's hash, or the analysis that computed it.
3. An **analysis** names the script, every staged input file by hash, what the script printed, and the results and checks it reported.
4. The **artifact store** holds each of those files, byte for byte; a hash proves a file is the one that was checked or analysed.
5. The **calculation** records the task (observable, estimand, variables varied, controlled and set for accuracy), the examination of its exact input files and job script, what the inputs actually set, the input reviewer's verdict, the scheduler job id, the researcher message that approved it to run (its inputs, where they waited for the researcher, and its submission), and the run check.
6. The **hypothesis** records the predictions registered before the outcome, and when. A `SUPPORTED` assessment must cite evidence from a calculation one of its predictions named before the calculation was submitted; evidence that existed before, or that nobody predicted, can only challenge it.
7. The **plan** versions and **researcher messages** show who directed the work and why.

The notebook pages follow the same chain, with links from evidence to the stored files.

## The research notebook

Pages under `notebooks/<investigation>/`:

| Page | Contents |
|---|---|
| `index.md` | Question and target claim; compared quantities and their alignment; web access and the attached materials; the plan and its versions; tables of hypotheses, agents (with every proposal) and calculations; links to rounds; what waits for the researcher; model usage |
| `round-NN.md` | One page per round, answering the questions laid out in [`templates/ROUND_REPORT.md`](../templates/ROUND_REPORT.md): what we hypothesized and why, what we predicted, tested and observed, what changed our assessment (with the Chief Scientist's comparisons across hypotheses and the values of the evidence they cite), what remains unknown, and what is next; then the researcher's messages and the agents' reports |
| `hypotheses/<id>.md` | One hypothesis: statement, falsifiers, its agent, predictions, calculations (where each ran, or that it was computed outside CEDAR), analyses, workbench commands (context, not evidence), evidence, assessment history, agent reports |
| `timeline.md` | Every event, one line each, in order |

A **round** is one stretch of investigating. Round 0 is the setup. A new round starts each
time the investigation moves into `INVESTIGATING`, after planning or after a review, or when
the Chief Scientist starts one explicitly. A `round_started` event opens the round it names.

Rules the pages follow:

- **Generated, not written.** The same log always gives the same pages. Each page states the events it shows.
- **Attributed words.** Reasons, rationales, messages and reports are quoted and attributed to their author. An agent's report is labeled as its own words; it is not evidence, and values in it count only where they appear in recorded evidence.
- **Values from evidence.** Numbers come from evidence objects at full recorded precision, with quality and failed checks shown next to them.
- **Job state is not validity.** Scheduler state and evidence quality are separate columns and sentences.
- **Unknowns are listed.** Unaligned observables, unassessed or unsettled hypotheses, and evidence that is not `VALIDATED` are listed as open.
- **Context is not evidence.** Attached materials and what agents read on the web inform the work, and reports say where a fact came from; values count only where they appear in recorded evidence.
- **No hidden reasoning.** The notebook contains the agents' stated summaries and the reasons they recorded, never model-internal reasoning. Secrets never reach the log, so they cannot reach the notebook.

The dispatcher regenerates the notebook of every investigation whose log grew, on each
tick. `/report` in the chat and `cedar report` regenerate it on demand. Writers take a
lock and render from the log as it is once they hold it, and only pages whose content
changed are rewritten, each atomically.
