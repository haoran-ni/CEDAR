# Evidence and provenance

How Sim2Real Scientist records what happened in an investigation, and how a claim in a
report can be traced back to the files it rests on. [`DECISIONS.md`](DECISIONS.md) lists the
rules the record enforces; this document describes the structures.

## Two layers

| Layer | What it is | Who writes it |
|---|---|---|
| Event log | The source of truth: typed, append-only events in SQLite | Only `ResearchLedger`, after `projection.apply` accepts each event |
| Research notebook | Markdown pages generated from the event log | Only `notebook.update_notebook`; never edited by hand |

Research state (hypotheses, calculations, evidence, plans) is whatever replaying the log
produces. The notebook is a view of that log. Deleting the notebook loses nothing: it is
regenerated from the log.

## What lives where

Under the data directory (`$SIM2REAL_DATA_DIR`, else `$XDG_DATA_HOME/sim2real`, else
`~/.local/share/sim2real`):

| Path | Contents |
|---|---|
| `events.sqlite` | The event log of every investigation |
| `artifacts/objects/<aa>/<sha256>` | Stored files: structures, examined inputs and job scripts, raw outputs. Written once, read-only, addressed by content hash |
| `transcripts/` | Each agent session's conversation, one JSON line per message (mode 600). Operational memory for resuming agents, not the research record |
| `notebooks/<investigation>/` | The research notebook |
| `dispatcher.lock`, `dispatcher.heartbeat` | The dispatcher's lock and last-tick time |

Run directories, where jobs actually execute, are under the compute profile's `runs_root`.
The compute profile itself and the API keys live in the configuration directory, never in
the data directory.

## The event log

Each event is an envelope around one typed payload. The envelope records:

- `event_id`, `investigation_id`, and `stream_version` (contiguous from 1 per investigation)
- `occurred_at` (UTC) and the `actor`: its kind (user, Chief Scientist, hypothesis agent, Evidence Judge, protocol engine, job manager, job monitor, validator, system) and id
- `round` and `plan_version` at the time of recording
- an optional `idempotency_key`, so a retried write is recorded once
- an optional `causation_id`

Payloads cover the whole research process:

| Area | Events |
|---|---|
| Problem | `investigation_created`, `investigation_status_changed`, `observable_alignment_recorded` |
| Direction | `user_intervention_recorded` (the researcher's words, verbatim, with any authorization), `plan_revised` (with the message that prompted it), `round_started` |
| Hypotheses | `hypothesis_proposed`, `hypothesis_status_changed`, `prediction_registered` |
| Computation | `calculation_requested`, `calculation_linked`, `calculation_inputs_examined`, `calculation_state_changed` |
| Results | `evidence_recorded`, `assessment_recorded` |
| Agents | `agent_task_queued`, `agent_run_recorded` (model, tokens, outcome, and the agent's final report) |

Events are never updated or deleted; database triggers refuse both. A correction is a new
event.

## Evidence objects

An `evidence_recorded` event carries an `EvidenceObject`:

- **kind**: direct computation, archived computation, published experiment, inference, or indirect constraint
- **observable**, with its definition and unit, and the **value** with unit, reference state and uncertainty components
- **quality**: `VALIDATED`, `PROVISIONAL`, `INCONCLUSIVE` or `INVALID`, with every **quality check** that produced it
- **sources**: stored artifacts (raw outputs, inputs) or external citations
- **citations**: for each value read from an output, the stored file, the line range and the quoted text
- **interpreted_by**: the agent whose reading it is, if an agent read it
- **limitations**, **assumptions**, and **non-comparability** conditions
- the **method**: the code, protocol and declared settings

A computed value always cites the output lines it came from, and those lines are checked
verbatim against the stored, hashed file before the evidence is recorded.

## Tracing a claim

Starting from a statement in a report, for example "hyp-xc is INCONCLUSIVE":

1. The **assessment** names the evidence it cites and gives the judge's rationale.
2. Each **evidence object** names its calculation, the quality checks, and the cited output lines with the artifact's hash.
3. The **artifact store** holds the cited output, byte for byte; its hash proves it is the file that was checked.
4. The **calculation** records the task (observable, estimand, variables varied, controlled and set for accuracy), the examination of its exact input files and job script, the scheduler job id, and the researcher message that authorized submission.
5. The **hypothesis** records the predictions registered before the outcome, and when.
6. The **plan** versions and **researcher messages** show who directed the work and why.

The notebook pages follow the same chain, with links from evidence to the stored files.

## The research notebook

Pages under `notebooks/<investigation>/`:

| Page | Contents |
|---|---|
| `index.md` | Question and target claim; compared quantities and their alignment; the plan and its versions; tables of hypotheses and calculations; links to rounds; what waits for the researcher; model usage |
| `round-NN.md` | One page per round, answering the questions laid out in [`templates/ROUND_REPORT.md`](../templates/ROUND_REPORT.md): what we hypothesized and why, what we predicted, tested and observed, what changed our assessment, what remains unknown, and what is next; then the researcher's messages and the agents' reports |
| `hypotheses/<id>.md` | One hypothesis: statement, falsifiers, predictions, calculations, evidence, assessment history, agent reports |
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
- **No hidden reasoning.** The notebook contains the agents' stated summaries and the reasons they recorded, never model-internal reasoning. Secrets never reach the log, so they cannot reach the notebook.

The dispatcher regenerates the notebook of every investigation whose log grew, on each
tick. `/report` in the chat and `sim2real report` regenerate it on demand. Writers take a
lock and render from the log as it is once they hold it, and only pages whose content
changed are rewritten, each atomically.
