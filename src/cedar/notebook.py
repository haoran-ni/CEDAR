"""The research notebook: readable Markdown pages generated from the event log.

The event log is the record; the notebook is a view of it. Pages are regenerated whenever
the log grows and are never edited by hand, and each page says which events it shows.
What the researcher or an agent wrote is quoted and attributed. Values come from recorded
evidence, never from an agent's prose.

Pages, in ``notebooks/<investigation>/`` under the data directory:

- ``index.md``: question, compared quantities, plan, hypotheses, agents, calculations, rounds
- ``round-NN.md``: one page per round, under the headings in ``ROUND_SECTIONS``
- ``hypotheses/<id>.md``: everything recorded about one hypothesis
- ``timeline.md``: every event, one line each
"""

import fcntl
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from cedar.contracts.base import Actor, ActorKind, ArtifactRef, ExternalSource, Observable, Quantity
from cedar.contracts.analysis import AnalysisRun
from cedar.contracts.review import InputReviewRecorded, ReviewSeverity
from cedar.contracts.events import (
    AgentRetired,
    AgentRunRecorded,
    AgentsProposed,
    AgentsStarted,
    AgentTaskQueued,
    ProposalWithdrawn,
    AnalysisRecorded,
    ApproverDecision,
    AssessmentRecorded,
    CalculationInputsExamined,
    CalculationLinked,
    CalculationRequested,
    CalculationStateChanged,
    ClosingProposed,
    EvidenceRecorded,
    HypothesisProposed,
    HypothesisStatusChanged,
    InvestigationCreated,
    InvestigationStatusChanged,
    MaterialAttached,
    ModeChanged,
    NextActionsRecommended,
    SynthesisRecorded,
    ObservableAlignmentRecorded,
    PlanRevised,
    PredictionRegistered,
    RecordedEvent,
    RetrievalPolicyChanged,
    RoundStarted,
    BudgetChanged,
    RunChecked,
    SchedulerNoteRecorded,
    SubmissionRefused,
    RunRegistered,
    UserInterventionRecorded,
    WorkbenchPolicyChanged,
    WorkbenchRunRecorded,
)
from cedar.contracts.evidence import EvidenceObject
from cedar.contracts.examination import FindingSeverity
from cedar.contracts.hypothesis import HypothesisSpec, Quantifiability
from cedar.contracts.investigation import AlignmentStatus
from cedar.contracts.plan import PermissionAction
from cedar.contracts.status import (
    Assessment,
    CalculationState,
    Completeness,
    EvidenceQuality,
    HypothesisStatus,
    InvestigationMode,
    SchedulerState,
)
from cedar.models.prices import dollars
from cedar.state.projection import (
    CalculationRecord,
    HypothesisRecord,
    InvestigationState,
    apply,
    evidence_owner,
    hypothesis_evidence,
    review_deadline,
)
from cedar.summaries import approval_requests
from cedar.web import web_summary

if TYPE_CHECKING:
    from cedar.workspace import Workspace

ROUND_SECTIONS = (
    "What did we hypothesize?",
    "Why?",
    "What did we predict?",
    "What did we test?",
    "What did we observe?",
    "What changed our assessment?",
    "What remains unknown?",
    "What is next?",
    "Researcher's messages",
    "Agent reports",
)
"""Headings of every round page, in order; ``templates/ROUND_REPORT.md`` describes each."""

_NOTE = ("*Generated from the event log; do not edit. Quoted text is its author's own words. Values come from "
         "recorded evidence.*")
_CLOSED = frozenset({HypothesisStatus.DISCARDED, HypothesisStatus.SUPERSEDED})
_FINISHED = frozenset({CalculationState.QUALITY_ASSESSED, CalculationState.REJECTED, CalculationState.SUBMISSION_FAILED,
                       CalculationState.CANCELLED})
_UNSETTLED = frozenset({Assessment.NOT_TESTED, Assessment.INCONCLUSIVE, Assessment.NOT_COMPARABLE})


# -- text -----------------------------------------------------------------------------------

def _flat(text: str) -> str:
    return " ".join(text.split())


def _short(text: str, limit: int = 160) -> str:
    text = _flat(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _said(text: str) -> str:
    """Someone's own words, inline."""
    return f"“{_flat(text)}”"


def _listed(label: str, items: Sequence[str], indent: str = "", quoted: bool = False) -> list[str]:
    """A labeled bullet: inline for one item, nested bullets for several."""
    shown = [_said(x) if quoted else _flat(x) for x in items]
    if len(shown) == 1:
        return [f"{indent}- {label}: {shown[0]}"]
    return [f"{indent}- {label}:"] + [f"{indent}  - {x}" for x in shown]


def _quote(text: str, indent: str = "") -> list[str]:
    """Someone's own words, as a block quote."""
    return [f"{indent}> {line}".rstrip() for line in text.strip().splitlines()]


def _code(text: str) -> str:
    """Text exactly as it appears in a file, as inline code."""
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    return f"{fence} {text} {fence}" if text.startswith("`") or text.endswith("`") else f"{fence}{text}{fence}"


def _cell(text: str) -> str:
    return _flat(text).replace("|", "\\|")


def _when(event: RecordedEvent) -> str:
    return _clock(event.occurred_at)


def _clock(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _quantity(q: Quantity | None) -> str:
    if q is None:
        return "no value"
    text = f"{q.value!r} {q.unit}"
    if q.reference:
        text += f" (reference: {q.reference})"
    for u in q.uncertainties:
        text += f" ± {u.value!r} ({u.kind.value}; {_flat(u.basis)})"
    return text


def _observable(o: Observable) -> str:
    return f"{o.name}{f' [{o.unit}]' if o.unit else ''}, defined as {_flat(o.definition)}"


def _settings(values: dict) -> str:
    if not values:
        return "none"
    return ", ".join(f"{k} = {v if isinstance(v, str) else json.dumps(v)}" for k, v in values.items())


def _actor(actor: Actor) -> str:
    if actor.kind is ActorKind.USER:
        return f"researcher {actor.id}"
    if actor.kind is ActorKind.CHIEF_SCIENTIST:
        return "Chief Scientist"
    if actor.kind is ActorKind.EVIDENCE_JUDGE:
        return "Evidence Judge"
    if actor.kind in (ActorKind.HYPOTHESIS_AGENT, ActorKind.SYSTEM):
        return actor.id
    return actor.kind.value.replace("_", " ")


def round_file(number: int) -> str:
    return f"round-{number:02d}.md"


def _round_name(number: int) -> str:
    return "Round 0 (setup)" if number == 0 else f"Round {number}"


@dataclass(frozen=True)
class _Here:
    """Where a page sits in the notebook, for relative links."""

    depth: int

    def hyp(self, hypothesis_id: str) -> str:
        return f"[{hypothesis_id}]({'./' if self.depth else 'hypotheses/'}{hypothesis_id}.md)"

    def artifact(self, ref: ArtifactRef, text: str, line: int | None = None) -> str:
        anchor = f"#L{line}" if line else ""
        return f"[{text}]({'../' * (self.depth + 2)}artifacts/{ref.path}{anchor})"


ROOT, SUB = _Here(0), _Here(1)


def _section(title: str, body: list[str]) -> str:
    return f"## {title}\n\n" + "\n".join(body)


# -- shared pieces ----------------------------------------------------------------------------

def _spec_details(spec: HypothesisSpec, indent: str = "") -> list[str]:
    rows = [("Falsified by", spec.falsifiers), ("Discriminating observations", spec.discriminating_observations),
            ("Assumptions", spec.assumptions)]
    lines = [line for label, items in rows if items for line in _listed(label, items, indent)]
    lines += [f"{indent}- {label}: {', '.join(ids)}" for label, ids in (("Depends on", spec.depends_on),
                                                                       ("Interacts with", spec.interacts_with)) if ids]
    if spec.mechanism:
        lines.append(f"{indent}- Mechanism: {_flat(spec.mechanism)}")
    if spec.quantifiability is not Quantifiability.UNKNOWN:
        lines.append(f"{indent}- Quantifiability: {spec.quantifiability.value.replace('_', ' ')}")
    if spec.estimated_cost:
        lines.append(f"{indent}- Estimated cost: {_flat(spec.estimated_cost)}")
    return lines


def _run_check(check: RunChecked | None) -> str:
    if check is None:
        return "not checked"
    failed = [c.name for c in check.checks if not c.passed]
    return f"failed {', '.join(failed)}" if failed else "healthy"


def _calc_status(calc: CalculationRecord) -> str:
    text = calc.state.value
    if calc.scheduler_job_id:
        text += f"; job {calc.scheduler_job_id} {calc.scheduler_state.value if calc.scheduler_state else 'submitted'}"
    if calc.run_check is not None:
        text += f"; run {_run_check(calc.run_check)}"
    if calc.evidence_ids:
        text += f"; evidence {', '.join(calc.evidence_ids)}"
    return text


def _inputs(calc: CalculationRecord, indent: str) -> list[str]:
    if calc.registered:
        registration = calc.registration
        lines = [f"{indent}- Inputs registered from `{registration.source}` with its output: computed outside CEDAR "
                 f"({registration.code} {registration.code_version}), not examined before it ran"]
        lines += [f"{indent}  - {n.severity.value} {n.code}: {_flat(n.message)}" for n in registration.notes
                  if n.severity in (FindingSeverity.ERROR, FindingSeverity.REVIEW, FindingSeverity.WARNING)]
        return lines
    exam = calc.examination
    if exam is None:
        return [f"{indent}- Inputs: not examined yet"]
    structure = calc.task.structure.sha256 if calc.task.structure else None
    by_agent = [n for n, ref in sorted(exam.files.items())
                if n in calc.task.authored_files or ref.sha256 == structure]
    generated = [n for n in sorted(exam.files) if n not in by_agent]
    written = "; ".join(part for part in (
        f"written by the agent: {', '.join(by_agent)}" if by_agent else "",
        f"generated by the protocol: {', '.join(generated)}" if generated else "") if part)
    counts = Counter(f.severity for f in exam.findings)
    tally = ", ".join(f"{counts[s]} {s.value}" for s in (FindingSeverity.ERROR, FindingSeverity.REVIEW,
                                                          FindingSeverity.WARNING) if counts[s])
    lines = [f"{indent}- Inputs {written}; examination {exam.verdict.value}" + (f" ({tally})" if tally else "")]
    lines += [f"{indent}  - {f.severity.value} {f.code}: {_flat(f.message)}" for f in exam.findings
              if f.severity in (FindingSeverity.ERROR, FindingSeverity.REVIEW)]
    review = calc.input_review
    if review is not None:
        lines.append(f"{indent}  - input reviewer ({review.review_id}), an agent's judgement: {review.decision.value}: "
                     f"{_flat(review.summary)}")
        lines += [f"{indent}    - {f.severity.value}: {_flat(f.text)}" for f in review.findings
                  if f.severity is not ReviewSeverity.NOTE]
    elif exam.agent_review and calc.state is CalculationState.SPECIFIED:
        lines.append(f"{indent}  - waiting for the input reviewer")
    return lines


def _where(calc: CalculationRecord) -> str:
    """Where a calculation ran or runs."""
    if calc.registered:
        return "computed outside CEDAR"
    return "runs locally on an interactive node" if calc.task.execution == "local" else "runs through Slurm"


def _workbench_lines(record, here: _Here) -> list[str]:
    run = record.run
    code = f", exit code {run.exit_code}" if run.exit_code is not None else ""
    changed = f"; wrote {', '.join(sorted(run.changed)[:8])}" + (" …" if len(run.changed) > 8 else "") \
        if run.changed else ""
    output = f"; {here.artifact(run.output, 'output')}" if run.output else ""
    return [f"- {record.run_id} by {record.agent_id} for {here.hyp(record.hypothesis_id)}: "
            f"{_code(_short(run.command, 200))} "
            f"{run.outcome.value}{code}, {run.duration_seconds:g} s on {run.host}{changed}{output}",
            f"  - Purpose: {_said(run.purpose)}"]


def _users(calc: CalculationRecord, here: _Here) -> str:
    return "every agent" if calc.shared else ", ".join(here.hyp(h) for h in calc.hypothesis_ids)


def _origin(calc: CalculationRecord, events: Sequence[RecordedEvent]) -> tuple[str, Actor]:
    """Why the calculation exists and who asked: its request, or its registration."""
    if calc.registered:
        return calc.registration.description, calc.registration.requested_by
    request = events[calc.requested_at - 1]
    return request.payload.rationale, request.actor


def _calculation_lines(calc: CalculationRecord, rationale: str, requester: Actor, here: _Here,
                       status_label: str) -> list[str]:
    t = calc.task
    what = "registered run" if calc.registered else _flat(t.estimand)
    return [
        f"- **{calc.calculation_id}** for {_users(calc, here)}: {what} ({_flat(t.system_label)}); "
        f"{_where(calc)}",
        f"  - Why, by {_actor(requester)}: {_said(rationale)}",
        f"  - Protocol {t.protocol.name} {t.protocol.version}; observable {t.observable.name}; "
        f"fidelity {t.fidelity.value}",
        f"  - Varied: {_settings(t.interventions)}. Controls: {_settings(t.controls)}. "
        f"Numerical: {_settings(t.numerical_settings)}",
        *_inputs(calc, "  "),
        f"  - {status_label}: {_calc_status(calc)}",
    ]


def _evidence_lines(evidence_id: str, ev: EvidenceObject, state: InvestigationState, here: _Here) -> list[str]:
    if ev.analysis_id is not None:
        hypotheses = [state.analyses[ev.analysis_id].hypothesis_id]
        origin = f"from analysis {ev.analysis_id} of {', '.join(ev.calculation_ids)}"
    elif ev.calculation_id is not None:
        owner = evidence_owner(state, ev)
        hypotheses = [owner] if owner is not None else state.calculations[ev.calculation_id].hypothesis_ids
        origin = f"from {ev.calculation_id}" + (" (registered run)" if state.calculations[ev.calculation_id].registered
                                                else "")
    else:
        hypotheses, origin = [], f"({ev.kind.value.replace('_', ' ')})"
    linked = f" for {', '.join(here.hyp(h) for h in hypotheses)}" if hypotheses else ""
    lines = [f"- **{evidence_id}** {origin}{linked}: **{ev.quality.value}**; "
             f"{ev.observable.name} = {_quantity(ev.value)}",
             f"  - Summary by {_actor(ev.interpreted_by) if ev.interpreted_by else 'the protocol tools'}: "
             f"{_said(ev.summary)}"]
    for c in ev.citations:
        span = f"line {c.line_start}" if c.line_start == c.line_end else f"lines {c.line_start}–{c.line_end}"
        where = here.artifact(c.artifact, f"{c.artifact.label or 'output'} {span}", c.line_start)
        lines.append(f"  - Read from {where}: {_code(_flat(c.quote))}")
    passed = [c.name for c in ev.quality_checks if c.passed]
    if passed:
        lines.append(f"  - Checks passed: {', '.join(passed)}")
    lines += [f"  - Check failed: {c.name}" + (f": {_flat(c.detail)}" if c.detail else "")
              for c in ev.quality_checks if not c.passed]
    lines += [f"  - Limitation: {_flat(x)}" for x in ev.limitations]
    lines += [f"  - Not comparable when: {_flat(x)}" for x in ev.non_comparability]
    lines += [f"  - Assumption: {_flat(x)}" for x in ev.assumptions]
    if ev.method:
        lines.append(f"  - Method: {_settings(ev.method)}")
    outputs = [here.artifact(s, s.label or s.sha256[:12]) for s in ev.sources if isinstance(s, ArtifactRef)]
    if outputs:
        lines.append(f"  - Stored files: {', '.join(outputs)}")
    for s in ev.sources:
        if isinstance(s, ExternalSource):
            extra = "; ".join(x for x in (s.locator, f"doi {s.doi}" if s.doi else None, s.url) if x)
            lines.append(f"  - Source: {_flat(s.citation)}" + (f" ({extra})" if extra else ""))
    return lines


def _analysis_lines(analysis_id: str, hypothesis_id: str, run: AnalysisRun, here: _Here) -> list[str]:
    lines = [f"- **{analysis_id}**, a script written for {here.hyp(hypothesis_id)}: {_said(run.purpose)} "
             f"**{run.outcome.value}** (exit code {run.exit_code}, {run.duration_seconds:g} s)",
             f"  - {here.artifact(run.script, 'Script')}"
             + (f", {here.artifact(run.output, 'what it printed')}" if run.output else "")
             + "".join(f", {here.artifact(ref, name)}" for name, ref in run.products.items())]
    lines += [f"  - Staged from {i.calculation_id}: {', '.join(i.files)}" for i in run.inputs]

    def check(c, indent: str) -> str:
        return (f"{indent}- Script check {c.name} {'passed' if c.passed else 'FAILED'}"
                + (f": {_flat(c.detail)}" if c.detail else ""))

    lines += [check(c, "  ") for c in run.checks]
    for r in run.results:
        lines.append(f"  - Result {r.name} = {_quantity(r.value)}" + (f": {_flat(r.description)}" if r.description
                                                                       else ""))
        lines += [check(c, "    ") for c in r.checks]
    lines += [f"  - Note: {_flat(problem)}" for problem in run.problems]
    return lines


def _assessment_lines(hyp: HypothesisRecord, event: RecordedEvent, p: AssessmentRecorded, here: _Here,
                      name: str) -> list[str]:
    index = next(i for i, a in enumerate(hyp.assessments) if a.recorded_at == event.stream_version)
    before = hyp.assessments[index - 1].assessment if index else Assessment.NOT_TESTED
    cited = ", ".join(p.evidence_ids) or "no evidence"
    lines = [f"- {name}: {before.value} → **{p.assessment.value}** ({p.completeness.value.lower()}), "
             f"by {_actor(event.actor)}, citing {cited}",
             f"  - Rationale: {_said(p.rationale)}"]
    lines += [f"  - Limitation: {_flat(x)}" for x in p.limitations]
    return lines


def _run_lines(event: RecordedEvent, p: AgentRunRecorded) -> list[str]:
    model = ", ".join(p.served_models) or p.requested_model
    task = f"task {p.task_id}; " if p.task_id else ""
    if p.role == "chief_scientist":
        task = "review of the agents' reports; " + task
    dropped = (f"; {p.dropped_reasoning_blocks} earlier reasoning block{'s' if p.dropped_reasoning_blocks > 1 else ''} "
               "left out because the conversation changed") if p.dropped_reasoning_blocks else ""
    web = f"; {_web_use(p)}" if p.web_searches or p.web_fetches else ""
    lines = [f"- **{_actor(event.actor)}** ({p.role.replace('_', ' ')}; {task}{p.outcome}; {p.turns} turns; "
             f"{model}{web}; {_when(event)}{dropped}):"]
    return lines + (_quote(p.report, "  ") if p.report and p.report.strip() else ["  (no report)"])


_REPORTS_NOTE = ("In the agents' own words. A report is not evidence: its values count only where they appear in "
                 "recorded evidence.")


def _web_use(p: AgentRunRecorded) -> str:
    return f"{p.web_searches} web search{'es' if p.web_searches != 1 else ''}, {p.web_fetches} page" \
        f"{'s' if p.web_fetches != 1 else ''} fetched"


def _workbench_change(p: WorkbenchPolicyChanged) -> str:
    return "Agents' workbench turned " + ("on" if p.enabled else "off")


def _web_change(p: RetrievalPolicyChanged) -> str:
    setting = {"disabled": "off", "enabled": "on"}.get(p.policy.value, "only " + ", ".join(p.allowed_domains))
    return f"Web access for the Chief Scientist and the hypothesis agents set {setting}"


def _budget_change(max_usd: float | None) -> str:
    return "Budget removed" if max_usd is None else f"Budget set to {dollars(max_usd)}"


def _tokens(p: AgentRunRecorded) -> int:
    return p.input_tokens + p.output_tokens + p.cache_read_tokens + p.cache_write_tokens


# -- round pages ------------------------------------------------------------------------------

@dataclass
class _Round:
    number: int
    events: list[RecordedEvent] = field(default_factory=list)

    def of(self, kind: type) -> list[tuple[RecordedEvent, object]]:
        return [(e, e.payload) for e in self.events if isinstance(e.payload, kind)]


def _round_page(rnd: _Round, state: InvestigationState, next_start: RecordedEvent | None) -> str:
    first, last = rnd.events[0], rnd.events[-1]
    if rnd.number == 0:
        opening = ("Before the first round: the investigation is opened, the observables are aligned and hypotheses "
                   "are proposed.")
    else:
        opening = f"Started by {_actor(first.actor)}: {_said(first.payload.reason)}"
    meta = (f"Investigation `{state.investigation_id}` · events #{first.stream_version}–#{last.stream_version} · "
            f"{_when(first)} to {_when(last)} · [notebook index](index.md)")
    bodies = (_hypothesized(rnd, state), _why(rnd, state), _predicted(rnd), _tested(rnd, state),
              _observed(rnd, state), _assessed(rnd, state), _unknown(rnd, state), _next(rnd, state, next_start),
              _messages(rnd), _reports(rnd))
    blocks = [f"# {_round_name(rnd.number)}: {state.spec.title}", meta, opening, _NOTE]
    blocks += [_section(title, body) for title, body in zip(ROUND_SECTIONS, bodies, strict=True)]
    return "\n\n".join(blocks) + "\n"


def _hypothesized(rnd: _Round, state: InvestigationState) -> list[str]:
    lines, new = [], set()
    for event, p in rnd.of(HypothesisProposed):
        new.add(p.hypothesis_id)
        lines.append(f"- **{ROOT.hyp(p.hypothesis_id)}: {_flat(p.spec.title)}**, proposed by {_actor(event.actor)}. "
                     f"{_flat(p.spec.statement)}")
        lines += _spec_details(p.spec, "  ")
    if not lines:
        lines.append("No new hypotheses this round.")
    others = [h for h in state.hypotheses.values() if h.hypothesis_id not in new and h.status not in _CLOSED]
    if others:
        lines += ["", "Also open: " + ", ".join(f"{ROOT.hyp(h.hypothesis_id)} ({h.status.value})" for h in others)
                  + "."]
    return lines


def _priority_changes(state: InvestigationState, version: int) -> str:
    before, after = state.plans[version - 2].plan, state.plans[version - 1].plan
    changes = [f"{h} {before.priority(h).value} → {after.priority(h).value}"
               for h in after.hypothesis_priorities if before.priority(h) is not after.priority(h)]
    if before.objective != after.objective:
        changes.insert(0, f"objective now {_said(after.objective)}")
    return "; ".join(changes)


def _why(rnd: _Round, state: InvestigationState) -> list[str]:
    lines = []
    for event in rnd.events:
        p, who = event.payload, _actor(event.actor)
        if isinstance(p, HypothesisProposed):
            lines.append(f"- {ROOT.hyp(p.hypothesis_id)}, rationale by {who}: {_said(p.spec.rationale)}")
        elif isinstance(p, InvestigationStatusChanged):
            lines.append(f"- Investigation moved to {p.status.value} by {who}: {_said(p.reason)}")
        elif isinstance(p, ClosingProposed):
            lines.append(f"- Closing as {p.status.value} proposed by {who} ({p.closing_id}): {_said(p.reason)}")
        elif isinstance(p, ObservableAlignmentRecorded):
            label = f"Observables judged {p.alignment.status.value} by {who}"
            lines += _listed(label, p.alignment.rationale, quoted=True) if p.alignment.rationale else [f"- {label}"]
            if p.alignment.forward_model:
                lines.append(f"  - Forward model: {_said(p.alignment.forward_model)}")
        elif isinstance(p, PlanRevised):
            prompted = f", prompted by {p.intervention_id}" if p.intervention_id else ""
            lines.append(f"- Plan v{p.version} by {who}{prompted}: {_said(p.reason)}")
            changes = _priority_changes(state, p.version)
            if changes:
                lines.append(f"  - Changes: {changes}")
        elif isinstance(p, HypothesisStatusChanged) and event.actor.kind in (ActorKind.CHIEF_SCIENTIST,
                                                                             ActorKind.USER):
            lines.append(f"- {ROOT.hyp(p.hypothesis_id)} moved to {p.status.value} by {who}: {_said(p.reason)}")
        elif isinstance(p, AgentsProposed):
            following = f", carrying out {p.recommendation_id}" if p.recommendation_id else ""
            lines.append(f"- Agents proposed by {who} ({p.proposal_id}{following}): {_said(p.rationale)}")
            lines += [f"  - {ROOT.hyp(a.hypothesis_id)}: {_flat(a.role)}; first task: {_said(a.brief)}"
                      for a in p.agents]
        elif isinstance(p, AgentsStarted):
            how = (f"by the approver ({p.authorization_id}), in full automation"
                   if event.actor.kind is ActorKind.APPROVER else
                   f"by the researcher's {p.authorization_id}" if p.authorization_id else
                   "after the researcher's review window passed without a reply")
            lines.append(f"- Agents of {p.proposal_id} started {how}: "
                         + ", ".join(f"{a} on {ROOT.hyp(h)}" for h, a in p.agent_ids.items()))
        elif isinstance(p, AgentRetired):
            lines.append(f"- {p.agent_id} retired by {who}: {_said(p.reason)}")
        elif isinstance(p, ProposalWithdrawn):
            lines.append(f"- Proposal {p.proposal_id} withdrawn by {who}: {_said(p.reason)}")
        elif isinstance(p, ModeChanged):
            lines.append(f"- {_mode_change(p)}, by {who}: {_said(p.reason)}")
        elif isinstance(p, RetrievalPolicyChanged):
            lines.append(f"- {_web_change(p)} by {who} ({p.intervention_id})")
        elif isinstance(p, WorkbenchPolicyChanged):
            lines.append(f"- {_workbench_change(p)} by {who} ({p.intervention_id})")
        elif isinstance(p, BudgetChanged):
            lines.append(f"- {_budget_change(p.max_usd)} by {who} ({p.intervention_id})")
        elif isinstance(p, MaterialAttached):
            note = f": {_said(p.note)}" if p.note else ""
            lines.append(f"- {who} attached {p.material_id}, {ROOT.artifact(p.artifact, _flat(p.name))} "
                         f"({p.intervention_id}){note}")
        elif isinstance(p, ApproverDecision):
            lines.append(f"- The approver {_decided(p)} ({p.decision_id}): {_said(p.reason)}")
    return lines or ["No decisions were recorded this round."]


def _mode_change(p: ModeChanged) -> str:
    if p.mode is InvestigationMode.HUMAN_IN_THE_LOOP:
        return "Back to human-in-the-loop"
    return f"Full automation, within {p.max_core_hours:g} core-hours and {dollars(p.max_usd)} of model use"


def _decided(p: ApproverDecision) -> str:
    verdict = "approved" if p.approved else "declined"
    return (f"{verdict} starting the agents of {p.target_id}" if p.action is PermissionAction.START_AGENTS else
            f"{verdict} submitting {p.target_id}")


def _prediction_lines(event: RecordedEvent, p: PredictionRegistered, here: _Here) -> list[str]:
    pr = p.prediction
    lines = [f"- **{p.prediction_id}** for {here.hyp(p.hypothesis_id)}, registered by {_actor(event.actor)} "
             f"(event #{event.stream_version}, round {event.round}): {_said(pr.statement)}",
             f"  - Expected: {_said(pr.expected_outcome)}",
             f"  - Falsified if: {_said(pr.falsified_if)}",
             f"  - Observable: {_observable(pr.observable)}"]
    if pr.conditions:
        lines.append(f"  - Conditions: {_settings(pr.conditions)}")
    if pr.calculation_ids:
        lines.append(f"  - About calculations: {', '.join(pr.calculation_ids)}")
    return lines


def _predicted(rnd: _Round) -> list[str]:
    lines = [line for event, p in rnd.of(PredictionRegistered) for line in _prediction_lines(event, p, ROOT)]
    if not lines:
        return ["No predictions were registered this round."]
    return ["Predictions are recorded before the outcomes they concern and cannot be changed afterwards.", "", *lines]


def _tested(rnd: _Round, state: InvestigationState) -> list[str]:
    lines = []
    for event in rnd.events:
        p = event.payload
        if isinstance(p, CalculationRequested):
            lines += _calculation_lines(state.calculations[p.calculation_id], p.rationale, event.actor, ROOT,
                                        "At the end of the round")
        elif isinstance(p, RunRegistered):
            lines += _calculation_lines(state.calculations[p.calculation_id], p.description, p.requested_by, ROOT,
                                        "Registered")
        elif isinstance(p, WorkbenchRunRecorded):
            lines += _workbench_lines(state.workbench_runs[p.run_id], ROOT)
        elif isinstance(p, CalculationLinked):
            lines.append(f"- {ROOT.hyp(p.hypothesis_id)} reuses **{p.calculation_id}**: {_said(p.rationale)}")
        elif isinstance(p, CalculationStateChanged) and p.state is CalculationState.SUBMITTED:
            by = f", authorized by {p.authorization_id}" if p.authorization_id else ""
            lines.append(f"- **{p.calculation_id}** submitted to the scheduler as job {p.scheduler_job_id}{by}")
        elif isinstance(p, CalculationStateChanged) and p.state is CalculationState.CANCELLED:
            by = f", authorized by {p.authorization_id}" if p.authorization_id else ""
            lines.append(f"- **{p.calculation_id}** cancelled{by}: {_said(p.reason or '')}")
        elif isinstance(p, SubmissionRefused):
            lines.append(f"- **{p.calculation_id}** refused by the scheduler, to be tried again: {_said(p.reason)}")
        elif isinstance(p, CalculationStateChanged) and p.state is CalculationState.SUBMISSION_FAILED:
            lines.append(f"- **{p.calculation_id}** refused by the scheduler; its agent is to request it again: "
                         f"{_said(p.reason or '')}")
        elif isinstance(p, InputReviewRecorded):
            lines.append(f"- **{p.calculation_id}** reviewed by the input reviewer ({p.review_id}), an agent's "
                         f"judgement: {p.decision.value}: {_said(p.summary)}")
            lines += [f"  - {f.severity.value}: {_said(f.text)}" + (f" ({f.file}" + (f":{f.line}" if f.line else "")
                                                                    + ")" if f.file else "")
                      for f in p.findings if f.severity is not ReviewSeverity.NOTE]
        elif isinstance(p, CalculationStateChanged) and p.state is CalculationState.REJECTED:
            calc = state.calculations[p.calculation_id]
            by = ("the input reviewer" if p.authorization_id and calc.input_review is not None
                  and calc.input_review.review_id == p.authorization_id else
                  f"the researcher ({p.authorization_id})" if p.authorization_id else "the input examination")
            lines.append(f"- **{p.calculation_id}** rejected by {by}: {_said(p.reason or '')}")
    return lines or ["No calculations were requested or submitted this round."]


def _observed(rnd: _Round, state: InvestigationState) -> list[str]:
    lines = []
    for event in rnd.events:
        p = event.payload
        if isinstance(p, EvidenceRecorded):
            lines += _evidence_lines(p.evidence_id, p.evidence, state, ROOT)
        elif isinstance(p, (RunChecked, RunRegistered)):
            failed = [c for c in p.checks if not c.passed]
            checked = "registered run checked" if isinstance(p, RunRegistered) else "run checked"
            lines.append(f"- **{p.calculation_id}**: {checked}, " + (
                "healthy" if not failed else "failed " + "; ".join(
                    f"{c.name}" + (f" ({_flat(c.detail)})" if c.detail else "") for c in failed)))
            lines += [f"  - The protocol read {r.name} = {_quantity(r.value)} at {r.file} line {r.line_start}"
                      for r in p.readings]
        elif isinstance(p, AnalysisRecorded):
            lines += _analysis_lines(p.analysis_id, p.hypothesis_id, p.run, ROOT)
        elif (isinstance(p, CalculationStateChanged) and p.state is CalculationState.SCHEDULER_TERMINAL
              and p.scheduler_state is not SchedulerState.COMPLETED):
            lines.append(f"- **{p.calculation_id}**: the job ended {p.scheduler_state.value}"
                         + (f": {_said(p.reason)}" if p.reason else ""))
    return lines or ["No new evidence this round."]


def _assessed(rnd: _Round, state: InvestigationState) -> list[str]:
    lines = [line for event, p in rnd.of(AssessmentRecorded)
             for line in _assessment_lines(state.hypotheses[p.hypothesis_id], event, p, ROOT,
                                           ROOT.hyp(p.hypothesis_id))]
    lines = lines or ["No assessments were recorded this round."]
    for event, p in rnd.of(SynthesisRecorded):
        lines += ["", f"Comparison across hypotheses by {_actor(event.actor)} ({p.synthesis_id}, event "
                      f"#{event.stream_version}), in its words; values are those of the evidence it cites: "
                      f"{_said(p.summary)}"]
        for h in p.hypotheses:
            lines.append(f"- {ROOT.hyp(h.hypothesis_id)}: {_said(h.standing)}"
                         + (f" Accounts for: {_said(h.accounts_for)}" if h.accounts_for else ""))
            for evidence_id in h.evidence_ids:
                ev = state.evidence[evidence_id].evidence
                lines.append(f"  - {evidence_id}: {_flat(ev.observable.name)} = {_quantity(ev.value)} "
                             f"({ev.quality.value})")
        lines += [f"- Interaction: {_said(text)}" for text in p.interactions]
        lines.append(f"- Unexplained: {_said(p.residual)}")
        lines += [f"- Cannot be separated: {_said(text)}" for text in p.not_separable]
        if p.new_hypotheses:
            lines.append("- New hypotheses from this comparison: " + ", ".join(ROOT.hyp(h) for h in p.new_hypotheses))
    return lines


def _unknown(rnd: _Round, state: InvestigationState) -> list[str]:
    lines = []
    alignment = state.alignment
    if alignment.status is not AlignmentStatus.ALIGNED:
        label = f"The compared observables are {alignment.status.value}, so no numerical discrepancy is defined"
        lines += _listed(label, alignment.rationale) if alignment.rationale else [f"- {label}."]
    for hyp in state.hypotheses.values():
        if hyp.status in _CLOSED:
            continue
        last = hyp.assessments[-1] if hyp.assessments else None
        if last is None:
            lines.append(f"- {ROOT.hyp(hyp.hypothesis_id)} ({hyp.status.value}) has not been assessed.")
        elif last.assessment in _UNSETTLED or last.completeness is Completeness.PARTIAL:
            lines.append(f"- {ROOT.hyp(hyp.hypothesis_id)} is {last.assessment.value} "
                         f"({last.completeness.value.lower()}).")
    for _, p in rnd.of(EvidenceRecorded):
        if p.evidence.quality is not EvidenceQuality.VALIDATED:
            failed = [c.name for c in p.evidence.quality_checks if not c.passed]
            lines.append(f"- {p.evidence_id} is {p.evidence.quality.value}"
                         + (f"; failed checks: {', '.join(failed)}." if failed else "."))
    return lines or ["Nothing is recorded as open: the observables are aligned and every open hypothesis has a final "
                     "SUPPORTED or CHALLENGED assessment."]


def _recommended(event: RecordedEvent, p: NextActionsRecommended) -> list[str]:
    taken = ", ".join(p.chosen) if p.chosen else "none for now"
    lines = [f"- Next steps recommended by {_actor(event.actor)} ({p.recommendation_id}). Taking: {taken}. Why: "
             f"{_said(p.reason)}"]
    for c in p.candidates:
        mark = "chosen; " if c.candidate_id in p.chosen else ""
        bears = f" on {', '.join(ROOT.hyp(h) for h in c.hypothesis_ids)}" if c.hypothesis_ids else ""
        lines.append(f"  - **{c.candidate_id}** ({mark}{c.kind.value}, {c.priority.value} priority){bears}: "
                     f"{_said(c.action)}")
        lines += [f"    - If {ROOT.hyp(o.hypothesis_id)} holds: {_said(o.outcome)}" for o in c.expected_outcomes]
        lines += [f"    - Informative because: {_said(c.rationale)}",
                  f"    - Cost: {_said(c.cost)}; feasibility: {_said(c.feasibility)}",
                  f"    - Priority because: {_said(c.priority_reason)}"]
    return lines


def _next(rnd: _Round, state: InvestigationState, next_start: RecordedEvent | None) -> list[str]:
    lines = [line for event, p in rnd.of(NextActionsRecommended) for line in _recommended(event, p)]
    if next_start is not None:
        n = next_start.payload.round
        lines.append(f"- [{_round_name(n)}]({round_file(n)}) started next, by {_actor(next_start.actor)}: "
                     f"{_said(next_start.payload.reason)}")
    proposal = state.pending_proposal
    if proposal is not None:
        status = (f"held since the researcher's {proposal.feedback_ids[0]}" if proposal.feedback_ids else
                  f"starts at {_clock(review_deadline(state, proposal))} unless the researcher replies")
        lines.append(f"- Agents proposed ({proposal.proposal_id}), {status}: "
                     + ", ".join(f"{_flat(a.role)} on {ROOT.hyp(a.hypothesis_id)}" for a in proposal.agents))
    for task in state.pending_tasks:
        if task.role == "chief_scientist":
            who = "the Chief Scientist's review of the agents' reports"
        elif task.role == "evidence_judge":
            who = f"the Evidence Judge on {ROOT.hyp(task.hypothesis_id)}"
        else:
            who = f"{task.agent_id} on {ROOT.hyp(task.hypothesis_id)}"
        following = f", carrying out {task.recommendation_id}" if task.recommendation_id else ""
        lines.append(f"- Queued for {who}, by {task.queued_by}{following}: {_said(task.instruction)}")
    lines += [f"- Waiting for the researcher: `{command}` ({note})" for command, note in approval_requests(state)]
    lines += [f"- {c.calculation_id} not finished: {_calc_status(c)}" for c in state.calculations.values()
              if c.state not in _FINISHED]
    if not lines:
        lines.append("- Nothing is queued, running or waiting for the researcher.")
    plan = state.plan
    priorities = ", ".join(f"{h} {p.value}" for h, p in plan.hypothesis_priorities.items()) or "all normal"
    lines.append(f"- Plan v{state.plan_version}: {_said(plan.objective)} Priorities: {priorities}.")
    lines += [f"  - Note: {_flat(note)}" for note in plan.notes]
    return lines


def _messages(rnd: _Round) -> list[str]:
    lines = []
    for event, p in rnd.of(UserInterventionRecorded):
        grants = "; ".join(f"authorizes {a.action.value.replace('_', ' ')} of {', '.join(a.target_ids)}"
                           for a in p.authorizations)
        lines.append(f"- **{p.intervention_id}**, {_actor(event.actor)}, {_when(event)}"
                     + (f" ({grants})" if grants else "") + ":")
        lines += _quote(p.text, "  ")
    return lines or ["None this round."]


def _paragraphs(*blocks: list[str]) -> list[str]:
    """Non-empty blocks of lines, separated by blank lines."""
    lines: list[str] = []
    for block in (b for b in blocks if b):
        lines += ([""] if lines else []) + block
    return lines


def _reports(rnd: _Round) -> list[str]:
    runs = rnd.of(AgentRunRecorded)
    reports = [line for event, p in runs if p.role != "chief_scientist" or p.task_id
               for line in _run_lines(event, p)]
    chief = sum(1 for _, p in runs if p.role == "chief_scientist" and not p.task_id)
    tokens = sum(_tokens(p) for _, p in runs)
    lines = _paragraphs(
        [_REPORTS_NOTE] if reports else [], reports,
        [f"The Chief Scientist ran {chief} time{'s' if chief > 1 else ''} this round, answering the researcher; the "
         "conversation is kept in the agent transcripts."] if chief else [],
        [f"Model tokens processed this round: {tokens:,}."] if tokens else [])
    return lines or ["No agent ran this round."]


# -- hypothesis pages -------------------------------------------------------------------------

def _hypothesis_page(hyp: HypothesisRecord, state: InvestigationState, events: Sequence[RecordedEvent]) -> str:
    proposed = events[hyp.proposed_at - 1]
    spec = hyp.spec
    header = (f"[Investigation `{state.investigation_id}`](../index.md) · status {hyp.status.value} · assessment "
              f"{hyp.assessment.value} · priority {state.plan.priority(hyp.hypothesis_id).value} · proposed by "
              f"{_actor(proposed.actor)} in {_round_name(proposed.round).lower()}, {_when(proposed)}")
    statement = [_flat(spec.statement), "", f"- Rationale: {_said(spec.rationale)}", *_spec_details(spec)]
    agents = []
    for agent in (a for a in state.agents.values() if a.hypothesis_id == hyp.hypothesis_id):
        how = (f"by the approver ({agent.authorization_id}), in full automation"
               if agent.started_by.kind is ActorKind.APPROVER else
               f"by the researcher's {agent.authorization_id}" if agent.authorization_id else
               "after the review window passed without a reply")
        agents.append(f"- **{agent.agent_id}**, {_flat(agent.spec.role)}: started {how} (event #{agent.started_at}, "
                      f"proposal {agent.proposal_id})" + (f"; retired: {_said(agent.retire_reason)}"
                                                         if not agent.active else ""))
        agents.append(f"  - First task: {_said(agent.spec.brief)}")

    predictions = []
    for pid in hyp.prediction_ids:
        event = events[state.predictions[pid].registered_at - 1]
        predictions += _prediction_lines(event, event.payload, SUB)
    calculations = []
    for cid in hyp.calculation_ids:
        calc = state.calculations[cid]
        rationale, requester = _origin(calc, events)
        for event in events:  # a linked calculation was requested for another hypothesis; show why it is used here
            if (isinstance(event.payload, CalculationLinked) and event.payload.calculation_id == cid
                    and event.payload.hypothesis_id == hyp.hypothesis_id):
                rationale, requester = event.payload.rationale, event.actor
        calculations += _calculation_lines(calc, rationale, requester, SUB, "Now")
    evidence = [line for record in hypothesis_evidence(state, hyp.hypothesis_id)
                for line in _evidence_lines(record.evidence_id, record.evidence, state, SUB)]
    analyses = [line for record in state.analyses.values() if record.hypothesis_id == hyp.hypothesis_id
                for line in _analysis_lines(record.analysis_id, record.hypothesis_id, record.run, SUB)]
    workbench = [line for record in state.workbench_runs.values() if record.hypothesis_id == hyp.hypothesis_id
                 for line in _workbench_lines(record, SUB)]

    assessments = []
    for entry in hyp.assessments:
        event = events[entry.recorded_at - 1]
        assessments += _assessment_lines(hyp, event, event.payload, SUB,
                                         f"{_round_name(event.round)}, event #{event.stream_version}")

    reports = []
    for run in state.agent_runs:
        task = state.tasks.get(run.run.task_id) if run.run.task_id else None
        if task is not None and task.hypothesis_id == hyp.hypothesis_id:
            reports += _run_lines(events[run.recorded_at - 1], run.run)

    blocks = [f"# Hypothesis {hyp.hypothesis_id}: {_flat(spec.title)}", header, _NOTE,
              _section("Statement", statement),
              _section("Agent", agents or ["None started yet."]),
              _section("Predictions", predictions or ["None registered yet."]),
              _section("Calculations", calculations or ["None requested yet."]),
              _section("Analyses", analyses or ["None run yet."]),
              _section("Workbench commands (context, not evidence)", workbench or ["None run yet."]),
              _section("Evidence", evidence or ["None recorded yet."]),
              _section("Assessments", assessments or ["Not assessed yet."]),
              _section("Agent reports", [_REPORTS_NOTE, "", *reports] if reports else ["None yet."])]
    return "\n\n".join(blocks) + "\n"


# -- index and timeline -----------------------------------------------------------------------

def _provenance(item) -> list[str]:
    """One simulation or measurement as described when the investigation was opened."""
    data = item.model_dump(mode="json", exclude_defaults=True, exclude_none=True,
                           exclude={"label", "sources", "quantity", "observable", "temperature", "pressure"})
    parts = []
    if getattr(item, "observable", None) is not None:
        parts.append(f"observable: {_observable(item.observable)}")
    if item.quantity is not None:
        parts.append(f"value: {_quantity(item.quantity)}")
    for name in ("temperature", "pressure"):
        if getattr(item, name, None) is not None:
            parts.append(f"{name}: {_quantity(getattr(item, name))}")
    for key, value in data.items():
        shown = value if isinstance(value, str) else _settings(value) if isinstance(value, dict) else json.dumps(value)
        parts.append(f"{key.replace('_', ' ')}: {_flat(shown)}")
    parts += [f"source: {_flat(s.citation) if isinstance(s, ExternalSource) else s.path}" for s in item.sources]
    return [f"- **{_flat(item.label)}**"] + [f"  - {part}" for part in parts or ["no details recorded"]]


def _index_page(state: InvestigationState, events: Sequence[RecordedEvent], rounds: list[_Round]) -> str:
    spec, last = state.spec, events[-1]
    header = (f"Investigation `{state.investigation_id}` · {state.status.value} · {_round_name(state.round).lower()} · "
              f"plan v{state.plan_version} · compute profile {spec.constraints.compute_profile or 'none'}")
    generated = f"Generated from events #1–#{last.stream_version}; the latest was recorded {_when(last)}."

    a = state.alignment
    discrepancy = state.discrepancy()
    quantities = ["Simulations:", "", *[line for s in spec.simulations for line in _provenance(s)], "",
                  "Measurements:", "", *[line for e in spec.experiments for line in _provenance(e)], "",
                  f"Alignment: **{a.status.value}**", "",
                  f"- Simulated: {_observable(a.simulated)}", f"- Measured: {_observable(a.measured)}"]
    quantities += _listed("Why", a.rationale) if a.rationale else []
    if a.forward_model:
        quantities.append(f"- Forward model: {_flat(a.forward_model)}")
    quantities += ["", f"Discrepancy (simulated − measured): {_quantity(discrepancy.difference)}" if
                   discrepancy.comparable else "Discrepancy: not defined; " + "; ".join(discrepancy.reasons) + "."]

    current = state.plans[-1]
    priorities = ", ".join(f"{h} {p.value}" for h, p in current.plan.hypothesis_priorities.items()) or "all normal"
    plan = [f"Plan v{current.version}: {_said(current.plan.objective)} Priorities: {priorities}."]
    plan += [f"- Note: {_flat(n)}" for n in current.plan.notes]
    if len(state.plans) > 1:
        plan += ["", "Versions:", ""]
        for record in state.plans:
            prompted = f", prompted by {record.intervention_id}" if record.intervention_id else ""
            plan.append(f"- v{record.version} (event #{record.recorded_at}{prompted}): {_said(record.reason)}")

    hypotheses = ["| Hypothesis | Status | Assessment | Priority | Title |", "|---|---|---|---|---|"]
    hypotheses += [f"| {ROOT.hyp(h.hypothesis_id)} | {h.status.value} | {h.assessment.value} | "
                   f"{state.plan.priority(h.hypothesis_id).value} | {_cell(h.spec.title)} |"
                   for h in state.hypotheses.values()]
    if not state.hypotheses:
        hypotheses = ["None yet."]

    calculations = ["Job state comes from the scheduler; evidence quality comes from the validator's checks. A "
                    "completed job is not validated evidence.", "",
                    "| Calculation | For | Estimand | Inputs | Job | Job state | Run check | Evidence |",
                    "|---|---|---|---|---|---|---|---|"]
    for c in state.calculations.values():
        inputs = "registered" if c.registered else c.examination.verdict.value if c.examination else "not examined"
        evidence = "; ".join(f"{eid}: {state.evidence[eid].evidence.quality.value}, "
                             f"{state.evidence[eid].evidence.observable.name} = "
                             f"{_quantity(state.evidence[eid].evidence.value)}" for eid in c.evidence_ids) or "none"
        check = _run_check(c.run_check) if c.run_check is not None else "-"
        calculations.append(
            f"| {c.calculation_id} | {_users(c, ROOT)} | {_cell(c.task.estimand)} | "
            f"{inputs} | {c.scheduler_job_id or '-'} | {c.scheduler_state.value if c.scheduler_state else '-'} | "
            f"{_cell(check)} | {_cell(evidence)} |")
    if not state.calculations:
        calculations = ["None yet."]

    round_links = [f"- [{_round_name(r.number)}]({round_file(r.number)}): events #{r.events[0].stream_version}–"
                   f"#{r.events[-1].stream_version}, {_when(r.events[0])} to {_when(r.events[-1])}" for r in rounds]
    waiting = [f"- `{command}` ({note})" for command, note in approval_requests(state)] or ["Nothing."]

    usage, runs, searches, fetches, spent = Counter(), Counter(), Counter(), Counter(), Counter()
    for run in state.agent_runs:
        usage[run.role] += _tokens(run.run)
        runs[run.role] += 1
        searches[run.role] += run.run.web_searches
        fetches[run.role] += run.run.web_fetches
        spent[run.role] += run.run.cost_usd or 0
    model_usage = [f"- {role.replace('_', ' ')}: {runs[role]} run{'s' if runs[role] > 1 else ''}, "
                   f"{dollars(spent[role])}, {usage[role]:,} tokens"
                   + (f", {searches[role]} web searches, {fetches[role]} pages fetched"
                      if searches[role] or fetches[role] else "") for role in runs]
    budget = state.budget_usd
    model_usage.append(f"- Total: {dollars(state.model_cost)}" + (f" of a {dollars(budget)} budget" if budget else "")
                       + f", {state.model_tokens_used:,} tokens"
                       + (f"; {state.unpriced_runs} runs of models with unknown prices are not counted"
                          if state.unpriced_runs else ""))
    model_usage += [f"- {_budget_change(b.max_usd)} by the researcher's {b.intervention_id} (event #{b.recorded_at})"
                    for b in state.budget_changes]

    agents = ["Each hypothesis has at most one agent at a time; each agent records only for its own hypothesis and "
              "can read the others' work.", "",
              "| Agent | Hypothesis | Role | Started | Status | Runs | Spent |", "|---|---|---|---|---|---|---|"]
    for agent in state.agents.values():
        mine = [r for r in state.agent_runs if r.actor_id == agent.agent_id]
        started = (f"approver ({agent.authorization_id})" if agent.started_by.kind is ActorKind.APPROVER else
                   agent.authorization_id or "review window passed")
        status = "active" if agent.active else f"retired: {_short(agent.retire_reason or '', 80)}"
        limit = f" of {dollars(agent.spec.budget_usd)}" if agent.spec.budget_usd else ""
        agents.append(f"| {agent.tag} {agent.agent_id} | {ROOT.hyp(agent.hypothesis_id)} | {_cell(agent.spec.role)} | "
                      f"{started} | {_cell(status)} | {len(mine)} | {dollars(state.agent_cost(agent.agent_id))}{limit} |")
    if not state.agents:
        agents = ["None started yet."]
    if state.proposals:
        agents += ["", "Proposals:", ""]
        for proposal in state.proposals.values():
            outcome = {"pending": "waiting for the researcher's review", "started": "started",
                       "superseded": "replaced by a later proposal", "withdrawn": "withdrawn"}[proposal.status.value]
            replies = f"; researcher replied: {', '.join(proposal.feedback_ids)}" if proposal.feedback_ids else ""
            agents.append(f"- {proposal.proposal_id} (event #{proposal.recorded_at}, {outcome}{replies}): "
                          + ", ".join(f"{_flat(a.role)} on {ROOT.hyp(a.hypothesis_id)}" for a in proposal.agents))

    sources = [f"Web access for the Chief Scientist and the hypothesis agents: **{web_summary(state.retrieval)}**. "
               "The Evidence Judge and the approver never use the web. What agents read on the web or in attached "
               "materials is context, not evidence.", ""]
    sources += [f"- Set {web_summary(r)} by the researcher's {r.intervention_id} (event #{r.recorded_at})"
                for r in state.retrieval_changes]
    sources += ["", f"Agents' workbench (sandboxed commands in their workspaces): **"
                    f"{'on' if state.workbench_enabled else 'off'}**. What a command prints or writes is context, "
                    "not evidence.", ""]
    sources += [f"- Turned {'on' if r.enabled else 'off'} by the researcher's {r.intervention_id} "
                f"(event #{r.recorded_at})" for r in state.workbench_changes]
    sources += ["", "Materials the researcher attached, which every agent of the investigation can view:", ""]
    sources += [f"- **{m.material_id}** {ROOT.artifact(m.artifact, _flat(m.name))} ({m.artifact.media_type or 'file'}, "
                f"{m.artifact.size_bytes:,} bytes; {m.intervention_id})" + (f": {_said(m.note)}" if m.note else "")
                for m in state.materials.values()] or ["None."]

    blocks = [f"# {spec.title}", header, generated, _NOTE,
              _section("Question", [_flat(spec.question), "", f"Target claim: {_flat(spec.target_claim)}"]),
              _section("Compared quantities", quantities),
              _section("Web access, workbench and materials", sources),
              _section("Plan", plan),
              _section("Hypotheses", hypotheses),
              _section("Agents", agents),
              _section("Calculations", calculations),
              _section("Rounds", round_links),
              _section("Waiting for the researcher", waiting),
              _section("Model usage", model_usage),
              _section("Timeline", ["[Every recorded event, in order](timeline.md)."])]
    return "\n\n".join(blocks) + "\n"


def _describe(event: RecordedEvent) -> str:
    p = event.payload
    match p:
        case InvestigationCreated():
            return f"Investigation opened: {_said(p.spec.title)}"
        case InvestigationStatusChanged():
            authorized = f", authorized by {p.authorization_id}" if p.authorization_id else ""
            return f"Investigation moved to {p.status.value}{authorized}: {_said(p.reason)}"
        case ClosingProposed():
            return f"{p.closing_id}: closing proposed as {p.status.value}: {_said(p.reason)}"
        case ObservableAlignmentRecorded():
            return f"Observables judged {p.alignment.status.value}"
        case UserInterventionRecorded():
            grants = "".join(f" [authorizes {a.action.value.replace('_', ' ')}: {', '.join(a.target_ids)}]"
                             for a in p.authorizations)
            return f"{p.intervention_id}: {_said(p.text)}{grants}"
        case PlanRevised():
            return f"Plan v{p.version}" + (f" ({p.intervention_id})" if p.intervention_id else "") + \
                f": {_said(p.reason)}"
        case RoundStarted():
            return f"Round {p.round} started: {_said(p.reason)}"
        case HypothesisProposed():
            return f"{p.hypothesis_id} proposed: {_said(p.spec.title)}"
        case HypothesisStatusChanged():
            return f"{p.hypothesis_id} moved to {p.status.value}: {_said(p.reason)}"
        case PredictionRegistered():
            return f"{p.prediction_id} registered for {p.hypothesis_id}: {_said(p.prediction.statement)}"
        case CalculationRequested():
            return f"{p.calculation_id} requested for {p.hypothesis_id}: {_said(p.task.estimand)}"
        case CalculationLinked():
            return f"{p.hypothesis_id} linked to {p.calculation_id}"
        case CalculationInputsExamined():
            return f"{p.calculation_id} inputs examined: {p.examination.verdict.value}"
        case RunChecked():
            return f"Run of {p.calculation_id} checked: {_run_check(p)}"
        case RunRegistered():
            users = p.hypothesis_id or "every agent"
            return (f"{p.calculation_id} registered for {users} from {p.source}, by {p.requested_by.id}: "
                    f"{_run_check(p.as_run_check())}")
        case AnalysisRecorded():
            results = ", ".join(f"{r.name} = {_quantity(r.value)}" for r in p.run.results)
            return f"{p.analysis_id} for {p.hypothesis_id}: {p.run.outcome.value}" + (f"; {results}" if results else "")
        case CalculationStateChanged():
            extra = "".join(x for x in (
                f" ({p.scheduler_state.value})" if p.scheduler_state else "",
                f" as job {p.scheduler_job_id}" if p.scheduler_job_id else "",
                f", authorized by {p.authorization_id}" if p.authorization_id else "",
                f": {_said(p.reason)}" if p.reason else ""))
            return f"{p.calculation_id} moved to {p.state.value}{extra}"
        case SchedulerNoteRecorded():
            return f"{p.calculation_id}: {_said(p.note)}"
        case SubmissionRefused():
            return f"{p.calculation_id}: the scheduler refused the submission, to be tried again: {_said(p.reason)}"
        case EvidenceRecorded():
            ev = p.evidence
            return f"{p.evidence_id} recorded: {ev.quality.value}, {ev.observable.name} = {_quantity(ev.value)}"
        case AssessmentRecorded():
            return (f"{p.hypothesis_id} assessed {p.assessment.value} ({p.completeness.value.lower()}), citing "
                    f"{', '.join(p.evidence_ids) or 'no evidence'}")
        case AgentTaskQueued():
            if p.role == "chief_scientist":
                return f"{p.task_id} queued for the Chief Scientist's review of the agents' reports"
            who = p.agent_id or f"the {p.role.replace('_', ' ')}"
            following = f", carrying out {p.recommendation_id}" if p.recommendation_id else ""
            return f"{p.task_id} queued for {who} on {p.hypothesis_id}{following}"
        case AgentsProposed():
            following = f" (carrying out {p.recommendation_id})" if p.recommendation_id else ""
            return (f"{p.proposal_id}: agents proposed{following}: "
                    + ", ".join(f"{a.role} on {a.hypothesis_id}" for a in p.agents))
        case ModeChanged():
            return f"{_mode_change(p)}: {_said(p.reason)}"
        case RetrievalPolicyChanged():
            return f"{_web_change(p)} ({p.intervention_id})"
        case WorkbenchPolicyChanged():
            return f"{_workbench_change(p)} ({p.intervention_id})"
        case WorkbenchRunRecorded():
            return (f"{p.run_id} for {p.hypothesis_id}: {_code(_short(p.run.command, 120))} "
                    f"{p.run.outcome.value} on {p.run.host}")
        case BudgetChanged():
            return f"{_budget_change(p.max_usd)} ({p.intervention_id})"
        case MaterialAttached():
            return f"{p.material_id} attached ({p.intervention_id}): {_said(p.name)}"
        case ApproverDecision():
            return f"{p.decision_id}: the approver {_decided(p)}: {_said(p.reason)}"
        case SynthesisRecorded():
            return f"{p.synthesis_id}: {len(p.hypotheses)} hypotheses compared: {_said(p.summary)}"
        case NextActionsRecommended():
            taken = ", ".join(p.chosen) or "none for now"
            return (f"{p.recommendation_id}: {len(p.candidates)} next steps compared; taking {taken}: "
                    f"{_said(p.reason)}")
        case AgentsStarted():
            how = f"by {p.authorization_id}" if p.authorization_id else "after the review window"
            return f"{p.proposal_id} started {how}: " + ", ".join(p.agent_ids.values())
        case AgentRetired():
            return f"{p.agent_id} retired: {_said(p.reason)}"
        case ProposalWithdrawn():
            return f"{p.proposal_id} withdrawn: {_said(p.reason)}"
        case AgentRunRecorded():
            web = f"; {_web_use(p)}" if p.web_searches or p.web_fetches else ""
            return (f"{p.role.replace('_', ' ')} run ended {p.outcome} after {p.turns} turns ({_tokens(p):,} "
                    f"tokens{web})")
        case InputReviewRecorded():
            return f"{p.review_id}: the input reviewer judged {p.calculation_id} {p.decision.value}: {_said(p.summary)}"
    return p.type


def _timeline_page(state: InvestigationState, events: Sequence[RecordedEvent]) -> str:
    lines = [f"- #{e.stream_version} · {_when(e)} · {_round_name(e.round).lower()} · plan v{e.plan_version} · "
             f"{_actor(e.actor)} · {_short(_describe(e), 240)}" for e in events]
    blocks = [f"# Timeline: {state.spec.title}",
              f"Every recorded event of investigation `{state.investigation_id}`, in order, one line each; long text "
              "is shortened. [Notebook index](index.md).", _NOTE, "\n".join(lines)]
    return "\n\n".join(blocks) + "\n"


# -- building and writing ---------------------------------------------------------------------

def render_notebook(events: Sequence[RecordedEvent]) -> dict[str, str]:
    """Every page of one investigation's notebook, keyed by path relative to its directory.

    The output depends only on the events, so rendering the same log twice gives the same
    pages. A ``round_started`` event opens the round it names.
    """
    if not events:
        return {}
    pages: dict[str, str] = {}
    rounds: list[_Round] = []
    state: InvestigationState | None = None
    for event in events:
        number = event.payload.round if isinstance(event.payload, RoundStarted) else event.round
        if rounds and rounds[-1].number != number:
            pages[round_file(rounds[-1].number)] = _round_page(rounds[-1], state, next_start=event)
        if not rounds or rounds[-1].number != number:
            rounds.append(_Round(number))
        rounds[-1].events.append(event)
        state = apply(state, event)
    pages[round_file(rounds[-1].number)] = _round_page(rounds[-1], state, next_start=None)
    for hyp in state.hypotheses.values():
        pages[f"hypotheses/{hyp.hypothesis_id}.md"] = _hypothesis_page(hyp, state, events)
    pages["timeline.md"] = _timeline_page(state, events)
    pages["index.md"] = _index_page(state, events, rounds)
    return pages


def write_pages(directory: Path, pages: dict[str, str]) -> tuple[str, ...]:
    """Write the pages whose content changed, each atomically; return their paths."""
    changed = []
    for relative, text in sorted(pages.items()):
        path = directory / relative
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.partial")
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
        changed.append(relative)
    return tuple(changed)


@dataclass(frozen=True)
class NotebookUpdate:
    directory: Path
    changed: tuple[str, ...]

    @property
    def index(self) -> Path:
        return self.directory / "index.md"

    @property
    def summary(self) -> str:
        n = len(self.changed)
        return f"{n} page{'s' if n > 1 else ''} updated" if n else "already up to date"


def update_notebook(workspace: "Workspace", investigation_id: str) -> NotebookUpdate:
    """Regenerate one investigation's notebook from its event log, writing only changed pages.

    A lock serializes writers (the dispatcher and a chat may both update it). Each writer
    reads the log only once it holds the lock, so a slower writer never leaves an older
    view behind.
    """
    directory = workspace.notebook_dir(investigation_id)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "a") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        try:
            changed = write_pages(directory, render_notebook(workspace.store.read_stream(investigation_id)))
        finally:
            fcntl.lockf(lock, fcntl.LOCK_UN)
    return NotebookUpdate(directory, changed)
