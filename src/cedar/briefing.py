"""Briefings: what an agent is told at the start of each task, built by code from the research record.

Every task of a hypothesis agent, every review by the Chief Scientist and every task of the
Evidence Judge or the approver starts a new conversation. Its first message is a briefing
generated here from the record, never by a model: it costs no tokens, leaves out nothing the
record holds about the agent's work, invents nothing, and the same record always gives the same
briefing. It names where the agent's earlier conversations and the research notebook are, so
the agent can read more history with the file tools when it wants.

Every briefing starts with the investigation as recorded: the question, the simulations and
measurements with their values and conditions, whether the compared quantities are the same
thing, the discrepancy, the plan, and the registered runs every agent shares.
"""

import hashlib
import json
from pathlib import Path

from cedar.contracts.base import ExternalSource, Observable, Quantity
from cedar.state.projection import AgentRecord, InvestigationState, hypothesis_evidence
from cedar.summaries import calculation_summary, investigation_summary

REPORTS_SHOWN = 3
"""Earlier reports of the agent shown in full, the latest last."""
REPORT_CHARS = 3000
COMMANDS_SHOWN = 10
SESSIONS_SHOWN = 5
EXCHANGES_SHOWN = 10
"""Recent messages between the researcher and the Chief Scientist carried into a fresh chat conversation."""


def session_name(*parts: str) -> str:
    """A session id from its parts, valid as a file name and at most 128 characters long."""
    name = ".".join(parts)
    if len(name) <= 128:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:16]
    return f"{name[:111]}.{digest}"


def _quantity(q: Quantity | None) -> str:
    if q is None:
        return "no value recorded"
    text = f"{q.value!r} {q.unit}"
    if q.reference:
        text += f" (reference: {q.reference})"
    for u in q.uncertainties:
        text += f" ± {u.value!r} ({u.kind.value}; {u.basis})"
    return text


def _observable(o: Observable) -> str:
    return f"'{o.name}'" + (f" [{o.unit}]" if o.unit else "") + f", defined as: {o.definition}"


def _provenance(item) -> str:
    """One simulation or measurement as described when the investigation was opened."""
    data = item.model_dump(mode="json", exclude_defaults=True, exclude_none=True,
                           exclude={"label", "sources", "quantity", "observable", "temperature", "pressure"})
    parts = []
    if getattr(item, "observable", None) is not None:
        parts.append(f"observable {_observable(item.observable)}")
    parts.append(f"value {_quantity(item.quantity)}" if item.quantity is not None else "no value recorded")
    for name in ("temperature", "pressure"):
        if getattr(item, name, None) is not None:
            parts.append(f"{name} {_quantity(getattr(item, name))}")
    for key, value in data.items():
        parts.append(f"{key.replace('_', ' ')} {value if isinstance(value, str) else json.dumps(value)}")
    parts += [f"source {s.citation if isinstance(s, ExternalSource) else s.path}" for s in item.sources]
    return f"  - {item.label}: " + "; ".join(parts)


def investigation_facts(state: InvestigationState) -> list[str]:
    """The investigation as the record holds it, for every agent."""
    spec, a = state.spec, state.alignment
    lines = [f"Investigation {state.investigation_id}: {spec.title} ({state.status.value}, round {state.round}, "
             f"plan v{state.plan_version})",
             f"- Question: {spec.question}", f"- Target claim: {spec.target_claim}", "- Simulations:"]
    lines += [_provenance(s) for s in spec.simulations] or ["  none recorded"]
    lines.append("- Measurements:")
    lines += [_provenance(e) for e in spec.experiments] or ["  none recorded"]
    lines += [f"- Compared quantities ({a.status.value}): simulated {_observable(a.simulated)}; measured "
              f"{_observable(a.measured)}"]
    lines += [f"  why: {reason}" for reason in a.rationale]
    if a.forward_model:
        lines.append(f"  forward model: {a.forward_model}")
    discrepancy = state.discrepancy() if spec.simulations and spec.experiments else None
    if discrepancy is not None and discrepancy.comparable:
        lines.append(f"- Discrepancy (simulated - measured): {_quantity(discrepancy.difference)}")
    elif discrepancy is not None:
        lines.append(f"- Discrepancy: not defined ({'; '.join(discrepancy.reasons)})")
    plan = state.plan
    lines.append(f"- Plan v{state.plan_version}: {plan.objective}")
    lines += [f"  note: {note}" for note in plan.notes]
    shared = [c for c in state.calculations.values() if c.shared]
    if shared:
        lines.append("- Runs registered for every agent, such as the calculations that disagree with the measurement:")
        lines += [f"  - {c.calculation_id} from {c.registration.source}: {c.registration.description}" for c in shared]
    if state.materials:
        lines.append("- Materials the researcher attached (view_material): " + "; ".join(
            f"{m.material_id} {m.name}" + (f" ({m.note})" if m.note else "") for m in state.materials.values()))
    return lines


def sessions_of(state: InvestigationState, actor_id: str, *, reviews: bool | None = None) -> list[str]:
    """The conversations an actor has had, oldest first; for the Chief Scientist, only its reviews (``reviews``) or
    only its chat with the researcher."""
    runs = [r for r in state.agent_runs if r.actor_id == actor_id]
    if reviews is not None:
        runs = [r for r in runs if (r.run.task_id is not None) == reviews]
    return list(dict.fromkeys(r.session_id for r in runs))


def _history(sessions: list[str], transcripts: Path, notebook: Path, pages: list[str]) -> list[str]:
    lines = ["History you may read with the file tools:"]
    if sessions:
        shown = sessions[-SESSIONS_SHOWN:]
        lines.append("- your earlier conversations, one JSON message per line, the latest last: "
                     + ", ".join(str(transcripts / f"{s}.jsonl") for s in shown)
                     + (f" (and {len(sessions) - len(shown)} older ones in {transcripts})" if len(sessions) > len(shown)
                        else ""))
    else:
        lines.append("- this is your first task: there are no earlier conversations")
    lines.append("- the research notebook, generated from the record: " + ", ".join(
        str(notebook / page) for page in pages) + f"; a page per round in {notebook}")
    return lines


def _reports(state: InvestigationState, actor_id: str, heading: str) -> list[str]:
    runs = [r for r in state.agent_runs if r.actor_id == actor_id and r.run.report][-REPORTS_SHOWN:]
    if not runs:
        return []
    lines = [heading]
    for run in runs:
        report = run.run.report.strip()
        if len(report) > REPORT_CHARS:
            report = report[:REPORT_CHARS] + " …[the rest is in the conversation file]"
        lines += [f"[{run.run.task_id or 'chat'}, run {run.run.outcome}]", report]
    return lines


def agent_briefing(state: InvestigationState, agent: AgentRecord, *, workspace: Path, runs_root: Path | None,
                   transcripts: Path, notebook: Path) -> str:
    """What a hypothesis agent is told at the start of each task: the investigation, everything recorded about its
    hypothesis and its own work, its earlier reports, the other agents, and where to read more."""
    hyp = state.hypotheses[agent.hypothesis_id]
    spec = hyp.spec
    lines = [f"Briefing from the research record (event {state.version}). Each task starts a new conversation; "
             "this briefing is what you know from earlier tasks, and the files named at the end hold the rest.", ""]
    lines += investigation_facts(state)
    lines += ["", f"Your hypothesis {hyp.hypothesis_id} [{hyp.status.value}; assessment {hyp.assessment.value}; plan "
                  f"priority {state.plan.priority(hyp.hypothesis_id).value}]: {spec.statement}",
              f"  Rationale: {spec.rationale}", f"  Falsifiers: {'; '.join(spec.falsifiers)}"]
    if spec.discriminating_observations:
        lines.append(f"  Discriminating observations: {'; '.join(spec.discriminating_observations)}")
    if spec.assumptions:
        lines.append(f"  Assumptions: {'; '.join(spec.assumptions)}")
    if hyp.assessments:
        last = hyp.assessments[-1]
        lines.append(f"  Latest assessment by the Evidence Judge ({last.completeness.value}): {last.assessment.value}, "
                     f"citing {', '.join(last.evidence_ids) or 'no evidence'}: {last.rationale}")
    lines += [f"Prediction {pid}: {state.predictions[pid].prediction.statement} Expected: "
              f"{state.predictions[pid].prediction.expected_outcome}. Falsified if: "
              f"{state.predictions[pid].prediction.falsified_if}." for pid in hyp.prediction_ids]
    if hyp.calculation_ids:
        lines.append("Your calculations:")
        for cid in hyp.calculation_ids:
            calc = state.calculations[cid]
            lines += ["  " + line for line in calculation_summary(state, calc).splitlines()]
            where = Path(calc.registration.source) if calc.registered else (
                Path(runs_root) / calc.workdir if runs_root is not None and calc.workdir else None)
            if where is not None:
                lines.append(f"    files in {where}")
    evidence = hypothesis_evidence(state, hyp.hypothesis_id)
    if evidence:
        lines.append("Your evidence:")
        for record in evidence:
            ev = record.evidence
            value = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
            source = (f"analysis {ev.analysis_id} of {', '.join(ev.calculation_ids)}" if ev.analysis_id else
                      f"reading of {ev.calculation_id}" if ev.calculation_id else ev.kind.value.replace("_", " "))
            lines.append(f"  {record.evidence_id} ({source}): {ev.observable.name} = {value}; {ev.quality.value}")
    analyses = [a for a in state.analyses.values() if a.hypothesis_id == hyp.hypothesis_id]
    if analyses:
        lines.append("Your analyses:")
        for record in analyses:
            results = ", ".join(f"{r.name} = {r.value.value!r} {r.value.unit}" for r in record.run.results)
            lines.append(f"  {record.analysis_id} ({record.run.outcome.value}): {record.run.purpose}"
                         + (f"; results {results}" if results else ""))
    commands = [w for w in state.workbench_runs.values() if w.hypothesis_id == hyp.hypothesis_id][-COMMANDS_SHOWN:]
    if commands:
        lines.append("Your latest workbench commands:")
        lines += [f"  {w.run_id} ({w.run.outcome.value}): {w.run.command}" for w in commands]
    reports = _reports(state, agent.agent_id, "Your earlier reports, the latest last:")
    if reports:
        lines += ["", *reports]
    others = [a for a in state.agents.values() if a.active and a.agent_id != agent.agent_id]
    if others:
        lines += ["", "The other agents (other_agents shows their work; you may read their files):"]
        lines += [f"  {a.tag} {a.agent_id} ({a.spec.role}) on {a.hypothesis_id} "
                  f"[{state.hypotheses[a.hypothesis_id].status.value}]: "
                  f"{state.hypotheses[a.hypothesis_id].spec.statement}" for a in others]
    lines += ["", f"Your workspace, the only place you write: {workspace}"]
    lines += _history(sessions_of(state, agent.agent_id), transcripts, notebook,
                      [f"hypotheses/{hyp.hypothesis_id}.md", "index.md"])
    return "\n".join(lines)


def review_briefing(state: InvestigationState, *, transcripts: Path, notebook: Path, actor_id: str) -> str:
    """What the Chief Scientist is told at the start of a background review, before the review material."""
    lines = [f"Briefing from the research record (event {state.version}). Each review starts a new conversation; "
             "this briefing and the material below are what you know, and the files named here hold the rest.", ""]
    lines += investigation_facts(state)
    lines += ["", "The investigation now:", investigation_summary(state)]
    reviews = [r for r in state.agent_runs if r.role == "chief_scientist" and r.run.task_id and r.run.report][-2:]
    if reviews:
        lines += ["", "What your latest background reviews concluded, the latest last:"]
        lines += [r.run.report.strip()[:REPORT_CHARS] for r in reviews]
    lines += [""] + _history(sessions_of(state, actor_id, reviews=True), transcripts, notebook,
                             ["index.md", "timeline.md"])
    return "\n".join(lines)


def role_briefing(state: InvestigationState, *, transcripts: Path, notebook: Path, actor_id: str, role: str) -> str:
    """What the Evidence Judge or the approver is told at the start of each task."""
    lines = [f"Briefing from the research record (event {state.version}). Each task starts a new conversation; your "
             f"tools show the rest of the record.", ""]
    lines += investigation_facts(state)
    if role == "evidence_judge":
        assessed = [(h, hyp.assessments[-1]) for h, hyp in state.hypotheses.items() if hyp.assessments]
        if assessed:
            lines += ["", "Your latest assessments:"]
            lines += [f"  {h}: {a.assessment.value} ({a.completeness.value}), citing "
                      f"{', '.join(a.evidence_ids) or 'no evidence'}" for h, a in assessed]
    lines += [""] + _history(sessions_of(state, actor_id), transcripts, notebook, ["index.md"])
    return "\n".join(lines)


def chat_briefing(state: InvestigationState, *, transcripts: Path, notebook: Path, sessions: list[str]) -> str:
    """What the Chief Scientist is told when its conversation with the researcher starts afresh because the earlier
    one grew too long: the investigation, its state, and the latest exchanges as recorded."""
    lines = [f"Briefing from the research record (event {state.version}). Your conversation with the researcher "
             "grew long, so it continues in a new one; this is what it held, and the files named here hold all of it.",
             ""]
    lines += investigation_facts(state)
    lines += ["", "The investigation now:", investigation_summary(state)]
    said = [(i.recorded_at, f"[researcher, {i.intervention_id}] {i.text}") for i in state.interventions.values()]
    said += [(r.recorded_at, f"[you] {r.run.report.strip()}") for r in state.agent_runs
             if r.role == "chief_scientist" and r.run.task_id is None and r.run.report]
    recent = [text for _, text in sorted(said)][-EXCHANGES_SHOWN:]
    if recent:
        lines += ["", "The latest exchanges, oldest first:"]
        lines += [text if len(text) <= REPORT_CHARS else text[:REPORT_CHARS] + " …" for text in recent]
    lines += ["", "History you may read with the file tools:",
              "- the earlier conversation, one JSON message per line: "
              + ", ".join(str(transcripts / f"{s}.jsonl") for s in sessions[-SESSIONS_SHOWN:]),
              f"- the research notebook: {notebook / 'index.md'}, {notebook / 'timeline.md'}"]
    return "\n".join(lines)
