"""Plain-text views of research state, shared by the chat, the Chief Scientist and other agents.

These summarize the record; they never add to it. Job status and scientific status are
always shown separately.
"""

from collections.abc import Iterable
from pathlib import Path

from cedar.automation import mode_summary, pending_decisions
from cedar.contracts.base import canonical_json
from cedar.contracts.examination import ExaminationVerdict, FindingSeverity
from cedar.contracts.plan import PermissionAction
from cedar.contracts.status import CalculationState
from cedar.models.prices import dollars
from cedar.state.projection import (
    CalculationRecord,
    InvestigationState,
    RecommendationRecord,
    SynthesisRecord,
    authorization_for,
    evidence_owner,
)
from cedar.swarm import agent_label, proposal_note
from cedar.web import web_summary


def calculation_summary(state: InvestigationState, calc: CalculationRecord, *, findings: bool = True) -> str:
    """One calculation: its state, what it is for, what every hypothesis predicted for it, and its evidence with
    whose it is. Every agent sees all of it."""
    if calc.registered:
        registration = calc.registration
        owner = "shared with every agent" if calc.shared else f"for {registration.hypothesis_id}"
        settings = ", ".join(f"{name} {canonical_json(value)}" for name, value in
                             {**calc.task.controls, **calc.task.numerical_settings}.items())
        lines = [f"{calc.calculation_id}: registered run from {registration.source} ({owner}; computed outside "
                 f"CEDAR, FHI-aims {registration.code_version})",
                 f"  {registration.description}", f"  settings its inputs used: {settings or 'none recorded'}"]
    else:
        head = f"{calc.calculation_id}: {calc.state.value}"
        if calc.examination is not None:
            head += f"; inputs {calc.examination.verdict.value}"
        if calc.scheduler_job_id:
            job_state = calc.scheduler_state.value if calc.scheduler_state else "submitted"
            head += f"; job {calc.scheduler_job_id} {job_state}"
        lines = [head, f"  {calc.task.observable.name} via {calc.task.protocol.name} {calc.task.protocol.version}; "
                       f"estimand: {calc.task.estimand}; for {', '.join(calc.hypothesis_ids)}"]
        if calc.rejected_by_researcher:
            lines.append(f"  rejected by the researcher ({calc.ended_by}): {calc.end_reason}")
        elif calc.state is CalculationState.CANCELLED and calc.end_reason:
            lines.append("  cancelled" + (f" ({calc.ended_by})" if calc.ended_by else "") + f": {calc.end_reason}")
    if findings and calc.examination is not None:
        for f in calc.examination.findings:
            if f.severity is not FindingSeverity.INFO or f.code == "generated":
                lines.append(f"  [{f.severity.value}] {f.code}: {f.message}{_place(f)}")
    if calc.run_check is not None:
        failed = [c for c in calc.run_check.checks if not c.passed]
        lines.append("  run check: " + ("healthy" if not failed else
                                        "FAILED " + "; ".join(f"{c.name} ({c.detail or 'no detail'})" for c in failed)))
        lines += [f"  protocol read {r.name} = {r.value.value!r} {r.value.unit} at {r.file} line {r.line_start}: "
                  f"{r.quote}" for r in calc.run_check.readings]
    # What every hypothesis expects of this run, side by side.
    lines += [f"  predicted by {r.hypothesis_id} ({r.prediction_id}): {r.prediction.expected_outcome}; falsified "
              f"if {r.prediction.falsified_if}" for r in state.predictions.values()
              if calc.calculation_id in r.prediction.calculation_ids]
    for evidence_id in calc.evidence_ids:
        evidence = state.evidence[evidence_id].evidence
        value = f"{evidence.value.value!r} {evidence.value.unit}" if evidence.value else "no value"
        failed = [c.name for c in evidence.quality_checks if not c.passed]
        origin = f" (analysis {evidence.analysis_id})" if evidence.analysis_id else ""
        owner = evidence_owner(state, evidence)
        lines.append(f"  evidence {evidence_id}{origin}" + (f" for {owner}" if owner else "")
                     + f": {evidence.quality.value}, {evidence.observable.name} = {value}"
                     + (f"; failed checks: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)


def _place(finding) -> str:
    """Where in the inputs a finding is, e.g. " (control.in:42)"; nothing if it names no file."""
    return f" ({finding.file}{':' + str(finding.line) if finding.line else ''})" if finding.file else ""


def short_id(calculation_id: str, among: Iterable[str] = ()) -> str:
    """The first characters of a calculation id, eight unless more are needed to tell it from those ``among``; the
    chat accepts them in place of the full id."""
    stem = calculation_id.removeprefix("calc_")
    others = [other.removeprefix("calc_") for other in among if other != calculation_id]
    length = 8
    while length < len(stem) and any(other.startswith(stem[:length]) for other in others):
        length += 1
    return stem[:length]


def recommendation_text(record: RecommendationRecord, *, brief: bool = False) -> str:
    """The Chief Scientist's candidate next steps and its choice; ``brief`` gives one line per candidate."""
    rec = record.recommendation
    taken = ", ".join(rec.chosen) if rec.chosen else "none for now"
    lines = [f"Next steps recommended by the Chief Scientist ({rec.recommendation_id}, round {record.round}). Taking: "
             f"{taken}. Why: {rec.reason}"]
    for c in rec.candidates:
        mark = "chosen; " if c.candidate_id in rec.chosen else ""
        bears = f" on {', '.join(c.hypothesis_ids)}" if c.hypothesis_ids else ""
        lines.append(f"  {c.candidate_id} ({mark}{c.kind.value}, {c.priority.value} priority){bears}: {c.action}")
        if brief:
            continue
        lines += [f"    if {o.hypothesis_id} holds: {o.outcome}" for o in c.expected_outcomes]
        lines += [f"    informative because: {c.rationale}",
                  f"    cost: {c.cost}; feasibility: {c.feasibility}",
                  f"    priority because: {c.priority_reason}"]
    return "\n".join(lines)


def synthesis_text(state: InvestigationState, record: SynthesisRecord) -> str:
    """The Chief Scientist's comparison across hypotheses, with the values of the evidence it cites."""
    p = record.synthesis
    lines = [f"Comparison across hypotheses ({p.synthesis_id}, round {record.round}): {p.summary}"]
    for h in p.hypotheses:
        lines.append(f"- {h.hypothesis_id}: {h.standing}" + (f" Accounts for: {h.accounts_for}" if h.accounts_for
                                                              else ""))
        for evidence_id in h.evidence_ids:
            ev = state.evidence[evidence_id].evidence
            value = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
            lines.append(f"  {evidence_id}: {ev.observable.name} = {value} ({ev.quality.value})")
    lines += [f"- Interaction: {text}" for text in p.interactions]
    lines.append(f"- Unexplained: {p.residual}")
    lines += [f"- Cannot be separated: {text}" for text in p.not_separable]
    if p.new_hypotheses:
        lines.append(f"- New hypotheses from this comparison: {', '.join(p.new_hypotheses)}")
    return "\n".join(lines)


def task_owner(task) -> str:
    """Who a queued task is for, in words."""
    if task.role == "chief_scientist":
        return "Chief Scientist's review of agent reports"
    if task.role == "evidence_judge":
        return f"Evidence Judge on {task.hypothesis_id}"
    return f"{task.agent_id} on {task.hypothesis_id}"


def awaiting_submission(state: InvestigationState) -> list[str]:
    """Calculations whose examined inputs wait for the researcher's authorization to be submitted."""
    if not state.spec.constraints.submit_requires_approval:
        return []
    return [c.calculation_id for c in state.calculations.values() if c.state is CalculationState.VALIDATED_INPUT
            and authorization_for(state, PermissionAction.SUBMIT, c.calculation_id) is None]


def requested_by(state: InvestigationState, calc: CalculationRecord) -> str:
    """The agent that asked for a calculation, by its short name and id, with its hypothesis."""
    hypothesis_id = calc.hypothesis_ids[0] if calc.hypothesis_ids else None
    agents = [a for a in state.agents.values() if a.hypothesis_id == hypothesis_id]
    agent = next((a for a in agents if a.active), agents[-1] if agents else None)
    return f"{agent.tag} {agent.agent_id} on {hypothesis_id}" if agent else (hypothesis_id or "no hypothesis")


def awaiting_review(state: InvestigationState) -> list[str]:
    """Calculations whose inputs wait for the researcher's review: the examiner could not verify part of them."""
    return [c.calculation_id for c in state.calculations.values()
            if c.state is CalculationState.SPECIFIED and c.examination is not None
            and c.examination.verdict is ExaminationVerdict.NEEDS_REVIEW
            and authorization_for(state, PermissionAction.APPROVE_INPUTS, c.calculation_id) is None]


def run_directory(calc: CalculationRecord, runs_root: str | Path | None = None) -> str | None:
    """Where a calculation's examined inputs are: under ``runs_root`` if given, else relative to it."""
    if calc.workdir is None:
        return None
    return str(Path(runs_root) / calc.workdir) if runs_root is not None else calc.workdir


def review_note(state: InvestigationState, calc: CalculationRecord, runs_root: str | Path | None = None) -> str:
    """What the researcher needs to review inputs: who asks, what the examiner could not verify and where, and
    where the files are."""
    reasons = "; ".join(f"{f.message}{_place(f)}" for f in calc.examination.findings
                        if f.severity is FindingSeverity.REVIEW)
    directory = run_directory(calc, runs_root)
    return (f"{requested_by(state, calc)}; needs review: {reasons}"
            + (f"; files in {directory}" if directory else ""))


def submission_note(state: InvestigationState, calc: CalculationRecord, runs_root: str | Path | None = None) -> str:
    """What the researcher needs to decide on a submission: who asks, for what, where it runs and what it costs."""
    exam = calc.examination
    directory = run_directory(calc, runs_root)
    where = ("locally, on the interactive node" if calc.task.execution == "local"
             else f"through Slurm, partition {exam.partition}" if exam.partition else "through Slurm")
    size = (f"{exam.requested_tasks} tasks × {exam.requested_minutes} min = {exam.requested_core_hours:g} core-hours"
            if exam.requested_tasks and exam.requested_minutes and exam.requested_core_hours is not None
            else "size not known")
    warnings = [f.code for f in exam.findings if f.severity is FindingSeverity.WARNING]
    parts = [requested_by(state, calc), f"{calc.task.system_label}: {calc.task.estimand}", where, size,
             f"warnings: {', '.join(warnings)}" if warnings else "no warnings",
             f"examined inputs ({', '.join(sorted(exam.files))}) in the run directory {directory}"
             if directory else f"examined inputs: {', '.join(sorted(exam.files))}"]
    return "; ".join(parts)


def approval_requests(state: InvestigationState, runs_root: str | Path | None = None) -> list[tuple[str, str]]:
    """What waits for the researcher: the command that grants it, naming a calculation by the short form of its
    id, and what it concerns, with run directories under ``runs_root`` if given. In full automation, what the
    approver may still decide is not the researcher's to wait on."""
    needs = []
    for_approver = {d.target_id for d in pending_decisions(state)}
    proposal = state.pending_proposal
    if proposal is not None and proposal.proposal_id not in for_approver:
        needs.append(("/approve agents", proposal_note(state, proposal)))
    review, submission = set(awaiting_review(state)), set(awaiting_submission(state))
    for calc in state.calculations.values():
        short = short_id(calc.calculation_id, state.calculations)
        if calc.calculation_id in review:
            needs.append((f"/approve inputs {short}", review_note(state, calc, runs_root)))
        if calc.calculation_id in submission and calc.calculation_id not in for_approver:
            needs.append((f"/approve submit {short}", submission_note(state, calc, runs_root)))
    closing = state.pending_closing
    if closing is not None:
        needs.append(("/approve close", f"the Chief Scientist proposes closing the investigation as "
                                        f"{closing.status.value} ({closing.closing_id}): {closing.reason}"))
    return needs


def agent_requests(state: InvestigationState) -> list[str]:
    """What the active agents asked of the researcher in their latest reports."""
    found = []
    for agent in state.agents.values():
        runs = [r for r in state.agent_runs if r.actor_id == agent.agent_id] if agent.active else []
        report = runs[-1].run.structured if runs else None
        if report is not None:
            found += [f"{agent.tag} ({agent.agent_id}) on {agent.hypothesis_id} asks: {r.text}" for r in report.requests
                      if r.to == "researcher"]
    return found


def pending_approvals(state: InvestigationState, runs_root: str | Path | None = None) -> list[str]:
    """What waits for the researcher, with the command that grants it."""
    return [f"{command}  ({note})" for command, note in approval_requests(state, runs_root)]


def investigation_summary(state: InvestigationState) -> str:
    spec = state.spec
    lines = [
        f"Investigation {state.investigation_id}: {spec.title}",
        f"  status {state.status.value}; round {state.round}; plan v{state.plan_version}; "
        f"compute profile {spec.constraints.compute_profile or 'none'}",
        f"  mode: {mode_summary(state)}",
        f"  web access for the Chief Scientist and the hypothesis agents: {web_summary(state.retrieval)}",
        f"  question: {spec.question}",
        f"  observables: simulated '{state.alignment.simulated.name}' vs measured '{state.alignment.measured.name}' "
        f"({state.alignment.status.value})",
        f"  plan objective: {state.plan.objective}",
    ]
    if state.materials:
        lines.append("Materials the researcher attached (view_material shows them):")
        lines += [f"  {m.material_id}: {m.name}" + (f": {m.note}" if m.note else "") for m in state.materials.values()]
    if not state.hypotheses:
        lines.append("Hypotheses: none yet")
    for hyp in state.hypotheses.values():
        agent = state.active_agent(hyp.hypothesis_id)
        lines.append(f"Hypothesis {hyp.hypothesis_id} [{hyp.status.value}; assessment {hyp.assessment.value}; "
                     f"priority {state.plan.priority(hyp.hypothesis_id).value}; "
                     f"agent {agent_label(agent) if agent else 'none'}]: {hyp.spec.title}")
        if hyp.assessments:
            last = hyp.assessments[-1]
            lines.append(f"  latest assessment ({last.completeness.value}): {last.rationale}")
        lines += [f"  prediction {pid} (about no particular calculation): {state.predictions[pid].prediction.statement} "
                  f"Expected: {state.predictions[pid].prediction.expected_outcome}" for pid in hyp.prediction_ids
                  if not state.predictions[pid].prediction.calculation_ids]
    if state.calculations:
        lines.append("Calculations:")
        lines += ["  " + line for c in state.calculations.values()
                  for line in calculation_summary(state, c, findings=False).splitlines()]
    if state.syntheses:
        latest = list(state.syntheses.values())[-1]
        lines += ["Latest comparison across hypotheses:", "  " + synthesis_text(state, latest).replace("\n", "\n  ")]
    if state.recommendations:
        latest = list(state.recommendations.values())[-1]
        lines += ["Latest recommendation:", "  " + recommendation_text(latest, brief=True).replace("\n", "\n  ")]
    proposal = state.pending_proposal
    if proposal is not None:
        lines.append("Agents proposed, not started: " + "; ".join(f"{s.hypothesis_id}: {s.role}" for s in proposal.agents)
                     + f" ({proposal_note(state, proposal)})")
    if state.pending_tasks:
        lines.append("Queued agent work: " + "; ".join(f"{t.task_id} ({task_owner(t)})" for t in state.pending_tasks))
    approvals = pending_approvals(state)
    if approvals:
        lines.append("Waiting for the researcher:")
        lines += [f"  {a}" for a in approvals]
    requests = agent_requests(state)
    if requests:
        lines.append("Requests from agents to the researcher (pass them on):")
        lines += [f"  {r}" for r in requests]
    budget = state.budget_usd
    lines.append(f"Spent on models: {dollars(state.model_cost)}" + (f" of the {dollars(budget)} budget, "
                                                                   f"{dollars(state.budget_left)} left"
                                                                   if budget is not None else " (no budget)")
                 + f"; {state.model_tokens_used:,} tokens")
    return "\n".join(lines)
