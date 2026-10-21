"""Plain-text views of research state, shared by the chat, the Chief Scientist and other agents.

These summarize the record; they never add to it. Job status and scientific status are
always shown separately.
"""

from cedar.automation import mode_summary, pending_decisions
from cedar.contracts.examination import ExaminationVerdict, FindingSeverity
from cedar.contracts.plan import PermissionAction
from cedar.contracts.status import CalculationState
from cedar.state.projection import (
    CalculationRecord,
    InvestigationState,
    RecommendationRecord,
    authorization_for,
    evidence_owner,
)
from cedar.swarm import agent_label, proposal_note
from cedar.web import web_summary


def calculation_summary(state: InvestigationState, calc: CalculationRecord, *, findings: bool = True,
                        viewer: str | None = None) -> str:
    """One calculation; ``viewer`` is the hypothesis whose agent reads it, which then sees only its own work."""
    head = f"{calc.calculation_id}: {calc.state.value}"
    if calc.examination is not None:
        head += f"; inputs {calc.examination.verdict.value}"
    if calc.scheduler_job_id:
        job_state = calc.scheduler_state.value if calc.scheduler_state else "submitted"
        head += f"; job {calc.scheduler_job_id} {job_state}"
    used_by = "" if viewer is not None else f"; for {', '.join(calc.hypothesis_ids)}"
    lines = [head, f"  {calc.task.observable.name} via {calc.task.protocol.name} {calc.task.protocol.version}; "
                   f"estimand: {calc.task.estimand}{used_by}"]
    if findings and calc.examination is not None:
        for f in calc.examination.findings:
            if f.severity is not FindingSeverity.INFO or f.code == "generated":
                where = f" ({f.file}{':' + str(f.line) if f.line else ''})" if f.file else ""
                lines.append(f"  [{f.severity.value}] {f.code}: {f.message}{where}")
    if calc.run_check is not None:
        failed = [c for c in calc.run_check.checks if not c.passed]
        lines.append("  run check: " + ("healthy" if not failed else
                                        "FAILED " + "; ".join(f"{c.name} ({c.detail or 'no detail'})" for c in failed)))
        lines += [f"  protocol read {r.name} = {r.value.value!r} {r.value.unit} at {r.file} line {r.line_start}: "
                  f"{r.quote}" for r in calc.run_check.readings]
    if viewer is None:  # what every hypothesis expects of this run, side by side; agents see only their own
        lines += [f"  predicted by {r.hypothesis_id} ({r.prediction_id}): {r.prediction.expected_outcome}; falsified "
                  f"if {r.prediction.falsified_if}" for r in state.predictions.values()
                  if calc.calculation_id in r.prediction.calculation_ids]
    for evidence_id in calc.evidence_ids:
        evidence = state.evidence[evidence_id].evidence
        if viewer is not None and evidence_owner(state, evidence) not in (None, viewer):
            continue  # another hypothesis's reading of a shared run
        value = f"{evidence.value.value!r} {evidence.value.unit}" if evidence.value else "no value"
        failed = [c.name for c in evidence.quality_checks if not c.passed]
        origin = f" (analysis {evidence.analysis_id})" if evidence.analysis_id else ""
        lines.append(f"  evidence {evidence_id}{origin}: {evidence.quality.value}, {evidence.observable.name} = "
                     f"{value}" + (f"; failed checks: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)


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


def task_owner(task) -> str:
    """Who a queued task is for, in words."""
    if task.role == "chief_scientist":
        return "Chief Scientist's review of agent reports"
    if task.role == "evidence_judge":
        return f"Evidence Judge on {task.hypothesis_id}"
    return f"{task.agent_id} on {task.hypothesis_id}"


def approval_requests(state: InvestigationState) -> list[tuple[str, str]]:
    """What waits for the researcher: the command that grants it, and what it concerns. In full automation, what
    the approver may still decide is not the researcher's to wait on."""
    needs = []
    for_approver = {d.target_id for d in pending_decisions(state)}
    proposal = state.pending_proposal
    if proposal is not None and proposal.proposal_id not in for_approver:
        needs.append(("/approve agents", proposal_note(state, proposal)))
    for calc in state.calculations.values():
        if (calc.state is CalculationState.SPECIFIED and calc.examination is not None
                and calc.examination.verdict is ExaminationVerdict.NEEDS_REVIEW
                and authorization_for(state, PermissionAction.APPROVE_INPUTS, calc.calculation_id) is None):
            reasons = "; ".join(f.message for f in calc.examination.findings if f.severity is FindingSeverity.REVIEW)
            needs.append((f"/approve inputs {calc.calculation_id}", f"needs review: {reasons}"))
        if (calc.state is CalculationState.VALIDATED_INPUT and state.spec.constraints.submit_requires_approval
                and calc.calculation_id not in for_approver
                and authorization_for(state, PermissionAction.SUBMIT, calc.calculation_id) is None):
            needs.append((f"/approve submit {calc.calculation_id}", f"{calc.task.system_label}: {calc.task.estimand}"))
    closing = state.pending_closing
    if closing is not None:
        needs.append(("/approve close", f"the Chief Scientist proposes closing the investigation as "
                                        f"{closing.status.value} ({closing.closing_id}): {closing.reason}"))
    return needs


def pending_approvals(state: InvestigationState) -> list[str]:
    """What waits for the researcher, with the command that grants it."""
    return [f"{command}  ({note})" for command, note in approval_requests(state)]


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
    budget = state.model_token_budget
    lines.append(f"Model tokens used: {state.model_tokens_used}" + (f" of {budget}" if budget else ""))
    return "\n".join(lines)
