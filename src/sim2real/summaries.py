"""Plain-text views of research state, shared by the chat, the Chief Scientist and other agents.

These summarize the record; they never add to it. Job status and scientific status are
always shown separately.
"""

from sim2real.contracts.examination import ExaminationVerdict, FindingSeverity
from sim2real.contracts.plan import PermissionAction
from sim2real.contracts.status import CalculationState
from sim2real.state.projection import CalculationRecord, InvestigationState, authorization_for


def calculation_summary(state: InvestigationState, calc: CalculationRecord, *, findings: bool = True) -> str:
    head = f"{calc.calculation_id}: {calc.state.value}"
    if calc.examination is not None:
        head += f"; inputs {calc.examination.verdict.value}"
    if calc.scheduler_job_id:
        job_state = calc.scheduler_state.value if calc.scheduler_state else "submitted"
        head += f"; job {calc.scheduler_job_id} {job_state}"
    lines = [head, f"  {calc.task.observable.name} via {calc.task.protocol.name} {calc.task.protocol.version}; "
                   f"estimand: {calc.task.estimand}; for {', '.join(calc.hypothesis_ids)}"]
    if findings and calc.examination is not None:
        for f in calc.examination.findings:
            if f.severity is not FindingSeverity.INFO or f.code == "generated":
                where = f" ({f.file}{':' + str(f.line) if f.line else ''})" if f.file else ""
                lines.append(f"  [{f.severity.value}] {f.code}: {f.message}{where}")
    if calc.evidence_id:
        evidence = state.evidence[calc.evidence_id].evidence
        value = f"{evidence.value.value} {evidence.value.unit}" if evidence.value else "no value"
        failed = [c.name for c in evidence.quality_checks if not c.passed]
        lines.append(f"  evidence {calc.evidence_id}: {evidence.quality.value}, {value}"
                     + (f"; failed checks: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)


def approval_requests(state: InvestigationState) -> list[tuple[str, str]]:
    """What waits for the researcher: the command that grants it, and what it concerns."""
    needs = []
    for calc in state.calculations.values():
        if (calc.state is CalculationState.SPECIFIED and calc.examination is not None
                and calc.examination.verdict is ExaminationVerdict.NEEDS_REVIEW
                and authorization_for(state, PermissionAction.APPROVE_INPUTS, calc.calculation_id) is None):
            reasons = "; ".join(f.message for f in calc.examination.findings if f.severity is FindingSeverity.REVIEW)
            needs.append((f"/approve inputs {calc.calculation_id}", f"needs review: {reasons}"))
        if (calc.state is CalculationState.VALIDATED_INPUT and state.spec.constraints.submit_requires_approval
                and authorization_for(state, PermissionAction.SUBMIT, calc.calculation_id) is None):
            needs.append((f"/approve submit {calc.calculation_id}", f"{calc.task.system_label}: {calc.task.estimand}"))
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
        f"  question: {spec.question}",
        f"  observables: simulated '{state.alignment.simulated.name}' vs measured '{state.alignment.measured.name}' "
        f"({state.alignment.status.value})",
        f"  plan objective: {state.plan.objective}",
    ]
    if not state.hypotheses:
        lines.append("Hypotheses: none yet")
    for hyp in state.hypotheses.values():
        lines.append(f"Hypothesis {hyp.hypothesis_id} [{hyp.status.value}; assessment {hyp.assessment.value}; "
                     f"priority {state.plan.priority(hyp.hypothesis_id).value}]: {hyp.spec.title}")
        if hyp.assessments:
            last = hyp.assessments[-1]
            lines.append(f"  latest assessment ({last.completeness.value}): {last.rationale}")
    if state.calculations:
        lines.append("Calculations:")
        lines += ["  " + line for c in state.calculations.values()
                  for line in calculation_summary(state, c, findings=False).splitlines()]
    if state.pending_tasks:
        lines.append("Queued agent work: " + "; ".join(f"{t.task_id} ({t.role} for {t.hypothesis_id})"
                                                      for t in state.pending_tasks))
    approvals = pending_approvals(state)
    if approvals:
        lines.append("Waiting for the researcher:")
        lines += [f"  {a}" for a in approvals]
    budget = spec.constraints.model_token_budget
    lines.append(f"Model tokens used: {state.model_tokens_used}" + (f" of {budget}" if budget else ""))
    return "\n".join(lines)
