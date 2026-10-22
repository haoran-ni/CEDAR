"""Full automation: while the researcher is away, an approver takes their routine decisions, and the Evidence
Judge assesses new evidence before every review of the Chief Scientist.

The researcher switches it on for one investigation with ``/mode full-automation <core-hours> <dollars>`` and off
with ``/mode human-in-the-loop`` or ``/halt``. The approver decides whether proposed agents start and whether
calculations are submitted, within the core-hours allowed; inputs needing review, cancellations and closing stay
with the researcher. Each pending item is offered to the approver once per stretch of full automation; one it
leaves undecided, or one over the core-hours left, waits for the researcher. When the dollars allowed for model use
are spent, the dispatcher switches the investigation back to human-in-the-loop and pauses it.
"""

from dataclasses import dataclass

from cedar.contracts.plan import PermissionAction, Priority
from cedar.models.prices import dollars
from cedar.contracts.status import CalculationState, InvestigationStatus
from cedar.state.projection import InvestigationState, authorization_for, hypothesis_evidence

_WORKING = (InvestigationStatus.INVESTIGATING, InvestigationStatus.REVIEWING)


@dataclass(frozen=True)
class Decision:
    """Something waiting for the approver."""

    action: PermissionAction
    target_id: str
    since: int
    """Stream version from which it waits in this stretch of full automation."""


def pending_decisions(state: InvestigationState) -> list[Decision]:
    """What the approver may decide now: a proposal of agents waiting to start, and calculations waiting to be
    submitted whose requested core-hours fit what is left. Nothing it has decided in this stretch."""
    automation = state.automation
    if automation is None or state.status not in _WORKING:
        return []
    found = []
    proposal = state.pending_proposal
    if (proposal is not None and not proposal.feedback_ids and state.status is InvestigationStatus.INVESTIGATING
            and state.decided_now(proposal.proposal_id) is None):
        found.append(Decision(PermissionAction.START_AGENTS, proposal.proposal_id,
                              max(proposal.recorded_at, automation.recorded_at)))
    if not state.spec.constraints.submit_requires_approval:
        return found
    left = state.automation_core_hours_left
    for calc in state.calculations.values():
        if (calc.state is not CalculationState.VALIDATED_INPUT or state.decided_now(calc.calculation_id) is not None
                or authorization_for(state, PermissionAction.SUBMIT, calc.calculation_id) is not None
                or all(state.plan.priority(h) is Priority.ON_HOLD for h in calc.hypothesis_ids)):
            continue
        hours = calc.examination.requested_core_hours
        if hours is None or hours > left:
            continue  # waits for the researcher
        ready = next((v for v, s in reversed(calc.history) if s is CalculationState.VALIDATED_INPUT), 0)
        found.append(Decision(PermissionAction.SUBMIT, calc.calculation_id, max(ready, automation.recorded_at)))
    return found


def approver_due(state: InvestigationState) -> bool:
    """Whether to wake the approver: something waits for it that it has not been offered in this stretch, and it
    is not already at work."""
    automation = state.automation
    if automation is None:
        return False
    tasks = [t for t in state.tasks.values() if t.role == "approver" and t.queued_at > automation.recorded_at]
    if any(t.completed_by is None and t.dropped_at is None for t in tasks):
        return False
    decisions = pending_decisions(state)
    if not tasks:
        return bool(decisions)
    last = max(t.queued_at for t in tasks)
    return any(d.since > last for d in decisions)


def judgements_due(state: InvestigationState) -> list[tuple[str, list[str]]]:
    """In full automation, each hypothesis with evidence recorded since it was last assessed or sent to the
    Evidence Judge, with that evidence's ids."""
    if state.automation is None or state.status not in _WORKING:
        return []
    due = []
    for hypothesis_id, hyp in state.hypotheses.items():
        tasks = [t for t in state.tasks.values() if t.role == "evidence_judge" and t.hypothesis_id == hypothesis_id]
        if any(t.completed_by is None and t.dropped_at is None for t in tasks):
            continue
        last = max([a.recorded_at for a in hyp.assessments] + [t.queued_at for t in tasks], default=0)
        new = [r.evidence_id for r in hypothesis_evidence(state, hypothesis_id) if r.recorded_at > last]
        if new:
            due.append((hypothesis_id, new))
    return due


def mode_summary(state: InvestigationState) -> str:
    """The mode in a few words, with what full automation has left."""
    automation = state.automation
    if automation is None:
        return "human-in-the-loop"
    return (f"full automation: {max(state.automation_core_hours_left, 0):g} of {automation.max_core_hours:g} "
            f"core-hours and {dollars(max(state.automation_cost_left, 0))} of {dollars(automation.max_usd)} left")
