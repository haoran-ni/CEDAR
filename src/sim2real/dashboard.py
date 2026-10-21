"""What the live panel shows, computed from the record without any terminal code.

One row per hypothesis, with four separate statuses that must not be confused: the
hypothesis's lifecycle, its agent's activity, its calculations' scheduler states, and the
quality of their evidence, next to the Evidence Judge's assessment. Ages come from event
times and the dispatcher's own reports; there are no progress percentages, because nothing
measures progress.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sim2real.contracts.events import RecordedEvent
from sim2real.contracts.examination import ExaminationVerdict
from sim2real.contracts.plan import PermissionAction
from sim2real.contracts.status import CalculationState, HypothesisStatus
from sim2real.dispatch import Activity
from sim2real.state.projection import CalculationRecord, HypothesisRecord, InvestigationState, authorization_for
from sim2real.summaries import approval_requests

STALE_DISPATCHER_SECONDS = 600
"""A dispatcher silent for longer than this is reported as not running."""

_C = CalculationState


@dataclass(frozen=True)
class SwarmRow:
    hypothesis_id: str
    lifecycle: str
    priority: str
    agent: str
    jobs: str
    evidence: str
    assessment: str


def age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    return f"{hours // 24} d"


def short_id(calculation_id: str) -> str:
    """The first characters of a calculation id; the chat accepts them in place of the full id."""
    return calculation_id.removeprefix("calc_")[:8]


def job_status(state: InvestigationState, calc: CalculationRecord) -> str:
    """Where a calculation stands, in the scheduler's terms once it has a job."""
    exam = calc.examination
    if calc.state is _C.SPECIFIED:
        if exam is None:
            text = "being examined"
        elif exam.verdict is ExaminationVerdict.NEEDS_REVIEW and authorization_for(
                state, PermissionAction.APPROVE_INPUTS, calc.calculation_id) is None:
            text = "needs /approve inputs"
        else:
            text = "examined"
    elif calc.state is _C.REJECTED:
        text = "inputs rejected"
    elif calc.state is _C.VALIDATED_INPUT:
        needs = state.spec.constraints.submit_requires_approval and authorization_for(
            state, PermissionAction.SUBMIT, calc.calculation_id) is None
        text = "needs /approve submit" if needs else "authorized, not yet submitted"
    elif calc.state is _C.SUBMISSION_PENDING:
        text = "submitting"
    elif calc.state is _C.CANCELLED:
        text = "cancelled" + (f" (job {calc.scheduler_job_id})" if calc.scheduler_job_id else "")
    else:
        scheduler = calc.scheduler_state.value if calc.scheduler_state else "SUBMITTED"
        text = f"job {calc.scheduler_job_id} {scheduler}"
        if calc.state in (_C.SCHEDULER_TERMINAL, _C.PARSING):
            text += ", output not yet read"
    return f"{short_id(calc.calculation_id)} {text}"


def _times(events: Sequence[RecordedEvent]) -> dict[int, datetime]:
    return {e.stream_version: e.occurred_at for e in events}


def agent_status(state: InvestigationState, hyp: HypothesisRecord, times: dict[int, datetime],
                 activity: Activity | None, now: datetime) -> str:
    hid = hyp.hypothesis_id
    if activity is not None and (activity.investigation_id, activity.hypothesis_id) == (state.investigation_id, hid):
        who = "judge" if activity.role == "evidence_judge" else "agent"
        step = f"; last step {activity.last_step}" if activity.last_step else ""
        return f"{who} working for {age((now - activity.started_at).total_seconds())}{step}"
    pending = [t for t in state.pending_tasks if t.hypothesis_id == hid]
    if pending:
        who = "judge" if pending[0].role == "evidence_judge" else "agent"
        return f"{who} queued {age((now - times[pending[0].queued_at]).total_seconds())} ago"
    if hyp.status is HypothesisStatus.WAITING_FOR_JOB:
        return "agent waiting for jobs"
    runs = [r for r in state.agent_runs if r.run.task_id and state.tasks[r.run.task_id].hypothesis_id == hid]
    if runs:
        last = runs[-1]
        who = "judge" if last.role == "evidence_judge" else "agent"
        return f"{who} last ran {age((now - times[last.recorded_at]).total_seconds())} ago ({last.run.outcome})"
    return "no agent yet"


def swarm_rows(state: InvestigationState, events: Sequence[RecordedEvent], activity: Activity | None,
               now: datetime) -> list[SwarmRow]:
    times = _times(events)
    rows = []
    for hyp in state.hypotheses.values():
        calcs = [state.calculations[c] for c in hyp.calculation_ids]
        qualities = Counter(state.evidence[c.evidence_id].evidence.quality.value for c in calcs if c.evidence_id)
        last = hyp.assessments[-1] if hyp.assessments else None
        rows.append(SwarmRow(
            hypothesis_id=hyp.hypothesis_id,
            lifecycle=hyp.status.value,
            priority=state.plan.priority(hyp.hypothesis_id).value,
            agent=agent_status(state, hyp, times, activity, now),
            jobs="; ".join(job_status(state, c) for c in calcs) or "none",
            evidence=", ".join(f"{n} {quality}" for quality, n in qualities.items()) or "none",
            assessment=f"{last.assessment.value} ({last.completeness.value.lower()})" if last else "NOT_TESTED",
        ))
    return rows


def dispatcher_status(heartbeat_age: float | None, activity: Activity | None, now: datetime) -> tuple[str, bool]:
    """A short description of the dispatcher, and whether it needs the researcher's attention."""
    if heartbeat_age is None:
        return "no dispatcher has run: start `sim2real dispatch`", True
    if heartbeat_age > STALE_DISPATCHER_SECONDS:
        return f"dispatcher silent for {age(heartbeat_age)}: is `sim2real dispatch` running?", True
    if activity is not None:
        who = "Evidence Judge" if activity.role == "evidence_judge" else f"agent on {activity.hypothesis_id}"
        return f"dispatcher running the {who}", False
    return f"dispatcher seen {age(heartbeat_age)} ago", False


def work_summary(state: InvestigationState) -> str:
    """Counts for the compact view: hypotheses, jobs at the scheduler, evidence by quality."""
    calcs = list(state.calculations.values())
    counts = [(sum(c.state is _C.RUNNING for c in calcs), "running"),
              (sum(c.state in (_C.SUBMISSION_PENDING, _C.SUBMITTED, _C.QUEUED) for c in calcs), "queued"),
              (sum(c.state in (_C.SCHEDULER_TERMINAL, _C.PARSING) for c in calcs), "output not yet read")]
    jobs = ", ".join(f"{n} {label}" for n, label in counts if n) or "none active"
    qualities = Counter(record.evidence.quality.value for record in state.evidence.values())
    evidence = ", ".join(f"{n} {quality}" for quality, n in qualities.items()) or "none"
    n = len(state.hypotheses)
    return f"{n} hypothes{'is' if n == 1 else 'es'} · jobs: {jobs} · evidence: {evidence}"


def status_line(state: InvestigationState) -> str:
    budget = state.spec.constraints.model_token_budget
    tokens = f"{state.model_tokens_used:,}" + (f" of {budget:,}" if budget else "")
    waiting = len(approval_requests(state))
    parts = [state.investigation_id, state.status.value, "setup" if state.round == 0 else f"round {state.round}",
             f"plan v{state.plan_version}", f"{tokens} tokens"]
    if waiting:
        parts.append(f"{waiting} waiting for you (/pending)")
    return " · ".join(parts)
