"""What the live panel shows, computed from the record without any terminal code.

One row per hypothesis, with four separate statuses that must not be confused: the
hypothesis's lifecycle, its agent's role and activity (or the agent proposed for it), its
calculations' scheduler states, and the quality of their evidence, next to the Evidence
Judge's assessment. Several agents can be working at once. Ages come from event times and
the dispatcher's own reports; there are no progress percentages, because nothing measures
progress.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sim2real.contracts.events import RecordedEvent
from sim2real.contracts.examination import ExaminationVerdict
from sim2real.contracts.plan import PermissionAction
from sim2real.contracts.status import CalculationState, HypothesisStatus, InvestigationMode
from sim2real.dispatch import Activity
from sim2real.models.types import ModelSettings
from sim2real.state.projection import (
    CalculationRecord,
    HypothesisRecord,
    InvestigationState,
    authorization_for,
    hypothesis_evidence,
)
from sim2real.summaries import approval_requests
from sim2real.swarm import minutes_left

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
            text += ", run not yet checked"
        elif calc.run_check is not None and not calc.run_check.healthy:
            text += ", run failed its checks"
    return f"{short_id(calc.calculation_id)} {text}"


def _times(events: Sequence[RecordedEvent]) -> dict[int, datetime]:
    return {e.stream_version: e.occurred_at for e in events}


def _role(text: str, limit: int = 32) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def agent_status(state: InvestigationState, hyp: HypothesisRecord, times: dict[int, datetime],
                 activities: Sequence[Activity], now: datetime) -> str:
    hid = hyp.hypothesis_id
    agent = state.active_agent(hid)
    mine = [a for a in activities if (a.investigation_id, a.hypothesis_id) == (state.investigation_id, hid)]
    if not mine and agent is None:
        proposal = state.pending_proposal
        spec = next((s for s in proposal.agents if s.hypothesis_id == hid), None) if proposal else None
        if spec is not None:
            if proposal.feedback_ids:
                return f"proposed {_role(spec.role)}; held for your reply"
            if state.automation is not None:
                return f"proposed {_role(spec.role)}; for the approver"
            left = minutes_left(state, proposal, now)
            return f"proposed {_role(spec.role)}; " + (f"starts in {age(left * 60)}" if left > 0 else "starting")
        retired = [a for a in state.agents.values() if a.hypothesis_id == hid]
        return "agent retired" if retired else "no agent"
    name = _role(agent.spec.role) if agent else "agent"
    for activity in mine:
        if activity.role == "evidence_judge":
            return f"judge working for {age((now - activity.started_at).total_seconds())}"
    if mine:
        step = f"; last step {mine[0].last_step}" if mine[0].last_step else ""
        return f"{name}: working for {age((now - mine[0].started_at).total_seconds())}{step}"
    pending = [t for t in state.pending_tasks if t.hypothesis_id == hid]
    if pending:
        who = "judge" if pending[0].role == "evidence_judge" else name
        return f"{who}: queued {age((now - times[pending[0].queued_at]).total_seconds())} ago"
    if hyp.status is HypothesisStatus.WAITING_FOR_JOB:
        return f"{name}: waiting for jobs"
    runs = [r for r in state.agent_runs if r.run.task_id and state.tasks[r.run.task_id].hypothesis_id == hid]
    if runs:
        last = runs[-1]
        who = "judge" if last.role == "evidence_judge" else name
        return f"{who}: last ran {age((now - times[last.recorded_at]).total_seconds())} ago ({last.run.outcome})"
    return f"{name}: not run yet"


def swarm_rows(state: InvestigationState, events: Sequence[RecordedEvent], activities: Sequence[Activity],
               now: datetime) -> list[SwarmRow]:
    times = _times(events)
    rows = []
    for hyp in state.hypotheses.values():
        calcs = [state.calculations[c] for c in hyp.calculation_ids]
        qualities = Counter(r.evidence.quality.value for r in hypothesis_evidence(state, hyp.hypothesis_id))
        last = hyp.assessments[-1] if hyp.assessments else None
        rows.append(SwarmRow(
            hypothesis_id=hyp.hypothesis_id,
            lifecycle=hyp.status.value,
            priority=state.plan.priority(hyp.hypothesis_id).value,
            agent=agent_status(state, hyp, times, activities, now),
            jobs="; ".join(job_status(state, c) for c in calcs) or "none",
            evidence=", ".join(f"{n} {quality}" for quality, n in qualities.items()) or "none",
            assessment=f"{last.assessment.value} ({last.completeness.value.lower()})" if last else "NOT_TESTED",
        ))
    return rows


def dispatcher_status(heartbeat_age: float | None, activities: Sequence[Activity], now: datetime,
                      automatic: bool = False) -> tuple[str, bool]:
    """A short description of the dispatcher, and whether it needs the researcher's attention; with
    ``automatic`` the chat starts one when there is work, so a missing one needs no attention."""
    if automatic and (heartbeat_age is None or heartbeat_age > STALE_DISPATCHER_SECONDS):
        return "no dispatcher running; one starts by itself when there is work", False
    if heartbeat_age is None:
        return "no dispatcher has run: start `sim2real dispatch`", True
    if heartbeat_age > STALE_DISPATCHER_SECONDS:
        return f"dispatcher silent for {age(heartbeat_age)}: is `sim2real dispatch` running?", True
    if activities:
        roles = Counter(a.role for a in activities)
        parts = []
        if roles["hypothesis_agent"]:
            n = roles["hypothesis_agent"]
            parts.append(f"{n} hypothesis agent{'s' if n > 1 else ''}")
        if roles["evidence_judge"]:
            parts.append("the Evidence Judge")
        if roles["chief_scientist"]:
            parts.append("the Chief Scientist's review")
        if roles["approver"]:
            parts.append("the approver")
        return f"dispatcher running {' and '.join(parts)}", False
    return f"dispatcher seen {age(heartbeat_age)} ago", False


def subagent_status(state: InvestigationState | None, activities: Sequence[Activity], now: datetime,
                    dispatcher_running: bool = True, automatic: bool = False) -> str:
    """Which sub-agents (hypothesis agents and the Evidence Judge) are working on the investigation right now,
    and how much of their work is queued; queued work waits for a dispatcher if none is running."""
    mine = [a for a in activities if state is not None and a.investigation_id == state.investigation_id]
    working = []
    for activity in mine:
        took = age((now - activity.started_at).total_seconds())
        if activity.role == "hypothesis_agent":
            agent = state.agents.get(activity.agent_id)
            working.append(f"{agent.spec.role if agent else activity.agent_id} on {activity.hypothesis_id} ({took})")
        elif activity.role == "evidence_judge":
            working.append(f"Evidence Judge on {activity.hypothesis_id} ({took})")
        elif activity.role == "approver":
            working.append(f"approver deciding ({took})")
    text = f"Working sub-agents: {'; '.join(working)}" if working else "No working sub-agents"
    running = {a.task_id for a in mine}
    queued = [t for t in state.pending_tasks if t.task_id not in running] if state is not None else []
    if queued:
        text += f" · {len(queued)} task{'s' if len(queued) > 1 else ''} queued"
        if not dispatcher_running:
            text += (", waiting for the dispatcher to start" if automatic else
                     ", waiting for a dispatcher: start `sim2real dispatch`")
    if any(a.role == "chief_scientist" for a in mine):
        text += " · the Chief Scientist is reviewing their reports"
    return text


MODEL_ROLES = (("chief_scientist", "Chief Scientist"), ("hypothesis_agent", "hypothesis agents"),
               ("evidence_judge", "Evidence Judge"), ("approver", "approver"))


def model_line(settings: Callable[[str], ModelSettings], state: InvestigationState | None) -> str:
    """The model each agent role asks the provider for, and any other model that answered a role's latest run."""
    configured = {}
    for role, label in MODEL_ROLES:
        try:
            configured[label] = settings(role)
        except KeyError:
            continue
    if not configured:
        return "Model: none configured (sim2real models init)"

    def name(s: ModelSettings) -> str:
        return s.model + (f", effort {s.effort}" if s.effort else "")

    if len({(s.provider, name(s)) for s in configured.values()}) == 1:
        only = next(iter(configured.values()))
        text = f"Model: {name(only)} ({only.provider})"
    else:
        text = "Models: " + " · ".join(f"{label} {name(s)} ({s.provider})" for label, s in configured.items())
    for role, label in MODEL_ROLES:
        runs = [r.run for r in state.agent_runs if r.role == role] if state is not None else []
        if runs and any(m != runs[-1].requested_model for m in runs[-1].served_models):
            text += f" · {label}: latest run answered by {', '.join(runs[-1].served_models)}"
    return text


def work_summary(state: InvestigationState) -> str:
    """Counts for the compact view: hypotheses, jobs at the scheduler, evidence by quality."""
    calcs = list(state.calculations.values())
    counts = [(sum(c.state is _C.RUNNING for c in calcs), "running"),
              (sum(c.state in (_C.SUBMISSION_PENDING, _C.SUBMITTED, _C.QUEUED) for c in calcs), "queued"),
              (sum(c.state in (_C.SCHEDULER_TERMINAL, _C.PARSING) for c in calcs), "not yet checked")]
    jobs = ", ".join(f"{n} {label}" for n, label in counts if n) or "none active"
    qualities = Counter(record.evidence.quality.value for record in state.evidence.values())
    evidence = ", ".join(f"{n} {quality}" for quality, n in qualities.items()) or "none"
    n = len(state.hypotheses)
    return f"{n} hypothes{'is' if n == 1 else 'es'} · jobs: {jobs} · evidence: {evidence}"


def mode_line(state: InvestigationState | None) -> tuple[str, bool]:
    """The panel's own line for the mode, and whether it is full automation, which the panel highlights; a new
    investigation starts in human-in-the-loop. ``/mode`` gives the details."""
    mode = InvestigationMode.HUMAN_IN_THE_LOOP if state is None else state.mode
    return f"Mode: {mode.value}", mode is InvestigationMode.FULL_AUTOMATION


def status_line(state: InvestigationState, now: datetime) -> str:
    budget = state.spec.constraints.model_token_budget
    tokens = f"{state.model_tokens_used:,}" + (f" of {budget:,}" if budget else "")
    waiting = len(approval_requests(state))
    agents = sum(a.active for a in state.agents.values())
    automation = state.automation
    parts = [state.investigation_id, state.status.value, "setup" if state.round == 0 else f"round {state.round}",
             f"plan v{state.plan_version}", f"{agents} agent{'' if agents == 1 else 's'}", f"{tokens} tokens"]
    proposal = state.pending_proposal
    if proposal is not None:
        n = len(proposal.agents)
        left = minutes_left(state, proposal, now)
        when = ("held for your reply" if proposal.feedback_ids else "for the approver" if automation is not None else
                f"start in {age(left * 60)} unless you reply" if left > 0 else "starting")
        parts.append(f"{n} agent{'s' if n > 1 else ''} proposed: {when} (/agents)")
    if waiting:
        parts.append(f"{waiting} waiting for you (/pending)")
    return " · ".join(parts)
