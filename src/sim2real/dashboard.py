"""What the live panel shows, computed from the record without any terminal code.

Two tables. One row per agent (the Chief Scientist, each hypothesis agent, the Evidence
Judge and, in full automation, the approver), with what it is doing: working, queued,
waiting for jobs, proposed, idle, retired or off, for how long, and on which task. And one
row per hypothesis, with separate statuses that must not be confused: its lifecycle, its
calculations' scheduler states and the quality of their evidence, next to the Evidence
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
from sim2real.automation import pending_decisions
from sim2real.contracts.events import HypothesisStatusChanged
from sim2real.state.projection import (
    AgentRecord,
    AgentTaskRecord,
    CalculationRecord,
    InvestigationState,
    agent_id_for,
    authorization_for,
    hypothesis_evidence,
)
from sim2real.summaries import approval_requests
from sim2real.swarm import FIRST_TASK, WAKE_TASK, minutes_left

STALE_DISPATCHER_SECONDS = 600
"""A dispatcher silent for longer than this is reported as not running."""

_C = CalculationState


@dataclass(frozen=True)
class SwarmRow:
    hypothesis_id: str
    lifecycle: str
    priority: str
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
        if needs and any(d.target_id == calc.calculation_id for d in pending_decisions(state)):
            text = "for the approver"
        else:
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


@dataclass(frozen=True)
class AgentRow:
    """One agent in the panel's table of agents: what it is, and what it is doing now."""

    key: str
    name: str
    role: str
    status: str
    """working, queued, waiting for jobs, proposed (with when it starts), idle, retired or off."""
    since: str
    """How long it has been so, where that is known."""
    task: str
    hypothesis_id: str | None = None
    """The hypothesis it works on or assesses, if any."""


def task_label(task: AgentTaskRecord | None) -> str:
    """A task in a few words."""
    if task is None:
        return "—"
    if task.role == "chief_scientist":
        return "review the agents' reports"
    if task.role == "evidence_judge":
        return f"assess {task.hypothesis_id}"
    if task.role == "approver":
        return "decide on proposals and submissions"
    if task.instruction.startswith(WAKE_TASK):
        return "analyze its finished calculations"
    return _short(task.instruction.removeprefix(FIRST_TASK).splitlines()[0], 70)


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:max(limit - 1, 0)] + "…"


def fit_tasks(rows: Sequence[AgentRow], width: int, headers: Sequence[str]) -> list[str]:
    """Each row's task, shortened so that the table fits ``width`` columns: the other columns keep their widths and
    every column takes two more for padding."""
    others = [max([len(h)] + [len(text) for text in column]) for h, column in zip(
        headers, zip(*[(r.name, r.role, r.status, r.since or "—") for r in rows]) if rows else [() for _ in headers])]
    room = max(width - sum(others) - 2 * (len(others) + 1), 16)
    return [_short(r.task, room) for r in rows]


def agent_rows(state: InvestigationState | None, events: Sequence[RecordedEvent], activities: Sequence[Activity],
               now: datetime, chief_since: datetime | None = None) -> list[AgentRow]:
    """Every agent of the investigation and what it is doing: the Chief Scientist, the hypothesis agents (working
    ones first, then those proposed, then those retired), the Evidence Judge and, in full automation, the approver.
    ``chief_since`` is when the Chief Scientist began answering the researcher, if it is answering now."""
    times = _times(events)
    mine = [a for a in activities if state is not None and a.investigation_id == state.investigation_id]
    pending = state.pending_tasks if state is not None else []

    def ago(moment: datetime | None) -> str:
        return age((now - moment).total_seconds()) if moment is not None else ""

    def at(version: int | None) -> datetime | None:
        return times.get(version) if version is not None else None

    def runs_of(actor_id: str | None = None, role: str | None = None) -> list:
        if state is None:
            return []
        return [r for r in state.agent_runs if (actor_id is None or r.actor_id == actor_id)
                and (role is None or r.role == role)]

    # The Chief Scientist: answering the researcher, reviewing, or idle.
    review = next((a for a in mine if a.role == "chief_scientist"), None)
    queued_review = next((t for t in pending if t.role == "chief_scientist"), None)
    if chief_since is not None:
        chief = ("working", ago(chief_since), "answer your message" + (" · reviewing the agents' reports" if review
                                                                       else ""))
    elif review is not None:
        chief = ("working", ago(review.started_at), "review the agents' reports")
    elif queued_review is not None:
        chief = ("queued", ago(at(queued_review.queued_at)), "review the agents' reports")
    else:
        last = runs_of(role="chief_scientist")
        did = (("last: review the agents' reports" if last[-1].run.task_id else "last: answer your message")
               if last else "—")
        chief = ("idle", ago(at(last[-1].recorded_at)) if last else "", did)
    rows = [AgentRow("chief", "Chief Scientist", "plans and coordinates", *chief)]

    if state is not None:
        rows += [_hypothesis_agent_row(state, a, mine, pending, ago, at, events) for a in state.agents.values()
                 if a.active]
        proposal = state.pending_proposal
        for spec in proposal.agents if proposal is not None else ():
            if proposal.feedback_ids:
                when = "held for your reply"
            elif state.automation is not None:
                when = "for the approver"
            else:
                left = minutes_left(state, proposal, now)
                when = f"starts in {age(left * 60)}" if left > 0 else "starting"
            rows.append(AgentRow(f"proposed:{spec.hypothesis_id}", agent_id_for(state, spec.hypothesis_id),
                                 _short(spec.role, 32), f"proposed · {when}",
                                 ago(at(proposal.recorded_at)), _short(spec.brief, 70), spec.hypothesis_id))
        rows += [AgentRow(a.agent_id, a.agent_id, _short(a.spec.role, 32), "retired",
                          ago(at(a.retired_at)), _short(a.retire_reason or "—", 70), a.hypothesis_id)
                 for a in state.agents.values() if not a.active]

    # The Evidence Judge: one at a time per investigation.
    judging = next((a for a in mine if a.role == "evidence_judge"), None)
    queued = [t for t in pending if t.role == "evidence_judge"]
    automated = state is not None and state.automation is not None
    if judging is not None:
        judge = ("working", ago(judging.started_at), f"assess {judging.hypothesis_id}", judging.hypothesis_id)
    elif queued:
        more = f" (and {len(queued) - 1} more)" if len(queued) > 1 else ""
        judge = ("queued", ago(at(queued[0].queued_at)), f"assess {queued[0].hypothesis_id}{more}",
                 queued[0].hypothesis_id)
    else:
        last = runs_of(role="evidence_judge")
        if last and last[-1].run.task_id:
            done = state.tasks[last[-1].run.task_id]
            judge = ("idle", ago(at(last[-1].recorded_at)), f"last: assess {done.hypothesis_id}", None)
        else:
            judge = ("idle", "", "assesses new evidence each round" if automated else
                     "assesses when the Chief Scientist asks", None)
    rows.append(AgentRow("judge", "Evidence Judge", "assesses evidence", *judge[:3], hypothesis_id=judge[3]))

    # The approver: only in full automation, or once it has worked.
    if state is not None and (automated or any(t.role == "approver" for t in state.tasks.values())):
        deciding = next((a for a in mine if a.role == "approver"), None)
        waiting = next((t for t in pending if t.role == "approver"), None)
        if not automated:
            approver = ("off", "", "only in full automation")
        elif deciding is not None:
            approver = ("working", ago(deciding.started_at), "decide on proposals and submissions")
        elif waiting is not None:
            approver = ("queued", ago(at(waiting.queued_at)), "decide on proposals and submissions")
        else:
            last = runs_of(role="approver")
            approver = ("idle", ago(at(last[-1].recorded_at)) if last else "", "nothing waits for a decision")
        rows.append(AgentRow("approver", "Approver", "decides for you", *approver))
    return rows


def _hypothesis_agent_row(state: InvestigationState, agent: AgentRecord, mine: Sequence[Activity],
                          pending: Sequence[AgentTaskRecord], ago, at, events: Sequence[RecordedEvent]) -> AgentRow:
    working = next((a for a in mine if a.agent_id == agent.agent_id), None)
    queued = [t for t in pending if t.agent_id == agent.agent_id]
    hyp = state.hypotheses[agent.hypothesis_id]
    if working is not None:
        step = f" · step {working.last_step}" if working.last_step else ""
        status = ("working", ago(working.started_at), task_label(state.tasks.get(working.task_id)) + step)
    elif queued:
        status = ("queued", ago(at(queued[0].queued_at)), task_label(queued[0]))
    elif hyp.status is HypothesisStatus.WAITING_FOR_JOB:
        since = next((e.stream_version for e in reversed(events) if isinstance(e.payload, HypothesisStatusChanged)
                      and e.payload.hypothesis_id == agent.hypothesis_id
                      and e.payload.status is HypothesisStatus.WAITING_FOR_JOB), None)
        status = ("waiting for jobs", ago(at(since)),
                  "calculations " + ", ".join(short_id(c) for c in hyp.awaiting_calculation_ids))
    else:
        runs = [r for r in state.agent_runs if r.actor_id == agent.agent_id]
        if runs:
            done = state.tasks.get(runs[-1].run.task_id) if runs[-1].run.task_id else None
            status = ("idle", ago(at(runs[-1].recorded_at)), f"last: {task_label(done)}")
        else:
            status = ("idle", ago(at(agent.started_at)), "not run yet")
    return AgentRow(agent.agent_id, agent.agent_id, _short(agent.spec.role, 32), *status,
                    hypothesis_id=agent.hypothesis_id)


def swarm_rows(state: InvestigationState) -> list[SwarmRow]:
    rows = []
    for hyp in state.hypotheses.values():
        calcs = [state.calculations[c] for c in hyp.calculation_ids]
        qualities = Counter(r.evidence.quality.value for r in hypothesis_evidence(state, hyp.hypothesis_id))
        last = hyp.assessments[-1] if hyp.assessments else None
        rows.append(SwarmRow(
            hypothesis_id=hyp.hypothesis_id,
            lifecycle=hyp.status.value,
            priority=state.plan.priority(hyp.hypothesis_id).value,
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
