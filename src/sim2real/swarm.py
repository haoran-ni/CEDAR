"""The hypothesis-agent swarm: proposals the researcher reviews, the agents, and the Chief Scientist's reviews.

The Chief Scientist proposes agents, one per hypothesis it wants tested. A proposal starts
when the researcher types ``/approve agents``, or by itself once its review window has
passed without a message from the researcher; any message holds it until the Chief
Scientist revises it. Agents work independently and see only their own hypothesis. When
every agent that was given work has finished its current step, the dispatcher wakes the
Chief Scientist to collect and compare what they reported.
"""

from datetime import datetime

from sim2real.contracts.base import Actor, ActorKind, new_id
from sim2real.contracts.events import (
    AgentRetired,
    AgentsStarted,
    AgentTaskQueued,
    EventDraft,
    InvestigationStatusChanged,
    ProposalWithdrawn,
)
from sim2real.contracts.plan import Priority
from sim2real.contracts.status import INVESTIGATION_MACHINE, InvestigationStatus
from sim2real.state.projection import (
    AgentRecord,
    AgentTaskRecord,
    InvestigationState,
    ProposalRecord,
    agent_id_for,
    review_deadline,
)

CHIEF = Actor(kind=ActorKind.CHIEF_SCIENTIST, id="chief-scientist")
REVIEW_ROLES = ("hypothesis_agent", "evidence_judge")
"""Roles whose reports the Chief Scientist reviews."""


def clock_time(moment: datetime) -> str:
    return moment.astimezone().strftime("%H:%M %Z")


def minutes_left(state: InvestigationState, proposal: ProposalRecord, now: datetime) -> float:
    return (review_deadline(state, proposal) - now).total_seconds() / 60


def proposal_note(state: InvestigationState, proposal: ProposalRecord) -> str:
    """Where a waiting proposal stands, for the researcher; it does not change with the clock."""
    n = len(proposal.agents)
    agents = f"{n} hypothesis agent{'s' if n > 1 else ''} proposed ({proposal.proposal_id})"
    if proposal.feedback_ids:
        return (f"{agents}, held since your message {proposal.feedback_ids[0]}: the Chief Scientist revises the "
                "proposal, or you start it as it stands")
    return f"{agents}; they start at {clock_time(review_deadline(state, proposal))} unless you reply"


def start_drafts(state: InvestigationState, proposal: ProposalRecord, actor: Actor,
                 authorization_id: str | None = None) -> list[EventDraft]:
    """Start a proposal's agents and queue each one's brief as its first task."""
    ids = {spec.hypothesis_id: agent_id_for(state, spec.hypothesis_id) for spec in proposal.agents}
    drafts = [EventDraft(actor=actor, payload=AgentsStarted(proposal_id=proposal.proposal_id, agent_ids=ids,
                                                            authorization_id=authorization_id))]
    for spec in proposal.agents:
        drafts.append(EventDraft(actor=actor, payload=AgentTaskQueued(
            task_id=new_id("task"), role="hypothesis_agent", hypothesis_id=spec.hypothesis_id,
            agent_id=ids[spec.hypothesis_id], instruction=f"Your first task, from the Chief Scientist: {spec.brief}")))
    return drafts


def halt_drafts(state: InvestigationState, actor: Actor, reason: str) -> list[EventDraft]:
    """Withdraw a waiting proposal, retire every agent and pause the investigation; a run in progress stops at its
    next model call, and nothing new starts until the investigation is resumed."""
    drafts = []
    proposal = state.pending_proposal
    if proposal is not None:
        drafts.append(EventDraft(actor=actor, payload=ProposalWithdrawn(proposal_id=proposal.proposal_id,
                                                                        reason=reason)))
    drafts += [EventDraft(actor=actor, payload=AgentRetired(agent_id=a.agent_id, reason=reason))
               for a in state.agents.values() if a.active]
    if state.status not in INVESTIGATION_MACHINE.suspend | INVESTIGATION_MACHINE.terminal:
        drafts.append(EventDraft(actor=actor, payload=InvestigationStatusChanged(
            status=InvestigationStatus.PAUSED, reason=reason)))
    return drafts


def agent_label(agent: AgentRecord) -> str:
    return f"{agent.agent_id} ({agent.spec.role})"


def agent_prompt(agent: AgentRecord) -> str:
    """What a hypothesis agent is told about itself, after the instructions all agents share."""
    return (f"\n\nYour assignment: you are {agent.agent_id}, the {agent.spec.role} for hypothesis "
            f"{agent.hypothesis_id}. Every tool acts on that hypothesis only.")


def agents_summary(state: InvestigationState, now: datetime) -> str:
    """The agents and any proposal waiting for the researcher, as text for the researcher and the Chief."""
    lines = []
    proposal = state.pending_proposal
    if proposal is not None:
        left = minutes_left(state, proposal, now)
        if proposal.feedback_ids:
            when = (f"held: the researcher replied ({', '.join(proposal.feedback_ids)}); it waits for a revised "
                    "proposal or /approve agents")
        elif state.status is not InvestigationStatus.INVESTIGATING:
            when = f"waits until the investigation is INVESTIGATING again (it is {state.status.value})"
        elif left > 0:
            when = (f"starts at {clock_time(review_deadline(state, proposal))} (in {max(1, round(left))} min) unless "
                    "the researcher replies; /approve agents starts it now")
        else:
            when = "the review window has passed; it starts at the dispatcher's next pass"
        lines += [f"Proposed by the Chief Scientist ({proposal.proposal_id}), {when}.",
                  f"  Why: {proposal.rationale}"]
        for spec in proposal.agents:
            limits = ", ".join(x for x in (f"{spec.token_budget:,} tokens" if spec.token_budget else "",
                                           f"{spec.max_turns} turns per run" if spec.max_turns else "") if x)
            lines += [f"  - {spec.hypothesis_id}: {spec.role}" + (f" ({limits})" if limits else ""),
                      f"    first task: {spec.brief}"]
    active = [a for a in state.agents.values() if a.active]
    retired = [a for a in state.agents.values() if not a.active]
    if active:
        lines.append("Agents:")
        for agent in active:
            budget = agent.spec.token_budget
            used = state.agent_tokens_used(agent.agent_id)
            lines.append(f"  - {agent_label(agent)} on {agent.hypothesis_id}; {used:,} tokens used"
                         + (f" of {budget:,}" if budget else ""))
    if retired:
        lines.append("Retired: " + "; ".join(f"{agent_label(a)} on {a.hypothesis_id}: {a.retire_reason}"
                                              for a in retired))
    without = [h for h, hyp in state.hypotheses.items()
               if state.active_agent(h) is None and hyp.status.value not in ("DISCARDED", "SUPERSEDED")
               and not (proposal and any(s.hypothesis_id == h for s in proposal.agents))]
    if without:
        lines.append("Open hypotheses without an agent: " + ", ".join(without))
    return "\n".join(lines) or "No agents yet."


# -- the Chief Scientist's reviews ------------------------------------------------------------

def _reviews(state: InvestigationState) -> list[AgentTaskRecord]:
    return [t for t in state.tasks.values() if t.role == "chief_scientist"]


def blocking_tasks(state: InvestigationState) -> list[AgentTaskRecord]:
    """Queued or running agent work the dispatcher will carry out; a review waits for all of it."""
    return [t for t in state.pending_tasks if t.role in REVIEW_ROLES
            and not (t.hypothesis_id and state.plan.priority(t.hypothesis_id) is Priority.ON_HOLD)]


def review_due(state: InvestigationState) -> bool:
    """Whether agents reported since the last review and none of them has work left in hand."""
    if state.status not in (InvestigationStatus.INVESTIGATING, InvestigationStatus.REVIEWING):
        return False
    reviews = _reviews(state)
    if any(t.completed_by is None for t in reviews) or blocking_tasks(state):
        return False
    since = reviews[-1].queued_at if reviews else 0
    return any(r.role in REVIEW_ROLES and r.recorded_at > since for r in state.agent_runs)


REVIEW_INSTRUCTION = ("Review the agents' reports: the hypothesis agents and the Evidence Judge have finished the work "
                      "they were given.")


def review_material(state: InvestigationState, task: AgentTaskRecord) -> str:
    """What the Chief Scientist reviews: the reports and researcher messages since the previous review."""
    reviews = _reviews(state)
    before = [t for t in reviews if t.queued_at < task.queued_at]
    since = before[-1].queued_at if before else 0
    lines = ["This is a background review: the researcher is not in this conversation, but sees your final message "
             "in the chat. Reports since your last review, in the agents' own words:"]
    for run in state.agent_runs:
        if run.role not in REVIEW_ROLES or not since < run.recorded_at <= task.queued_at:
            continue
        work = state.tasks.get(run.run.task_id) if run.run.task_id else None
        if run.role == "evidence_judge":
            who = f"Evidence Judge on {work.hypothesis_id}" if work else "Evidence Judge"
        else:
            agent = state.agents.get(run.actor_id)
            who = f"{agent_label(agent)} on {agent.hypothesis_id}" if agent else run.actor_id
        lines += ["", f"[{who}; run {run.run.outcome}]", (run.run.report or "(no report)").strip()]
    messages = [i for i in state.interventions.values() if since < i.recorded_at <= task.queued_at]
    if messages:
        lines += ["", "The researcher's messages meanwhile:"]
        lines += [f"[{i.intervention_id}] {i.text}" for i in messages]
    lines += ["", "Collect and compare these results across hypotheses; check investigation_state and the "
              "Evidence Judge's assessments first. Agents never see each other's work: noticing confounders, "
              "contradictions and interactions between hypotheses is your job. Then decide what follows, if "
              "anything: further work for an agent (delegate; pass on another agent's finding only when its test "
              "needs it, citing the evidence id), an assessment (request_judgement), new hypotheses and agents "
              "(propose_agents), a plan revision, or moving the investigation on. The researcher may be talking with "
              "you in the chat at the same time; the record shows what has already been done. End with a short "
              "message for the researcher: what came in, what it means, what you did, and what waits for them."]
    return "\n".join(lines)
