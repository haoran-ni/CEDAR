"""The hypothesis-agent swarm: proposals the researcher reviews, the agents, and the Chief Scientist's reviews.

The Chief Scientist proposes agents, one per hypothesis it wants tested. A proposal starts
when the researcher types ``/approve agents``, or by itself once its review window has
passed without a message from the researcher; any message holds it until the Chief
Scientist revises it. Each agent records work for its own hypothesis only, and can see and
read the others' work. When every agent that was given work has finished its current step,
and no proposed agents are about to start, the dispatcher wakes the Chief Scientist to collect
and compare what they reported, together with the evidence and assessments recorded since its
last review.
"""

from datetime import datetime

from cedar.contracts.base import Actor, ActorKind, new_id
from cedar.contracts.events import (
    AgentRetired,
    AgentsStarted,
    AgentTaskQueued,
    EventDraft,
    InvestigationStatusChanged,
    ModeChanged,
    ProposalWithdrawn,
)
from cedar.contracts.plan import Priority
from cedar.models.prices import dollars
from cedar.contracts.status import INVESTIGATION_MACHINE, CalculationState, InvestigationMode, InvestigationStatus
from cedar.state.projection import (
    AgentRecord,
    AgentRunRecord,
    AgentTaskRecord,
    ClosingRecord,
    EvidenceRecord,
    InvestigationState,
    ProposalRecord,
    WorkbenchRecord,
    agent_id_for,
    evidence_owner,
    review_deadline,
)

CHIEF = Actor(kind=ActorKind.CHIEF_SCIENTIST, id="chief-scientist")
MAX_CONTINUATIONS = 2
"""How many times a task cut off before its end is carried on with: by a temporary model-service error, or by the
investigation leaving the states its role works in."""
FIRST_TASK = "Your first task, from the Chief Scientist: "
"""How an agent's first task, its brief, begins."""
WAKE_TASK = "Your calculations have ended and their runs were checked:"
"""How the task begins that the dispatcher queues for an agent whose calculations have ended."""
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
            agent_id=ids[spec.hypothesis_id], instruction=FIRST_TASK + spec.brief)))
    return drafts


def _wind_down(state: InvestigationState, actor: Actor, reason: str) -> list[EventDraft]:
    """Withdraw a waiting proposal of agents and retire every agent."""
    drafts = []
    proposal = state.pending_proposal
    if proposal is not None:
        drafts.append(EventDraft(actor=actor, payload=ProposalWithdrawn(proposal_id=proposal.proposal_id,
                                                                        reason=reason)))
    drafts += [EventDraft(actor=actor, payload=AgentRetired(agent_id=a.agent_id, reason=reason))
               for a in state.agents.values() if a.active]
    return drafts


def close_drafts(state: InvestigationState, closing: ClosingRecord, actor: Actor,
                 authorization_id: str) -> list[EventDraft]:
    """Close the investigation as the Chief Scientist proposed, on the researcher's command: withdraw a waiting
    proposal of agents, retire every agent, then close it for good."""
    reason = f"closed by the researcher ({authorization_id}), as the Chief Scientist proposed ({closing.closing_id})"
    return [*_wind_down(state, actor, reason), EventDraft(actor=actor, payload=InvestigationStatusChanged(
        status=closing.status, reason=reason, authorization_id=authorization_id))]


def halt_drafts(state: InvestigationState, actor: Actor, reason: str,
                intervention_id: str | None = None) -> list[EventDraft]:
    """Withdraw a waiting proposal, retire every agent, switch full automation off and pause the investigation; a
    run in progress stops at its next model call, and nothing new starts until the investigation is resumed."""
    drafts = _wind_down(state, actor, reason)
    if state.automation is not None:
        drafts.append(EventDraft(actor=actor, payload=ModeChanged(
            mode=InvestigationMode.HUMAN_IN_THE_LOOP, reason=reason, intervention_id=intervention_id)))
    if state.status not in INVESTIGATION_MACHINE.suspend | INVESTIGATION_MACHINE.terminal:
        drafts.append(EventDraft(actor=actor, payload=InvestigationStatusChanged(
            status=InvestigationStatus.PAUSED, reason=reason)))
    return drafts


def agent_label(agent: AgentRecord) -> str:
    return f"{agent.tag} {agent.agent_id} ({agent.spec.role})"


def agent_prompt(agent: AgentRecord) -> str:
    """What a hypothesis agent is told about itself, after the instructions all agents share."""
    return (f"\n\nYour assignment: you are {agent.tag} ({agent.agent_id}), the {agent.spec.role} for hypothesis "
            f"{agent.hypothesis_id}. Everything you record is for that hypothesis; you can read the other agents' "
            "work.")


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
            limits = f"budget {dollars(spec.budget_usd)}" if spec.budget_usd else ""
            lines += [f"  - {spec.hypothesis_id}: {spec.role}" + (f" ({limits})" if limits else ""),
                      f"    first task: {spec.brief}"]
    active = [a for a in state.agents.values() if a.active]
    retired = [a for a in state.agents.values() if not a.active]
    if active:
        lines.append("Agents:")
        for agent in active:
            budget = agent.spec.budget_usd
            lines.append(f"  - {agent_label(agent)} on {agent.hypothesis_id}; {dollars(state.agent_cost(agent.agent_id))} "
                         "spent" + (f" of {dollars(budget)}" if budget else ""))
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
    """Queued or running agent work the dispatcher will carry out now; a review waits for all of it. A hypothesis
    agent works only while the investigation is INVESTIGATING, so its work waits, without holding a review back,
    while the investigation is REVIEWING."""
    return [t for t in state.pending_tasks if t.role in REVIEW_ROLES
            and not (t.hypothesis_id and state.plan.priority(t.hypothesis_id) is Priority.ON_HOLD)
            and not (t.role == "hypothesis_agent" and state.status is not InvestigationStatus.INVESTIGATING)]


def review_due(state: InvestigationState) -> bool:
    """Whether agents reported, or the approver declined something, since the last review, none of them has work
    left in hand, and no proposed agents are about to start: those are reviewed together with the rest. A proposal
    the researcher's reply holds does not hold the review back, nor one that cannot start because the investigation
    is not INVESTIGATING."""
    if state.status not in (InvestigationStatus.INVESTIGATING, InvestigationStatus.REVIEWING):
        return False
    reviews = _reviews(state)
    if any(t.completed_by is None for t in reviews) or blocking_tasks(state):
        return False
    if any(t.role == "approver" and t.completed_by is None and t.dropped_at is None for t in state.tasks.values()):
        return False  # its decisions are part of what is reviewed
    proposal = state.pending_proposal
    if proposal is not None and not proposal.feedback_ids and state.status is InvestigationStatus.INVESTIGATING:
        return False
    since = reviews[-1].queued_at if reviews else 0
    return (any(r.role in REVIEW_ROLES and r.recorded_at > since for r in state.agent_runs)
            or any(not a.approved and a.recorded_at > since for a in state.approvals.values()))


REVIEW_INSTRUCTION = ("Review the agents' reports: the hypothesis agents and the Evidence Judge have finished the work "
                      "they were given.")


def _evidence_line(state: InvestigationState, record: EvidenceRecord) -> str:
    """One piece of evidence as the record holds it: whose, from what, the value, its quality and failed checks."""
    ev = record.evidence
    runs = ev.calculation_ids or ((ev.calculation_id,) if ev.calculation_id else ())
    owner = evidence_owner(state, ev)
    hypotheses = [owner] if owner else sorted({h for c in runs for h in state.calculations[c].hypothesis_ids})
    if ev.analysis_id is not None:
        origin = f"analysis {ev.analysis_id} of {', '.join(runs)}"
    elif ev.calculation_id is not None:
        origin = f"reading of {ev.calculation_id}"
    else:
        origin = ev.kind.value.replace("_", " ")
    value = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
    failed = [c.name for c in ev.quality_checks if not c.passed]
    return (f"- {record.evidence_id} for {', '.join(hypotheses) or 'no hypothesis'} ({origin}): "
            f"{ev.observable.name} = {value}; {ev.quality.value}"
            + (f"; failed checks: {', '.join(failed)}" if failed else "")
            + (f"; not comparable when: {'; '.join(ev.non_comparability)}" if ev.non_comparability else ""))


def _commands_during(state: InvestigationState, run: AgentRunRecord) -> list[WorkbenchRecord]:
    """The workbench commands an agent ran during one of its runs: since its previous run ended."""
    earlier = max((r.recorded_at for r in state.agent_runs if r.actor_id == run.actor_id
                   and r.recorded_at < run.recorded_at), default=0)
    return [w for w in state.workbench_runs.values()
            if w.agent_id == run.actor_id and earlier < w.recorded_at < run.recorded_at]


def review_material(state: InvestigationState, task: AgentTaskRecord) -> str:
    """What the Chief Scientist reviews: the reports, the evidence and the Evidence Judge's assessments recorded, and
    the researcher's messages, since the previous review."""
    reviews = _reviews(state)
    first = task_chain(state, task)[0]  # a review carried on after it was cut off covers what the first one would
    before = [t for t in reviews if t.queued_at < first.queued_at]
    since = before[-1].queued_at if before else 0
    lines = ["This is a background review: the researcher is not in this conversation, but sees your final message "
             "in the chat."]
    automation = state.automation
    if automation is not None:
        lines.append(f"The investigation is in full automation ({automation.intervention_id or 'switched on'}): the "
                     "researcher is away. An approver decides on your agent proposals and on submissions, within "
                     f"{automation.max_core_hours:g} core-hours and {dollars(automation.max_usd)} of model use from the "
                     "switch; "
                     "inputs needing review and closing wait for the researcher. Do not wait for answers from the "
                     "researcher.")
    lines.append("Reports since your last review, in the agents' own words:")
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
        commands = _commands_during(state, run)
        if commands:
            lines.append("Workbench commands it ran (context, not evidence):")
            lines += [f"- {w.run_id}: {w.run.outcome.value}: {w.run.command}" for w in commands]
    evidence = [r for r in state.evidence.values() if since < r.recorded_at <= task.queued_at]
    lines += ["", "Evidence recorded since your last review, as the record holds it:"]
    lines += [_evidence_line(state, r) for r in evidence] or ["None."]
    assessments = [(h, a) for h, hyp in state.hypotheses.items() for a in hyp.assessments
                   if since < a.recorded_at <= task.queued_at]
    if assessments:
        lines += ["", "The Evidence Judge's assessments since your last review:"]
        lines += [f"- {h}: {a.assessment.value} ({a.completeness.value.lower()}), citing "
                  f"{', '.join(a.evidence_ids) or 'no evidence'}: {a.rationale}" for h, a in assessments]
    decisions = [a for a in state.approvals.values() if since < a.recorded_at <= task.queued_at]
    if decisions:
        lines += ["", "The approver's decisions since your last review:"]
        lines += [f"- {'approved' if a.approved else 'declined'} "
                  f"{'starting the agents of' if a.action.value == 'start_agents' else 'submitting'} {a.target_id}: "
                  f"{a.reason}" for a in decisions]
    messages = [i for i in state.interventions.values() if since < i.recorded_at <= task.queued_at]
    if messages:
        lines += ["", "The researcher's messages meanwhile:"]
        lines += [f"[{i.intervention_id}] {i.text}" for i in messages]
    lines += ["", "Collect and compare these results across hypotheses. A report is its agent's account, not "
              "evidence: weigh the recorded evidence and the Judge's assessments above, and check investigation_state "
              "for the rest. Each agent concentrates on its own hypothesis: comparing across hypotheses, "
              "noticing confounders, contradictions and interactions, is your job. Record that comparison with "
              "record_synthesis: every open hypothesis, how much of the discrepancy each accounts for (citing "
              "evidence), how they interact, what remains unexplained and what cannot be separated; when the "
              "comparison suggests new hypotheses, propose them (propose_hypothesis, then propose_agents) and name "
              "them in it. Then decide what follows, if "
              "anything, recording the candidates you weighed and your choice with recommend_next_actions before "
              "you act on it: further work for an agent (delegate; name another agent's calculations or evidence "
              "when its test needs them), an assessment (request_judgement), new hypotheses and agents "
              "(propose_agents), a plan revision, or moving the investigation on. The researcher may be talking with "
              "you in the chat at the same time; the record shows what has already been done. End with a short "
              "message for the researcher: what came in, what it means, what you did, and what waits for them. If "
              "you proposed closing the investigation, say so first: as what and why, and that /approve close "
              "closes it."]
    return "\n".join(lines)


# -- runs that end early --------------------------------------------------------------------------

def task_chain(state: InvestigationState, task: AgentTaskRecord) -> list[AgentTaskRecord]:
    """The task and the ones it continues, the first one first."""
    chain = [task]
    while chain[0].continues is not None and chain[0].continues in state.tasks:
        chain.insert(0, state.tasks[chain[0].continues])
    return chain


def continuation_text(state: InvestigationState, task: AgentTaskRecord, why: str) -> str:
    """The instruction that carries on with a task whose run was cut off."""
    first = task_chain(state, task)[0]
    return (f"Continue task {first.task_id}: the run carrying it out was cut off ({why}). The briefing above shows "
            f"what was recorded until then. The task was: {first.instruction}")


def ended_unseen(state: InvestigationState, hypothesis_id: str, since: int) -> list[str]:
    """The hypothesis's calculations whose jobs ended after ``since``: their runs were checked, or they were
    cancelled. Registered runs never ran here, and an examiner's rejection is answered at once."""
    found = []
    for cid in state.hypotheses[hypothesis_id].calculation_ids:
        calc = state.calculations[cid]
        if calc.registered:
            continue
        ended = next((v for v, s in calc.history if s in (CalculationState.QUALITY_ASSESSED,
                                                           CalculationState.CANCELLED)), None)
        if ended is not None and ended > since:
            found.append(cid)
    return found


def account_of_run(state: InvestigationState, task: AgentTaskRecord, actor_id: str, since: int, *, outcome: str,
                   error: str | None, turns: int, last_step: str | None) -> str:
    """CEDAR's account of a run that ended without the agent's own report, from what the record holds."""
    how = f"it ended {outcome}" + (f" ({error})" if error else "") + f" after {turns} model call{'s' * (turns != 1)}"
    lines = [f"No report from the agent: {how}" + (f"; its last step was {last_step}" if last_step else "") + ". "
             "CEDAR's account from the record of what it did in this run:"]
    h = task.hypothesis_id
    if task.role == "hypothesis_agent" and h in state.hypotheses:
        hyp = state.hypotheses[h]
        lines += [f"- requested {c}: {state.calculations[c].state.value}" for c in hyp.calculation_ids
                  if state.calculations[c].requested_at > since and not state.calculations[c].registered]
        lines += [f"- registered {c} from {state.calculations[c].registration.source}" for c in hyp.calculation_ids
                  if state.calculations[c].requested_at > since and state.calculations[c].registered]
        lines += [f"- registered prediction {p}: {state.predictions[p].prediction.statement}"
                  for p in hyp.prediction_ids if state.predictions[p].registered_at > since]
        lines += [f"- ran analysis {a.analysis_id} ({a.run.outcome.value}): {a.run.purpose}"
                  for a in state.analyses.values() if a.hypothesis_id == h and a.recorded_at > since]
        lines += [f"- recorded evidence {r.evidence_id}: {r.evidence.observable.name} = "
                  + (f"{r.evidence.value.value!r} {r.evidence.value.unit}" if r.evidence.value else "no value")
                  + f" ({r.evidence.quality.value})" for r in state.evidence.values()
                  if r.recorded_at > since and r.evidence.interpreted_by is not None
                  and r.evidence.interpreted_by.id == actor_id]
        commands = [w for w in state.workbench_runs.values() if w.agent_id == actor_id and w.recorded_at > since]
        if commands:
            lines.append(f"- ran {len(commands)} workbench command{'s' * (len(commands) != 1)}, the last: "
                         f"{commands[-1].run.command}")
        lines.append(f"- its hypothesis is now {hyp.status.value}")
    elif task.role == "evidence_judge" and h in state.hypotheses:
        lines += [f"- assessed {h}: {a.assessment.value} ({a.completeness.value})"
                  for a in state.hypotheses[h].assessments if a.recorded_at > since]
    else:
        lines += [f"- recommended next steps {r.recommendation_id}" for r in state.recommendations.values()
                  if r.recorded_at > since]
        lines += [f"- queued {t.task_id} for {t.agent_id or t.role}" for t in state.tasks.values()
                  if t.queued_at > since and t.queued_by == actor_id]
    if len(lines) == 1:
        lines.append("- nothing was recorded")
    return "\n".join(lines)
