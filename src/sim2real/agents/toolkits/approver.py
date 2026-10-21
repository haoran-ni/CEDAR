"""Tools of the approver: in full automation, decide on the researcher's behalf whether proposed agents start and
whether calculations are submitted. Nothing else: inputs needing review, cancellations and closing stay with the
researcher, and the record refuses the approver anything outside full automation.
"""

from dataclasses import dataclass

from pydantic import Field

from sim2real.agents.tools import AgentTool, ToolFailure
from sim2real.automation import mode_summary, pending_decisions
from sim2real.contracts.base import Actor, Contract, Identifier, NonEmptyStr, new_id
from sim2real.contracts.events import ApproverDecision, EventDraft, ProposalWithdrawn
from sim2real.contracts.plan import PermissionAction
from sim2real.state.ledger import ResearchLedger
from sim2real.state.projection import InvestigationState
from sim2real.summaries import calculation_summary, investigation_summary, recommendation_text
from sim2real.swarm import start_drafts


@dataclass
class ApproverContext:
    ledger: ResearchLedger
    investigation_id: str
    actor: Actor


class NoArgs(Contract):
    pass


class DecisionArgs(Contract):
    target_id: Identifier = Field(description="The proposal or calculation decided on.")
    reason: NonEmptyStr = Field(description="Why, in a sentence or two, for the researcher and the Chief Scientist.")


def decisions_text(state: InvestigationState) -> str:
    """What waits for the approver, in detail, with the Chief Scientist's latest recommendation."""
    lines = [f"Mode: {mode_summary(state)}."]
    decisions = pending_decisions(state)
    if not decisions:
        lines.append("Nothing waits for your decision.")
    for decision in decisions:
        if decision.action is PermissionAction.START_AGENTS:
            proposal = state.pending_proposal
            lines += ["", f"Agents proposed by the Chief Scientist ({proposal.proposal_id}). Why: {proposal.rationale}"]
            for spec in proposal.agents:
                limits = ", ".join(x for x in (f"{spec.token_budget:,} tokens" if spec.token_budget else "",
                                               f"{spec.max_turns} turns per run" if spec.max_turns else "") if x)
                lines.append(f"  - {spec.hypothesis_id} ({state.hypotheses[spec.hypothesis_id].spec.statement}): "
                             f"{spec.role}" + (f" ({limits})" if limits else "") + f"; first task: {spec.brief}")
        else:
            calc = state.calculations[decision.target_id]
            lines += ["", f"Submission of {calc.calculation_id}, requesting {calc.examination.requested_core_hours:g} "
                          "core-hours:", calculation_summary(state, calc)]
    if decisions and state.recommendations:
        lines += ["", recommendation_text(list(state.recommendations.values())[-1])]
    return "\n".join(lines)


def approver_tools(ctx: ApproverContext) -> list[AgentTool]:
    def decide(args: DecisionArgs, approved: bool) -> str:
        found: dict = {}

        def build(state: InvestigationState) -> list[EventDraft]:
            proposal = state.pending_proposal
            if proposal is not None and proposal.proposal_id == args.target_id:
                action = PermissionAction.START_AGENTS
            elif args.target_id in state.calculations:
                action = PermissionAction.SUBMIT
            else:
                raise ToolFailure(f"{args.target_id} is neither the proposal waiting to start nor a calculation")
            decision_id = new_id("decision")
            drafts = [EventDraft(actor=ctx.actor, payload=ApproverDecision(
                decision_id=decision_id, action=action, target_id=args.target_id, approved=approved,
                reason=args.reason))]
            if action is PermissionAction.START_AGENTS:
                drafts += (start_drafts(state, proposal, ctx.actor, decision_id) if approved else [EventDraft(
                    actor=ctx.actor, payload=ProposalWithdrawn(proposal_id=proposal.proposal_id,
                                                               reason=f"declined by the approver: {args.reason}"))])
            found.update(action=action, decision_id=decision_id)
            return drafts

        result = ctx.ledger.record_with(ctx.investigation_id, build)
        decision_id, state = found["decision_id"], result.state
        if found["action"] is PermissionAction.START_AGENTS:
            if not approved:
                return (f"Declined {args.target_id} ({decision_id}); it is withdrawn, and the Chief Scientist sees "
                        "your reason at its next review.")
            started = result.events[1].payload.agent_ids
            return (f"Approved {args.target_id} ({decision_id}): started "
                    f"{', '.join(f'{a} on {h}' for h, a in started.items())}. Their first tasks are queued.")
        if not approved:
            return (f"Declined the submission of {args.target_id} ({decision_id}); it waits for the researcher, and "
                    "the Chief Scientist sees your reason at its next review.")
        hours = state.approvals[decision_id].core_hours
        return (f"Approved the submission of {args.target_id} ({decision_id}, {hours:g} core-hours; "
                f"{state.automation_core_hours_left:g} left). The dispatcher submits it.")

    return [
        AgentTool("investigation_state", "The investigation's state: hypotheses, calculations with what every "
                  "hypothesis predicted for them, evidence, the latest recommendation and the mode.", NoArgs,
                  lambda _: investigation_summary(ctx.ledger.load(ctx.investigation_id))),
        AgentTool("decisions", "What waits for your decision, in detail.", NoArgs,
                  lambda _: decisions_text(ctx.ledger.load(ctx.investigation_id))),
        AgentTool("approve", "Approve: start the proposed agents, or let the calculation be submitted.", DecisionArgs,
                  lambda a: decide(a, True)),
        AgentTool("decline", "Decline: withdraw the proposal, or leave the calculation unsubmitted for the "
                  "researcher; the Chief Scientist sees your reason.", DecisionArgs, lambda a: decide(a, False)),
    ]
