"""Tools of the Chief Scientist: plan, delegate and explain; never compute or authorize.

The Chief can open an investigation, record observable alignment, propose and manage
hypotheses, revise the plan, start rounds, and queue work for hypothesis agents and the
Evidence Judge. Submissions, cancellations and input approvals belong to the researcher
and are granted with explicit commands, never through these tools.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from sim2real.agents.tools import AgentTool, ToolFailure
from sim2real.contracts.base import (
    Actor,
    Contract,
    ExternalSource,
    Identifier,
    NonEmptyStr,
    Observable,
    Quantity,
    Uncertainty,
    UncertaintyKind,
    new_id,
)
from sim2real.contracts.events import (
    AgentTaskQueued,
    EventDraft,
    HypothesisProposed,
    HypothesisStatusChanged,
    InvestigationCreated,
    InvestigationStatusChanged,
    ObservableAlignmentRecorded,
    PlanRevised,
    RoundStarted,
)
from sim2real.contracts.hypothesis import HypothesisSpec, Quantifiability
from sim2real.contracts.investigation import (
    AlignmentStatus,
    ExperimentProvenance,
    InvestigationConstraints,
    InvestigationSpec,
    ObservableAlignment,
    SimulationProvenance,
)
from sim2real.contracts.plan import Priority, ResearchPlan
from sim2real.contracts.status import HypothesisStatus, InvestigationStatus
from sim2real.contracts.task import ProtocolRef
from sim2real.hpc.profile import ProfileStore
from sim2real.protocols.base import ProtocolRegistry
from sim2real.state.ledger import ResearchLedger
from sim2real.summaries import investigation_summary, pending_approvals


@dataclass
class ChiefContext:
    ledger: ResearchLedger
    profiles: ProfileStore
    registry_for: Callable[[str], ProtocolRegistry]
    """Protocols available under a compute profile, by profile name."""
    actor: Actor
    investigation_id: str | None = None
    on_open: Callable[[str], list[str]] | None = None
    """Told the new investigation's id once it is opened; returns the ids under which the researcher's
    earlier messages were recorded."""


class NoArgs(Contract):
    pass


class ObservableArg(Contract):
    name: NonEmptyStr
    definition: NonEmptyStr = Field(description="Exactly what is computed or measured, not just a label.")
    unit: str | None = None


class ValueArg(Contract):
    value: float
    unit: NonEmptyStr
    reference: str | None = Field(default=None, description="Zero or reference state, e.g. 'vacuum level'.")
    uncertainty: float | None = Field(default=None, ge=0)
    uncertainty_basis: str | None = Field(default=None, description="How the uncertainty was determined.")


class SimulationArg(Contract):
    label: NonEmptyStr
    value: ValueArg | None = None
    code: str | None = None
    code_version: str | None = None
    method: str | None = None
    xc: str | None = None
    dispersion: str | None = None
    basis: str | None = None
    k_sampling: str | None = None
    spin: str | None = None
    coverage: str | None = None
    boundary_conditions: str | None = None
    notes: str | None = None


class ExperimentArg(Contract):
    label: NonEmptyStr
    observable: ObservableArg
    technique: NonEmptyStr
    value: ValueArg | None = None
    temperature_kelvin: float | None = None
    preparation: str | None = None
    coverage: str | None = None
    environment: str | None = None
    citation: str | None = Field(default=None, description="Where the measurement is published, as given by the "
                                                           "researcher; never invent one.")
    notes: str | None = None


class OpenArgs(Contract):
    investigation_id: Identifier = Field(description="Short, readable id, e.g. co-cu111-site.")
    title: NonEmptyStr
    question: NonEmptyStr
    target_claim: NonEmptyStr = Field(description="The claim a resolution would support.")
    objective: NonEmptyStr = Field(description="The first plan's objective.")
    simulations: list[SimulationArg] = Field(min_length=1)
    experiments: list[ExperimentArg] = Field(min_length=1)
    simulated_observable: ObservableArg
    measured_observable: ObservableArg
    compute_profile: Identifier | None = Field(default=None, description="An approved compute profile.")
    model_token_budget: int | None = Field(default=None, gt=0, description="Total tokens agents may process.")


class InvestigationStatusArgs(Contract):
    status: Literal["ALIGNING_OBSERVABLES", "PLANNING", "INVESTIGATING", "REVIEWING", "COMPLETED", "INCONCLUSIVE",
                    "PAUSED", "BLOCKED"]
    reason: NonEmptyStr


class AlignmentArgs(Contract):
    status: Literal["ALIGNED", "REQUIRES_FORWARD_MODEL", "NOT_COMPARABLE"]
    rationale: list[NonEmptyStr] = Field(min_length=1)
    forward_model: str | None = Field(default=None, description="How the simulated quantity maps to the signal.")


class HypothesisArgs(Contract):
    hypothesis_id: Identifier
    title: NonEmptyStr
    statement: NonEmptyStr = Field(description="One precise, falsifiable claim.")
    rationale: NonEmptyStr
    falsifiers: list[NonEmptyStr] = Field(min_length=1, description="Observations that would contradict it.")
    discriminating_observations: list[NonEmptyStr] = []
    assumptions: list[NonEmptyStr] = []
    mechanism: str | None = None
    quantifiability: Literal["directly_quantifiable", "approximately_quantifiable", "indirectly_constrained",
                             "non_identifiable", "unknown"] = "unknown"
    depends_on: list[Identifier] = Field(default=[], description="Existing hypotheses only; propose those first.")
    interacts_with: list[Identifier] = Field(default=[], description="Existing hypotheses only; propose those first.")


class HypothesisStatusArgs(Contract):
    hypothesis_id: Identifier
    status: Literal["SCREENED", "ACTIVE", "PLANNING_TEST", "PAUSED", "NEEDS_DATA", "DISCARDED", "SUPERSEDED",
                    "FAILED", "ANALYZING", "ASSESSED"]
    reason: NonEmptyStr


class PlanArgs(Contract):
    reason: NonEmptyStr
    intervention_id: Identifier | None = Field(default=None, description="The researcher message that asked for "
                                                                         "this change, if any.")
    objective: str | None = Field(default=None, description="New objective; omit to keep the current one.")
    priorities: dict[Identifier, Literal["high", "normal", "low", "on_hold"]] = Field(
        default={}, description="Priority changes by hypothesis; others keep theirs. on_hold stops new calculations "
                                "but lets submitted ones finish.")
    notes: list[NonEmptyStr] = []


class ReasonArgs(Contract):
    reason: NonEmptyStr


class TaskArgs(Contract):
    hypothesis_id: Identifier
    instruction: NonEmptyStr = Field(description="Specific work: what to test or assess, with which protocol, and "
                                                 "any limits the researcher set.")


class ReportsArgs(Contract):
    hypothesis_id: Identifier | None = None
    last: int = Field(default=5, ge=1, le=20)


class ProtocolArgs(Contract):
    name: str
    version: str


def chief_tools(ctx: ChiefContext) -> list[AgentTool]:
    def current() -> str:
        if ctx.investigation_id is None:
            raise ToolFailure("no investigation is open yet; open one with open_investigation")
        return ctx.investigation_id

    def record(*payloads, key: str | None = None):
        drafts = [EventDraft(actor=ctx.actor, payload=p, idempotency_key=key) for p in payloads]
        return ctx.ledger.record(current(), drafts)

    def list_compute_profiles(_: NoArgs) -> str:
        lines = []
        for name in ctx.profiles.names():
            profile = ctx.profiles.load(name)
            approved = "approved" if ctx.profiles.is_approved(name) else "NOT approved (cannot be used)"
            codes = ", ".join(f"{c} {i.version}" for c, i in profile.codes.items())
            lines.append(f"{name}: {approved}; partitions {', '.join(profile.limits.partitions)}; codes: {codes}")
        return "\n".join(lines) or "No compute profiles. The researcher can create one with `sim2real setup`."

    def open_investigation(args: OpenArgs) -> str:
        if ctx.investigation_id is not None:
            raise ToolFailure(f"investigation {ctx.investigation_id} is already open in this conversation")
        if args.compute_profile is not None and not ctx.profiles.is_approved(args.compute_profile):
            raise ToolFailure(f"compute profile {args.compute_profile} is not approved")

        def quantity(v: ValueArg | None) -> Quantity | None:
            if v is None:
                return None
            uncertainties = () if v.uncertainty is None else (Uncertainty(
                kind=UncertaintyKind.UNSPECIFIED, value=v.uncertainty, basis=v.uncertainty_basis or "as reported"),)
            return Quantity(value=v.value, unit=v.unit, reference=v.reference, uncertainties=uncertainties)

        def observable(o: ObservableArg) -> Observable:
            return Observable(name=o.name, definition=o.definition, unit=o.unit)

        simulations = tuple(SimulationProvenance(
            label=s.label, quantity=quantity(s.value), code=s.code, code_version=s.code_version, method=s.method,
            xc=s.xc, dispersion=s.dispersion, basis=s.basis, k_sampling=s.k_sampling, spin=s.spin, coverage=s.coverage,
            boundary_conditions=s.boundary_conditions, conditions={"notes": s.notes} if s.notes else {},
        ) for s in args.simulations)
        experiments = tuple(ExperimentProvenance(
            label=e.label, observable=observable(e.observable), technique=e.technique, quantity=quantity(e.value),
            temperature=Quantity(value=e.temperature_kelvin, unit="K") if e.temperature_kelvin is not None else None,
            preparation=e.preparation, coverage=e.coverage, environment=e.environment,
            conditions={"notes": e.notes} if e.notes else {},
            sources=(ExternalSource(citation=e.citation),) if e.citation else (),
        ) for e in args.experiments)
        spec = InvestigationSpec(
            title=args.title, question=args.question, target_claim=args.target_claim, simulations=simulations,
            experiments=experiments,
            alignment=ObservableAlignment(simulated=observable(args.simulated_observable),
                                          measured=observable(args.measured_observable)),
            constraints=InvestigationConstraints(compute_profile=args.compute_profile,
                                                 model_token_budget=args.model_token_budget),
        )
        ctx.ledger.record(args.investigation_id, [EventDraft(
            actor=ctx.actor, payload=InvestigationCreated(spec=spec, plan=ResearchPlan(objective=args.objective)),
            idempotency_key="investigation_created")])
        ctx.investigation_id = args.investigation_id
        recorded = ctx.on_open(args.investigation_id) if ctx.on_open is not None else []
        note = (f" The researcher's messages so far are recorded as {', '.join(recorded)}; cite them when they "
                "motivate a plan change.") if recorded else ""
        return f"Opened investigation {args.investigation_id} (status CREATED). Next: align the observables.{note}"

    def investigation_state(_: NoArgs) -> str:
        return investigation_summary(ctx.ledger.load(current()))

    def set_investigation_status(args: InvestigationStatusArgs) -> str:
        record(InvestigationStatusChanged(status=InvestigationStatus(args.status), reason=args.reason))
        return f"Investigation is now {args.status}."

    def record_alignment(args: AlignmentArgs) -> str:
        state = ctx.ledger.load(current())
        alignment = ObservableAlignment(simulated=state.alignment.simulated, measured=state.alignment.measured,
                                        status=AlignmentStatus(args.status), forward_model=args.forward_model,
                                        rationale=tuple(args.rationale))
        record(ObservableAlignmentRecorded(alignment=alignment))
        discrepancy = ctx.ledger.load(current()).discrepancy()
        if discrepancy.comparable:
            return f"Alignment recorded. Discrepancy (simulated - measured): {discrepancy.difference.value} " \
                   f"{discrepancy.difference.unit}."
        return "Alignment recorded. No numerical discrepancy: " + "; ".join(discrepancy.reasons)

    def propose_hypothesis(args: HypothesisArgs) -> str:
        spec = HypothesisSpec(
            title=args.title, statement=args.statement, rationale=args.rationale, falsifiers=tuple(args.falsifiers),
            discriminating_observations=tuple(args.discriminating_observations), assumptions=tuple(args.assumptions),
            mechanism=args.mechanism, quantifiability=Quantifiability(args.quantifiability),
            depends_on=tuple(args.depends_on), interacts_with=tuple(args.interacts_with))
        record(HypothesisProposed(hypothesis_id=args.hypothesis_id, spec=spec))
        return f"Hypothesis {args.hypothesis_id} proposed (PROPOSED). Screen and activate it before delegating."

    def set_hypothesis_status(args: HypothesisStatusArgs) -> str:
        record(HypothesisStatusChanged(hypothesis_id=args.hypothesis_id, status=HypothesisStatus(args.status),
                                       reason=args.reason, awaiting_calculation_ids=()))
        return f"Hypothesis {args.hypothesis_id} is now {args.status}."

    def revise_plan(args: PlanArgs) -> str:
        state = ctx.ledger.load(current())
        plan = state.plan
        priorities = {**plan.hypothesis_priorities, **{h: Priority(p) for h, p in args.priorities.items()}}
        revised = ResearchPlan(objective=args.objective or plan.objective, hypothesis_priorities=priorities,
                               budget=plan.budget, notes=tuple(args.notes) or plan.notes)
        record(PlanRevised(version=state.plan_version + 1, plan=revised, reason=args.reason,
                           intervention_id=args.intervention_id))
        return f"Plan v{state.plan_version + 1} recorded. Running calculations are unaffected."

    def start_round(args: ReasonArgs) -> str:
        state = ctx.ledger.load(current())
        record(RoundStarted(round=state.round + 1, reason=args.reason))
        return f"Round {state.round + 1} started."

    def queue(role: str, args: TaskArgs) -> str:
        status = ctx.ledger.load(current()).status
        allowed = (InvestigationStatus.INVESTIGATING,) if role == "hypothesis_agent" else (
            InvestigationStatus.INVESTIGATING, InvestigationStatus.REVIEWING)
        if status not in allowed:
            raise ToolFailure(f"the investigation is {status.value}; move it to "
                              f"{' or '.join(s.value for s in allowed)} before queuing {role.replace('_', ' ')} work")
        task_id = new_id("task")
        record(AgentTaskQueued(task_id=task_id, role=role, hypothesis_id=args.hypothesis_id,
                               instruction=args.instruction))
        return (f"Queued {task_id} for the {role.replace('_', ' ')} on {args.hypothesis_id}. The dispatcher runs it; "
                "results appear in investigation_state and agent_reports.")

    def agent_reports(args: ReportsArgs) -> str:
        state = ctx.ledger.load(current())
        runs = [r for r in state.agent_runs if r.role != "chief_scientist"]
        if args.hypothesis_id is not None:
            runs = [r for r in runs if r.run.task_id and state.tasks[r.run.task_id].hypothesis_id == args.hypothesis_id]
        lines = [f"[{r.role} {r.actor_id}, {r.run.outcome}] {r.run.report or '(no report)'}" for r in runs[-args.last:]]
        return "\n\n".join(lines) or "No agent reports yet."

    def approvals(_: NoArgs) -> str:
        needs = pending_approvals(ctx.ledger.load(current()))
        return ("These wait for the researcher, who grants them by typing the command:\n" + "\n".join(needs)) \
            if needs else "Nothing waits for the researcher's approval."

    def describe_protocol(args: ProtocolArgs) -> str:
        state = ctx.ledger.load(current())
        profile = state.spec.constraints.compute_profile
        if profile is None:
            raise ToolFailure("the investigation has no compute profile")
        protocol = ctx.registry_for(profile).get(ProtocolRef(name=args.name, version=args.version))
        if protocol is None:
            available = ", ".join(f"{r.name} {r.version}" for r in ctx.registry_for(profile).refs())
            raise ToolFailure(f"no protocol {args.name} {args.version}; available: {available or 'none'}")
        return protocol.describe()

    return [
        AgentTool("list_compute_profiles", "List compute profiles and whether each is approved for use.", NoArgs,
                  list_compute_profiles),
        AgentTool("open_investigation", "Open a new investigation once the researcher has described the "
                  "discrepancy: what was simulated, what was measured, with which values and conditions. Record "
                  "only what the researcher said or showed; leave unknown fields empty.", OpenArgs, open_investigation),
        AgentTool("investigation_state", "Current state: status, plan, hypotheses, calculations with job and "
                  "evidence status, queued agent work, and what waits for the researcher. Check it before "
                  "answering questions about progress.", NoArgs, investigation_state),
        AgentTool("set_investigation_status", "Move the investigation one step at a time: CREATED -> "
                  "ALIGNING_OBSERVABLES -> PLANNING -> INVESTIGATING <-> REVIEWING -> COMPLETED or INCONCLUSIVE "
                  "(PAUSED or BLOCKED from any open state), with the reason.", InvestigationStatusArgs,
                  set_investigation_status),
        AgentTool("record_alignment", "Record whether the simulated and measured quantities are the same thing. A "
                  "numerical discrepancy exists only once they are ALIGNED.", AlignmentArgs, record_alignment),
        AgentTool("propose_hypothesis", "Propose one falsifiable explanation of the discrepancy.", HypothesisArgs,
                  propose_hypothesis),
        AgentTool("set_hypothesis_status", "Screen, activate, pause, resume, discard or supersede a hypothesis, "
                  "with the reason.", HypothesisStatusArgs, set_hypothesis_status),
        AgentTool("revise_plan", "Record a new plan version: priorities, objective or notes. Cite the researcher's "
                  "message when it asked for the change.", PlanArgs, revise_plan),
        AgentTool("start_round", "Start a new investigation round.", ReasonArgs, start_round),
        AgentTool("delegate", "Queue work for the hypothesis agent of an ACTIVE or PLANNING_TEST hypothesis.",
                  TaskArgs, lambda a: queue("hypothesis_agent", a)),
        AgentTool("request_judgement", "Queue an assessment of a hypothesis by the Evidence Judge.", TaskArgs,
                  lambda a: queue("evidence_judge", a)),
        AgentTool("agent_reports", "Read the latest reports of hypothesis agents and the Evidence Judge.",
                  ReportsArgs, agent_reports),
        AgentTool("pending_approvals", "List what waits for the researcher (submissions, inputs needing review) "
                  "and the exact command they type to grant each.", NoArgs, approvals),
        AgentTool("describe_protocol", "Explain what a protocol under this investigation's compute profile "
                  "accepts.", ProtocolArgs, describe_protocol),
    ]
