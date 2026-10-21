"""Tools of the Evidence Judge: read evidence closely, compare calculations, record assessments.

The ledger enforces the rules an assessment must meet (preregistration, validated
evidence for a final verdict, no support from invalid evidence); when the Judge breaks
one, the refusal comes back as a tool error it can act on.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from cedar.agents.tools import AgentTool, ToolFailure
from cedar.contracts.base import Actor, Contract, Identifier, NonEmptyStr
from cedar.contracts.events import AssessmentRecorded, EventDraft
from cedar.contracts.status import Assessment, Completeness
from cedar.contracts.task import compare_tasks
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import hypothesis_evidence
from cedar.store.artifacts import ArtifactStore

SCRIPT_SHOWN_CHARS = 4000


@dataclass
class JudgeContext:
    ledger: ResearchLedger
    investigation_id: str
    actor: Actor
    artifacts: ArtifactStore | None = None
    """Where analysis scripts are stored, so the dossier can show them."""


class HypothesisArg(Contract):
    hypothesis_id: Identifier


class NoArgs(Contract):
    pass


class CompareArgs(Contract):
    calculation_ids: list[Identifier] = Field(min_length=2)
    varied: list[str] = Field(description="Variables the comparison is meant to vary, e.g. ['xc'].")


class AssessmentArgs(Contract):
    hypothesis_id: Identifier
    assessment: Literal["SUPPORTED", "CHALLENGED", "INCONCLUSIVE", "NOT_TESTED", "NOT_COMPARABLE"]
    completeness: Literal["PARTIAL", "FINAL"] = Field(description="FINAL only when no planned evidence is missing.")
    evidence_ids: list[Identifier] = []
    rationale: NonEmptyStr = Field(description="Why this follows from the cited evidence, in a few sentences.")
    limitations: list[NonEmptyStr] = []


def _analysis_lines(run, artifacts: ArtifactStore | None) -> list[str]:
    lines = [f"  analysis purpose: {run.purpose}; ran with {run.environment}"]
    lines += [f"  staged from {i.calculation_id}: {', '.join(i.files)}" for i in run.inputs]
    if artifacts is not None:
        script = artifacts.read(run.script).decode(errors="replace")
        shown = script if len(script) <= SCRIPT_SHOWN_CHARS else script[:SCRIPT_SHOWN_CHARS] + "\n…[script truncated]"
        lines += ["  script:", *(f"    {line}" for line in shown.splitlines())]
    return lines


def judge_tools(ctx: JudgeContext) -> list[AgentTool]:
    def load():
        return ctx.ledger.load(ctx.investigation_id)

    def dossier(args: HypothesisArg) -> str:
        state = load()
        hyp = state.hypotheses.get(args.hypothesis_id)
        if hyp is None:
            raise ToolFailure(f"unknown hypothesis {args.hypothesis_id}")
        lines = [f"Hypothesis {hyp.hypothesis_id} ({hyp.status.value}; current assessment {hyp.assessment.value})",
                 f"Statement: {hyp.spec.statement}", f"Rationale: {hyp.spec.rationale}",
                 f"Falsifiers: {'; '.join(hyp.spec.falsifiers)}"]
        if hyp.spec.assumptions:
            lines.append(f"Assumptions: {'; '.join(hyp.spec.assumptions)}")
        for pid in hyp.prediction_ids:
            record = state.predictions[pid]
            p = record.prediction
            lines.append(f"Prediction {pid} (recorded at event {record.registered_at}): {p.statement} Expected: "
                         f"{p.expected_outcome}. Falsified if: {p.falsified_if}.")
        for cid in hyp.calculation_ids:
            calc = state.calculations[cid]
            task = calc.task
            lines.append(f"Calculation {cid}: {calc.state.value}; {task.observable.name} ({task.observable.definition}); "
                         f"estimand: {task.estimand}; system: {task.system_label}")
            lines.append(f"  interventions {task.interventions}; controls {task.controls}; numerical "
                         f"{task.numerical_settings}")
            if calc.run_check is not None:
                failed = [c.name for c in calc.run_check.checks if not c.passed]
                lines.append("  run check: " + (f"FAILED {', '.join(failed)}" if failed else "healthy"))
        for record in hypothesis_evidence(state, hyp.hypothesis_id):
            ev = record.evidence
            value = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
            source = (f"analysis {ev.analysis_id} of {', '.join(ev.calculation_ids)}" if ev.analysis_id
                      else f"calculation {ev.calculation_id}" if ev.calculation_id else ev.kind.value)
            lines.append(f"Evidence {record.evidence_id} (recorded at event {record.recorded_at}) from {source}: "
                         f"{ev.quality.value}; {ev.observable.name} ({ev.observable.definition}) = {value}")
            lines.append(f"  summary by {ev.interpreted_by.id if ev.interpreted_by else 'the validator'}: {ev.summary}")
            lines += [f"  check {c.name}: {'passed' if c.passed else 'FAILED'} - {c.detail or ''}"
                      for c in ev.quality_checks]
            lines += [f"  limitation: {text}" for text in ev.limitations]
            lines += [f"  not comparable when: {text}" for text in ev.non_comparability]
            if ev.analysis_id is not None:
                lines += _analysis_lines(state.analyses[ev.analysis_id].run, ctx.artifacts)
        for entry in hyp.assessments:
            lines.append(f"Earlier assessment: {entry.assessment.value} ({entry.completeness.value}), citing "
                         f"{', '.join(entry.evidence_ids) or 'nothing'}: {entry.rationale}")
        return "\n".join(lines)

    def evidence_index(_: NoArgs) -> str:
        state = load()
        lines = []
        for record in state.evidence.values():
            ev = record.evidence
            value = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
            source = (f"analysis {ev.analysis_id} of {', '.join(ev.calculation_ids)}" if ev.analysis_id
                      else f"calculation {ev.calculation_id or '-'}")
            lines.append(f"{record.evidence_id}: {ev.quality.value}, {ev.observable.name} = {value}; {source}; "
                         f"{ev.summary}")
        return "\n".join(lines) or "No evidence recorded yet."

    def compare(args: CompareArgs) -> str:
        state = load()
        missing = [c for c in args.calculation_ids if c not in state.calculations]
        if missing:
            raise ToolFailure(f"unknown calculations: {missing}")
        report = compare_tasks([state.calculations[c].task for c in args.calculation_ids], args.varied)
        lines = [f"Controlled comparison: {'yes' if report.is_controlled else 'NO'}"]
        lines += [f"  {d.variable} ({d.category}, {d.kind.value}): {list(d.values)}" for d in report.differences]
        if report.needs_numerical_justification:
            lines.append("  Numerical settings differ: the comparison needs evidence that the target quantity is "
                         "converged with respect to them.")
        if report.unused_declarations:
            lines.append(f"  Declared as varied but identical: {', '.join(report.unused_declarations)}")
        return "\n".join(lines)

    def record_assessment(args: AssessmentArgs) -> str:
        ctx.ledger.record(ctx.investigation_id, [EventDraft(actor=ctx.actor, payload=AssessmentRecorded(
            hypothesis_id=args.hypothesis_id, assessment=Assessment(args.assessment),
            completeness=Completeness(args.completeness), evidence_ids=tuple(args.evidence_ids),
            rationale=args.rationale, limitations=tuple(args.limitations)))])
        return f"Assessment recorded: {args.hypothesis_id} {args.assessment} ({args.completeness})."

    return [
        AgentTool("hypothesis_dossier", "Everything recorded about a hypothesis: statement, falsifiers, "
                  "predictions with when they were recorded, its calculations with settings and run checks, evidence "
                  "with every quality check and limitation (with the script behind each analysis result), and "
                  "earlier assessments.", HypothesisArg, dossier),
        AgentTool("evidence_index", "List all evidence in the investigation.", NoArgs, evidence_index),
        AgentTool("compare_calculations", "Check whether calculations form a controlled comparison: which "
                  "differences are declared, numerical, or uncontrolled.", CompareArgs, compare),
        AgentTool("record_assessment", "Record your assessment of a hypothesis, citing evidence ids. The record "
                  "refuses assessments its rules do not allow, and says why.", AssessmentArgs, record_assessment),
    ]
