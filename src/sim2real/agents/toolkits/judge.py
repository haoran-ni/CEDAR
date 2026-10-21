"""Tools of the Evidence Judge: read evidence closely, compare calculations, record assessments.

The ledger enforces the rules an assessment must meet (preregistration, validated
evidence for a final verdict, no support from invalid evidence); when the Judge breaks
one, the refusal comes back as a tool error it can act on.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from sim2real.agents.tools import AgentTool, ToolFailure
from sim2real.contracts.base import Actor, Contract, Identifier, NonEmptyStr
from sim2real.contracts.events import AssessmentRecorded, EventDraft
from sim2real.contracts.status import Assessment, Completeness
from sim2real.contracts.task import compare_tasks
from sim2real.state.ledger import ResearchLedger


@dataclass
class JudgeContext:
    ledger: ResearchLedger
    investigation_id: str
    actor: Actor


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
            if calc.evidence_id:
                record = state.evidence[calc.evidence_id]
                ev = record.evidence
                value = f"{ev.value.value} {ev.value.unit}" if ev.value else "no value"
                lines.append(f"  evidence {record.evidence_id} (recorded at event {record.recorded_at}): "
                             f"{ev.quality.value}, {value}; kind {ev.kind.value}")
                lines += [f"    check {c.name}: {'passed' if c.passed else 'FAILED'} - {c.detail or ''}"
                          for c in ev.quality_checks]
                lines += [f"    limitation: {text}" for text in ev.limitations]
                lines += [f"    not comparable when: {text}" for text in ev.non_comparability]
        for entry in hyp.assessments:
            lines.append(f"Earlier assessment: {entry.assessment.value} ({entry.completeness.value}), citing "
                         f"{', '.join(entry.evidence_ids) or 'nothing'}: {entry.rationale}")
        return "\n".join(lines)

    def evidence_index(_: NoArgs) -> str:
        state = load()
        lines = []
        for record in state.evidence.values():
            ev = record.evidence
            value = f"{ev.value.value} {ev.value.unit}" if ev.value else "no value"
            lines.append(f"{record.evidence_id}: {ev.quality.value}, {ev.observable.name} = {value}; calculation "
                         f"{ev.calculation_id or '-'}; {ev.summary}")
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
                  "predictions with when they were recorded, its calculations with settings, evidence with every "
                  "quality check and limitation, and earlier assessments.", HypothesisArg, dossier),
        AgentTool("evidence_index", "List all evidence in the investigation.", NoArgs, evidence_index),
        AgentTool("compare_calculations", "Check whether calculations form a controlled comparison: which "
                  "differences are declared, numerical, or uncontrolled.", CompareArgs, compare),
        AgentTool("record_assessment", "Record your assessment of a hypothesis, citing evidence ids. The record "
                  "refuses assessments its rules do not allow, and says why.", AssessmentArgs, record_assessment),
    ]
