"""Tools of the Evidence Judge: read evidence closely, compare calculations, record assessments.

The ledger enforces the rules an assessment must meet (preregistration, validated
evidence for a final verdict, no support from invalid evidence); when the Judge breaks
one, the refusal comes back as a tool error it can act on.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from cedar.agents.tools import AgentTool, ToolFailure
from cedar.briefing import investigation_facts
from cedar.contracts.base import Actor, Contract, Identifier, NonEmptyStr
from cedar.contracts.events import AssessmentRecorded, EventDraft
from cedar.contracts.status import Assessment, Completeness
from cedar.contracts.task import compare_tasks
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import hypothesis_evidence, predicted_by
from cedar.store.artifacts import ArtifactStore
from cedar.summaries import review_lines, run_directory

SCRIPT_SHOWN_CHARS = 4000


@dataclass
class JudgeContext:
    ledger: ResearchLedger
    investigation_id: str
    actor: Actor
    artifacts: ArtifactStore | None = None
    """Where analysis scripts are stored, so the dossier can show them."""
    runs_root: str | None = None
    """Where run directories are, so the dossier can say where each calculation's inputs and outputs are."""


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
    lines += [f"  staged data inputs/data/{d.name}, from {d.origin}" for d in run.data]
    if artifacts is not None:
        script = artifacts.read(run.script).decode(errors="replace")
        shown = script if len(script) <= SCRIPT_SHOWN_CHARS else script[:SCRIPT_SHOWN_CHARS] + "\n…[script truncated]"
        lines += ["  script:", *(f"    {line}" for line in shown.splitlines())]
    return lines


def input_differences(calcs, varied) -> tuple[list[str], list[str]]:
    """Keywords the examined control.in files set differently that no task declares and the comparison does not
    vary: those the input reviewer classified as convergence aids or output only, and the rest, which are
    uncontrolled."""
    examined = [c for c in calcs if c.examination is not None and c.examination.settings and not c.registered]
    if len(examined) < 2:
        return [], []
    declared = {name for c in calcs for name in (*c.task.interventions, *c.task.controls, *c.task.numerical_settings)}
    names = sorted(set().union(*(c.examination.settings for c in examined)) - declared - set(varied))
    uncontrolled, aids = [], []
    for name in names:
        values = [c.examination.settings.get(name, "(absent)") for c in examined]
        if len(set(values)) == 1:
            continue
        roles = {c.input_review.settings.get(name) for c in examined if c.input_review is not None} - {None}
        line = f"    {name}: {values}" + (f" ({', '.join(sorted(roles))})" if roles else "")
        (aids if roles and roles <= {"convergence_aid", "output"} else uncontrolled).append(line)
    return uncontrolled, aids


def judge_tools(ctx: JudgeContext) -> list[AgentTool]:
    def load():
        return ctx.ledger.load(ctx.investigation_id)

    def dossier(args: HypothesisArg) -> str:
        state = load()
        hyp = state.hypotheses.get(args.hypothesis_id)
        if hyp is None:
            raise ToolFailure(f"unknown hypothesis {args.hypothesis_id}")
        lines = [*investigation_facts(state), "",
                 f"Hypothesis {hyp.hypothesis_id} ({hyp.status.value}; current assessment {hyp.assessment.value})",
                 f"Statement: {hyp.spec.statement}", f"Rationale: {hyp.spec.rationale}",
                 f"Falsifiers: {'; '.join(hyp.spec.falsifiers)}"]
        if hyp.spec.assumptions:
            lines.append(f"Assumptions: {'; '.join(hyp.spec.assumptions)}")
        for pid in hyp.prediction_ids:
            record = state.predictions[pid]
            p = record.prediction
            lines.append(f"Prediction {pid} (recorded at event {record.registered_at}): {p.statement} Expected: "
                         f"{p.expected_outcome}. Falsified if: {p.falsified_if}.")
        evidence = hypothesis_evidence(state, hyp.hypothesis_id)
        used = [c for record in evidence for c in (record.evidence.calculation_id, *record.evidence.calculation_ids)
                if c is not None]
        for cid in dict.fromkeys([*hyp.calculation_ids, *used]):
            calc = state.calculations[cid]
            task = calc.task
            lines.append(f"Calculation {cid}: {calc.state.value}; {task.observable.name} ({task.observable.definition}); "
                         f"estimand: {task.estimand}; system: {task.system_label}")
            if calc.registered:
                registration = calc.registration
                lines.append(f"  registered run from {registration.source}, "
                             + ("shared with every agent" if calc.shared else f"for {registration.hypothesis_id}")
                             + f": computed outside CEDAR (FHI-aims {registration.code_version}), its inputs not "
                             f"examined before it ran, its outputs there before any prediction; "
                             f"{registration.description}")
            lines.append(f"  interventions {task.interventions}; controls {task.controls}; numerical "
                         f"{task.numerical_settings}")
            exam = calc.examination
            if exam is not None and not calc.registered:
                if exam.settings:
                    lines.append("  control.in actually sets: " + "; ".join(f"{k} {v}".strip()
                                                                          for k, v in exam.settings.items()))
                directory = run_directory(calc, ctx.runs_root)
                if directory:
                    lines.append(f"  inputs and outputs in {directory}")
                if calc.input_review is not None:
                    lines += ["  " + line for line in review_lines(calc.input_review)]
            if calc.run_check is not None:
                failed = [c.name for c in calc.run_check.checks if not c.passed]
                lines.append("  run check: " + (f"FAILED {', '.join(failed)}" if failed else "healthy"))
        for record in evidence:
            ev = record.evidence
            value = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
            source = (f"analysis {ev.analysis_id} of {', '.join(ev.calculation_ids)}" if ev.analysis_id
                      else f"calculation {ev.calculation_id}" if ev.calculation_id else ev.kind.value)
            lines.append(f"Evidence {record.evidence_id} (recorded at event {record.recorded_at}) from {source}: "
                         f"{ev.quality.value}; {ev.observable.name} ({ev.observable.definition}) = {value}")
            lines.append(f"  summary by {ev.interpreted_by.id if ev.interpreted_by else 'the validator'}: {ev.summary}")
            tested = predicted_by(state, hyp.hypothesis_id, record)
            lines.append(f"  tests prediction {tested}: it named the calculation before it was submitted" if tested else
                         "  not a test of a prediction (it existed before, or was not predicted): it can challenge the "
                         "hypothesis or accompany a test, not support it")
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
        calcs = [state.calculations[c] for c in args.calculation_ids]
        report = compare_tasks([c.task for c in calcs], args.varied)
        uncontrolled, aids = input_differences(calcs, args.varied)
        lines = [f"Controlled comparison: {'yes' if report.is_controlled and not uncontrolled else 'NO'}"]
        lines += [f"  {d.variable} ({d.category}, {d.kind.value}): {list(d.values)}" for d in report.differences]
        if uncontrolled:
            lines.append("  Set differently in control.in, and declared by no task nor as varied (uncontrolled):")
            lines += uncontrolled
        if aids:
            lines.append("  Differ only in how a result is reached or what is printed, as the input reviewer "
                         "classified them:")
            lines += aids
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
