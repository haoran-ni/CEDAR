"""Tools a hypothesis agent uses to act on the research record.

Each tool goes through the ledger or an engine service under the agent's own identity,
so every research rule applies: the agent can write inputs and request calculations, but
the examiner judges them, submission needs the researcher's authorization, a reading
counts only after the validator has checked its citations, and an analysis script runs
only through the guarded runner, which records it.

An agent works on its own hypothesis: it records predictions, calculations, evidence and
status only for it. It sees the other agents' work (``other_agents``) and may read their run
directories and workspaces, read and analyse any calculation of the investigation, and use
what it finds as evidence for its own hypothesis. It may read any file the researcher can,
except credentials; it writes only in its workspace.
"""

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from cedar.agents.tools import AgentTool, ToolFailure, ToolOutcome
from cedar.agents.toolkits.files import FileAccess, workspace_tools
from cedar.briefing import investigation_facts
from cedar.contracts.agents import AgentReport, ReportFinding, ReportOutcome, ReportRequest
from cedar.contracts.base import (Actor, Contract, FileName, Identifier, NonEmptyStr, Observable, Quantity,
                                  Uncertainty, UncertaintyKind)
from cedar.contracts.events import EventDraft, HypothesisStatusChanged, PredictionRegistered
from cedar.contracts.evidence import FileCitation, OutputReading
from cedar.contracts.hypothesis import Prediction
from cedar.contracts.status import CalculationState, HypothesisStatus
from cedar.contracts.task import ProtocolRef, ResourceBudget, ScientificTaskSpec
from cedar.engine.analysis import RESULTS_FILE, RESULTS_FORMAT, AnalysisRunner
from cedar.engine.external import ExternalEvidenceRecorder, ExternalFact
from cedar.engine.inputs import ProtocolEngine
from cedar.engine.registration import RunRegistrar
from cedar.engine.results import ResultValidator
from cedar.engine.workbench import AgentBench
from cedar.hpc.node import local_dft
from cedar.paths import workspace_path
from cedar.protocols.base import ProtocolRegistry
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import CalculationRecord, InvestigationState, evidence_owner
from cedar.store.artifacts import ArtifactStore
from cedar.summaries import calculation_summary
from cedar.web import web_summary


@dataclass
class ResearchContext:
    ledger: ResearchLedger
    investigation_id: str
    hypothesis_id: str
    actor: Actor
    registry: ProtocolRegistry
    artifacts: ArtifactStore
    engine: ProtocolEngine | None
    """None while the investigation has no approved compute profile: no calculation can then be requested."""
    validator: ResultValidator | None
    """Without a compute profile, one that handles registered runs only."""
    files: FileAccess
    analysis: AnalysisRunner | None
    workspace: Path | None = None
    """The agent's own directory, the only place it writes."""
    registrar: RunRegistrar | None = None
    bench: AgentBench | None = None
    """Runs the agent's commands in its workspace, in the workbench's sandbox."""
    runs_root: Path | None = None
    """Where CEDAR's calculations run, under the investigation's compute profile."""
    workspaces: Path | None = None
    """Where the agents' workspaces are, one directory per hypothesis."""
    report: AgentReport | None = None
    """The report the agent ended its task with, once it has."""
    external: ExternalEvidenceRecorder | None = None
    """Records measured values and facts about the experiment as evidence, checking their quotes."""


NO_COMPUTE = ("this investigation has no approved compute profile, so no calculation can be requested. You can "
              "still register existing runs and read and analyse them, plan, register predictions and report; tell "
              "the Chief Scientist.")
NO_RESULTS = "no run can be read or analysed here; tell the Chief Scientist."
MAX_INPUT_BYTES = 50 * 1024 ** 2
"""Largest structure or input file a request may name by path."""


def _require_compute(ctx: ResearchContext) -> None:
    if ctx.engine is None or ctx.validator is None or ctx.analysis is None:
        raise ToolFailure(NO_COMPUTE)


def _require_results(ctx: ResearchContext) -> None:
    if ctx.validator is None or ctx.analysis is None:
        raise ToolFailure(NO_RESULTS)


class NoArgs(Contract):
    pass


class DescribeArgs(Contract):
    name: str = Field(description="Protocol name, e.g. fhiaims.run")
    version: str


class StatusArgs(Contract):
    status: Literal["ACTIVE", "PLANNING_TEST", "ANALYZING", "ASSESSED", "NEEDS_DATA"]
    reason: NonEmptyStr = Field(description="One or two sentences of scientific justification.")


class PredictionArgs(Contract):
    prediction_id: Identifier
    statement: NonEmptyStr
    observable_name: NonEmptyStr
    observable_definition: NonEmptyStr
    observable_unit: str | None = None
    expected_outcome: NonEmptyStr
    falsified_if: NonEmptyStr
    calculation_ids: list[Identifier] = Field(default=[], description="Planned calculations this predicts.")


class CalculationArgs(Contract):
    protocol_name: str
    protocol_version: str
    observable_name: NonEmptyStr = Field(description="What you will take from this run, e.g. 'total energy' or "
                                                     "'CO 2pi* Kohn-Sham eigenvalue'.")
    observable_definition: NonEmptyStr = Field(description="Precisely which quantity, and how it is obtained from "
                                                           "the run.")
    observable_unit: str | None = None
    estimand: NonEmptyStr = Field(description="Exactly what the calculation estimates.")
    system_label: NonEmptyStr
    structure_path: str | None = Field(default=None, description="Absolute path of a readable structure file.")
    structure_text: str | None = Field(default=None, description="The structure file's full text, if you wrote it.")
    interventions: dict[str, JsonValue] = Field(default={}, description="Variables this test deliberately changes.")
    controls: dict[str, JsonValue] = Field(default={}, description="Physical settings held fixed.")
    numerical_settings: dict[str, JsonValue] = Field(default={}, description="Accuracy settings.")
    input_files: dict[str, str] = Field(default={}, description="Input files you wrote, by name, with full text. "
                                                                "Files you leave out are generated by the protocol.")
    input_paths: dict[str, str] = Field(default={}, description="Input files by name, given as the absolute path of "
                                                                "a file you can read, e.g. one in your workspace.")
    max_core_hours: float | None = Field(default=None, gt=0)
    max_walltime_hours: float | None = Field(default=None, gt=0)
    cores: int | None = Field(default=None, gt=0, description="Cores to run on, if not the default.")
    execution: Literal["slurm", "local"] = Field(default="slurm", description="'slurm' submits a job; 'local' runs on "
                                                 "the interactive node CEDAR runs on, inside the researcher's "
                                                 "allocation, which investigation_overview says is possible or not. "
                                                 "Either needs the researcher's approval to start.")
    rationale: NonEmptyStr = Field(description="Why this calculation discriminates between explanations.")

    @model_validator(mode="after")
    def _one_structure(self) -> "CalculationArgs":
        if (self.structure_path is None) == (self.structure_text is None):
            raise ValueError("give exactly one of structure_path and structure_text")
        both = set(self.input_files) & set(self.input_paths)
        if both:
            raise ValueError(f"give {', '.join(sorted(both))} either as text or as a path, not both")
        return self


class FindingArg(Contract):
    statement: NonEmptyStr
    evidence_ids: list[Identifier] = Field(default=[], description="Recorded evidence it rests on; none makes it "
                                                                   "your inference.")


class RequestArg(Contract):
    to: Literal["researcher", "chief_scientist"]
    text: NonEmptyStr


class ReportArgs(Contract):
    outcome: Literal["concluded", "waiting_for_calculations", "needs_approval", "needs_researcher_data",
                     "needs_other_agent", "could_not_proceed"]
    summary: NonEmptyStr = Field(description="What you did, what you found and what it means for your hypothesis, "
                                             "in a few sentences. State hypotheses as hypotheses.")
    findings: list[FindingArg] = []
    open_questions: list[NonEmptyStr] = []
    requests: list[RequestArg] = Field(default=[], description="What you need from the researcher (data, a "
                                                               "decision, an approval) or from the Chief Scientist.")
    next_steps: list[NonEmptyStr] = []
    wait_for: list[Identifier] = Field(default=[], description="Calculations to wait for, with outcome "
                                                               "waiting_for_calculations: you are woken when they "
                                                               "have all ended.")


class CalculationArg(Contract):
    calculation_id: Identifier


class CitationArg(Contract):
    file: FileName = Field(description="Output file name within the run directory, e.g. aims.out")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: NonEmptyStr = Field(description="The cited lines' text, copied exactly.")


class ReadingArgs(Contract):
    calculation_id: Identifier
    observable_name: NonEmptyStr = Field(description="What the value is, e.g. 'total energy'.")
    observable_definition: NonEmptyStr = Field(description="Precisely which quantity, from which part of the "
                                                           "output.")
    value: float
    unit: NonEmptyStr
    citations: list[CitationArg] = Field(min_length=1)
    summary: NonEmptyStr
    limitations: list[NonEmptyStr] = []


class AnalysisInputArg(Contract):
    calculation_id: Identifier
    files: list[str] = Field(default=[], description="Files of the run directory to stage, by path relative to it. "
                                                     "Empty: the run's stored outputs and its input files.")


class AnalysisArgs(Contract):
    purpose: NonEmptyStr = Field(description="What the script computes and why, in a sentence.")
    script: NonEmptyStr = Field(description="Complete Python script. It reads inputs/<calculation_id>/<file>, "
                                            "prints what it finds, and writes the results and the checks it made to "
                                            f"{RESULTS_FILE} in the working directory: {RESULTS_FORMAT}. Top-level "
                                            "checks apply to every result; a result's own checks to it alone.")
    inputs: list[AnalysisInputArg] = Field(default=[], description="Calculations whose files the script reads.")
    materials: list[Identifier] = Field(default=[], description="Attached materials to stage, e.g. measured data; "
                                                                "each is inputs/data/<its file name>.")
    files: list[str] = Field(default=[], description="Other files to stage, by absolute path, e.g. a reference "
                                                     "structure in your workspace; each is inputs/data/<file name>.")
    save_products: list[str] = Field(default=[], description="Files the script writes that you want afterwards in "
                                                             "your workspace, by path relative to its working "
                                                             "directory; they are copied to "
                                                             "analyses/<analysis_id>/ there.")


class RegisterArgs(Contract):
    directory: str = Field(description="Absolute path of a finished FHI-aims run's directory, holding its "
                                       "control.in, geometry.in and output.")
    description: NonEmptyStr = Field(description="What the run is and why it matters for your hypothesis.")
    output_file: FileName = Field(default="aims.out", description="The output's file name in that directory.")
    system_label: str | None = Field(default=None, description="A short label; the directory's name by default.")


class CommandArgs(Contract):
    command: NonEmptyStr = Field(description="A shell command, run with bash in your workspace, e.g. "
                                             "'python build_slab.py', 'grep -n SCF /path/to/aims.out | tail' or "
                                             "'pytest -q tests'.")
    purpose: NonEmptyStr = Field(description="What you are checking or making, in a sentence.")
    directory: str = Field(default=".", description="Where to run it, relative to your workspace.")
    minutes: int | None = Field(default=None, ge=1, description="Time limit; the default is short. The command "
                                                                "is stopped when it runs out.")
    cores: int | None = Field(default=None, ge=1, description="Cores, for a command that uses several; one on a "
                                                              "login node.")


class ExternalArgs(Contract):
    kind: Literal["measurement", "constraint", "inference"] = Field(description=(
        "measurement: a measured value or result; constraint: a fact that bounds the comparison, such as the "
        "experiment's coverage, temperature or technique; inference: something derived from such facts under "
        "stated assumptions."))
    observable_name: NonEmptyStr
    observable_definition: NonEmptyStr = Field(description="Precisely what was measured or stated, and how.")
    value: float | None = Field(default=None, description="The value, if the fact has one.")
    unit: str | None = None
    uncertainty: float | None = Field(default=None, ge=0, description="As the source gives it.")
    uncertainty_basis: str | None = None
    source_kind: Literal["researcher_message", "investigation", "material", "file", "url"]
    source: NonEmptyStr = Field(description="researcher_message: its record id, e.g. msg-3; investigation: the "
                                            "label of a measurement recorded when the investigation was opened; "
                                            "material: its id; file: an absolute path; url: the page's address.")
    quote: NonEmptyStr = Field(description="The source's words, copied exactly, containing the value if there is "
                                           "one. For a figure, say what you read off it and where.")
    locator: str | None = Field(default=None, description="Page, figure or table within the source.")
    title: str | None = Field(default=None, description="For a URL, the page's title.")
    conditions: dict[str, str] = Field(default={}, description="Conditions the source states, e.g. temperature.")
    summary: NonEmptyStr
    assumptions: list[NonEmptyStr] = Field(default=[], description="Required for an inference.")
    limitations: list[NonEmptyStr] = []


class AgentsArgs(Contract):
    hypothesis_id: Identifier | None = Field(default=None, description="One hypothesis to show in full: its "
                                                                       "predictions, every calculation and its whole "
                                                                       "latest report. Omit for all of them, briefly.")


class ResultArgs(Contract):
    analysis_id: Identifier
    result_name: Identifier = Field(description=f"A result name from the analysis's {RESULTS_FILE}.")
    observable_name: NonEmptyStr
    observable_definition: NonEmptyStr
    summary: NonEmptyStr = Field(description="What the value is and how the script obtained it.")
    limitations: list[NonEmptyStr] = []


def _calculation_summary(ctx: ResearchContext, calc: CalculationRecord, state: InvestigationState | None = None) -> str:
    return calculation_summary(state or ctx.ledger.load(ctx.investigation_id), calc)


def _calculation(ctx: ResearchContext, calculation_id: str) -> tuple[InvestigationState, CalculationRecord]:
    """Any calculation of the investigation: the hypothesis's own, another hypothesis's, or a registered run."""
    state = ctx.ledger.load(ctx.investigation_id)
    calc = state.calculations.get(calculation_id)
    if calc is None:
        raise ToolFailure(f"no calculation {calculation_id} in this investigation")
    return state, calc


def _files_in(directory: Path, shown: int, visited: int = 5000) -> tuple[list[Path], int | str]:
    """Up to ``shown`` files under ``directory``, and how many more there are: a count, or a lower bound such as
    "120+" once ``visited`` entries have been looked at."""
    files, seen = [], 0
    for here, subdirectories, names in os.walk(directory):
        subdirectories.sort()
        seen += len(subdirectories) + len(names)
        files += [Path(here) / name for name in sorted(names)]
        if seen > visited:
            return files[:shown], f"{max(len(files) - shown, 0)}+"
    return files[:shown], len(files) - len(files[:shown])


def run_directory(ctx: ResearchContext, calc: CalculationRecord) -> Path | None:
    """Where a calculation's files are: its run directory, or the directory a registered run came from."""
    if calc.registered:
        return Path(calc.registration.source)
    if ctx.runs_root is None or calc.workdir is None:
        return None
    return Path(ctx.runs_root) / calc.workdir


def team_text(ctx: ResearchContext, state: InvestigationState, hypothesis_id: str | None = None) -> str:
    """The other agents' work, for an agent: their hypotheses, calculations with where their files are, evidence and
    latest reports; one hypothesis in full when ``hypothesis_id`` is given."""
    me = state.active_agent(ctx.hypothesis_id)
    lines = [f"Agents of {state.investigation_id}" + (f" (you are {me.tag}, {me.agent_id})" if me else "") + ":"]
    hypotheses = [hypothesis_id] if hypothesis_id is not None else list(state.hypotheses)
    if hypothesis_id is not None and hypothesis_id not in state.hypotheses:
        raise ToolFailure(f"no hypothesis {hypothesis_id} in this investigation")
    for h in hypotheses:
        hyp = state.hypotheses[h]
        agent = state.active_agent(h) or next((a for a in reversed(list(state.agents.values()))
                                               if a.hypothesis_id == h), None)
        who = f"{agent.tag} {agent.agent_id} ({agent.spec.role}{'' if agent.active else ', retired'})" if agent \
            else "no agent"
        mine = " (yours)" if h == ctx.hypothesis_id else ""
        lines += ["", f"{who} on {h}{mine} [{hyp.status.value}; assessment {hyp.assessment.value}]: "
                      f"{hyp.spec.statement}"]
        if ctx.workspaces is not None:
            lines.append(f"  workspace: {Path(ctx.workspaces) / h}")
        if hypothesis_id is not None:
            lines += [f"  prediction {pid}: {state.predictions[pid].prediction.statement} Expected: "
                      f"{state.predictions[pid].prediction.expected_outcome}" for pid in hyp.prediction_ids]
        for cid in hyp.calculation_ids:
            calc = state.calculations[cid]
            where = run_directory(ctx, calc)
            lines.append(f"  {cid}: {calc.task.system_label}; {calc.state.value}"
                         + (f"; files in {where}" if where is not None else ""))
            if hypothesis_id is not None:
                lines += ["    " + line for line in calculation_summary(state, calc).splitlines()[1:]]
        evidence = [r for r in state.evidence.values() if evidence_owner(state, r.evidence) == h]
        for record in evidence[-10:] if hypothesis_id is None else evidence:
            ev = record.evidence
            value = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
            lines.append(f"  evidence {record.evidence_id}: {ev.observable.name} = {value}; {ev.quality.value}")
        runs = [r for r in state.agent_runs if agent is not None and r.actor_id == agent.agent_id and r.run.report]
        if runs:
            report = runs[-1].run.report.strip()
            shown = report if hypothesis_id is not None or len(report) <= 600 else report[:600] + " …"
            lines.append(f"  latest report ({runs[-1].run.outcome}): {shown}")
    return "\n".join(lines)


def _data_files(ctx: ResearchContext, materials: Sequence[str], paths: Sequence[str]) -> dict:
    """Materials and files to stage for an analysis, stored by hash, by the plain file name they get."""
    state = ctx.ledger.load(ctx.investigation_id)
    found = []
    for material_id in materials:
        material = state.materials.get(material_id)
        if material is None:
            raise ToolFailure(f"no material {material_id} in this investigation")
        found.append((Path(material.name).name, material.artifact, f"material {material_id}"))
    for path in paths:
        resolved = ctx.files.regular_file(path)
        if resolved.stat().st_size > MAX_INPUT_BYTES:
            raise ToolFailure(f"{path} is larger than {MAX_INPUT_BYTES} bytes")
        found.append((resolved.name, ctx.artifacts.put_file(resolved), str(resolved)))
    data = {}
    for name, ref, origin in found:
        staged = name if name not in data else f"{len(data) + 1}-{name}"
        data[staged] = (ref, origin)
    return data


def _read_input(ctx: ResearchContext, path: str) -> bytes:
    resolved = ctx.files.regular_file(path)
    if resolved.stat().st_size > MAX_INPUT_BYTES:
        raise ToolFailure(f"{path} is larger than {MAX_INPUT_BYTES} bytes")
    return resolved.read_bytes()


def _next_step(calc: CalculationRecord) -> str:
    if calc.state is CalculationState.REJECTED:
        return "The inputs were rejected. Fix the errors above and make a new request."
    if calc.state is CalculationState.SPECIFIED and calc.examination is not None:
        return ("The examiner could not verify part of the inputs. The researcher must approve them before they "
                "can run; tell the Chief Scientist why the unusual content is needed.")
    if calc.state is CalculationState.VALIDATED_INPUT:
        return ("The inputs are approved. Submission needs the researcher's authorization. Register predictions "
                "for it now if you have not, then end with submit_report, waiting for it.")
    return "See the state above."


def research_tools(ctx: ResearchContext) -> list[AgentTool]:
    def record(*drafts: EventDraft):
        return ctx.ledger.record(ctx.investigation_id, list(drafts))

    def investigation_overview(_: NoArgs) -> str:
        state = ctx.ledger.load(ctx.investigation_id)
        hyp = state.hypotheses[ctx.hypothesis_id]
        spec = state.spec
        lines = [
            *investigation_facts(state), "",
            f"Compute profile: {spec.constraints.compute_profile or 'none'} (set by the investigation; it is not "
            "a calculation variable). Calculations can be requested only while the investigation is INVESTIGATING.",
            f"Your hypothesis {hyp.hypothesis_id} ({hyp.status.value}; assessment {hyp.assessment.value}; "
            f"plan priority {state.plan.priority(hyp.hypothesis_id).value}): {hyp.spec.statement}",
            f"Falsifiers: {'; '.join(hyp.spec.falsifiers)}",
            f"Web search and fetch: {web_summary(state.retrieval)} (set by the researcher)",
        ]
        if ctx.workspace is not None:
            lines.append(f"Your workspace (write_file, delete_file; read it with the file tools): {ctx.workspace}")
        if ctx.bench is not None:
            lines.append("Local DFT (request_calculation with execution 'local'): "
                         + local_dft(ctx.bench.workbench.host()))
        lines.append("You may read any file the researcher can, except credentials; other_agents shows the other "
                     "agents' work.")
        for pid in hyp.prediction_ids:
            p = state.predictions[pid].prediction
            lines.append(f"Prediction {pid}: {p.statement} Expected: {p.expected_outcome}. Falsified if: "
                         f"{p.falsified_if}.")
        for cid in hyp.calculation_ids:
            lines.append(_calculation_summary(ctx, state.calculations[cid], state))
        shared = [c for c in state.calculations.values() if c.shared]
        if shared:
            lines.append("Registered runs shared with every agent (read and analyse them like your own):")
            lines += [_calculation_summary(ctx, calc, state) for calc in shared]
        if state.budget_usd is not None:
            lines.append(f"Budget left for this investigation's model use: ${state.budget_left:.2f}")
        return "\n".join(lines)

    def describe_protocol(args: DescribeArgs) -> str:
        protocol = ctx.registry.get(ProtocolRef(name=args.name, version=args.version))
        if protocol is None:
            available = ", ".join(f"{r.name} {r.version}" for r in ctx.registry.refs())
            raise ToolFailure(f"no protocol {args.name} {args.version}; available: {available}")
        return protocol.describe()

    def set_hypothesis_status(args: StatusArgs) -> str:
        record(EventDraft(actor=ctx.actor, payload=HypothesisStatusChanged(
            hypothesis_id=ctx.hypothesis_id, status=HypothesisStatus(args.status), reason=args.reason)))
        return f"Hypothesis {ctx.hypothesis_id} is now {args.status}."

    def register_prediction(args: PredictionArgs) -> str:
        observable = Observable(name=args.observable_name, definition=args.observable_definition,
                                unit=args.observable_unit)
        record(EventDraft(actor=ctx.actor, payload=PredictionRegistered(
            prediction_id=args.prediction_id, hypothesis_id=ctx.hypothesis_id, prediction=Prediction(
                statement=args.statement, observable=observable, expected_outcome=args.expected_outcome,
                falsified_if=args.falsified_if, calculation_ids=tuple(args.calculation_ids)))))
        return f"Prediction {args.prediction_id} registered; it can no longer be changed."

    def request_calculation(args: CalculationArgs) -> str:
        _require_compute(ctx)
        protocol = ctx.registry.get(ProtocolRef(name=args.protocol_name, version=args.protocol_version))
        if protocol is None:
            raise ToolFailure(f"no protocol {args.protocol_name} {args.protocol_version}")
        observable = Observable(name=args.observable_name, definition=args.observable_definition,
                                unit=args.observable_unit)
        if args.structure_path is not None:
            structure_bytes = _read_input(ctx, args.structure_path)
        else:
            structure_bytes = args.structure_text.encode()
        structure = ctx.artifacts.put_bytes(structure_bytes, label=protocol.structure_file)
        authored = {name: ctx.artifacts.put_bytes(text.encode(), label=name) for name, text in args.input_files.items()}
        authored |= {name: ctx.artifacts.put_bytes(_read_input(ctx, path), label=name)
                     for name, path in args.input_paths.items()}
        task = ScientificTaskSpec(
            protocol=protocol.ref, observable=observable, estimand=args.estimand, system_label=args.system_label,
            structure=structure, interventions=args.interventions, controls=args.controls,
            numerical_settings=args.numerical_settings, authored_files=authored, execution=args.execution,
            budget=ResourceBudget(max_core_hours=args.max_core_hours, max_walltime_hours=args.max_walltime_hours,
                                  cores=args.cores),
        )
        outcome = ctx.ledger.request_calculation(ctx.investigation_id, ctx.actor, ctx.hypothesis_id, task,
                                                 args.rationale)
        calc = ctx.ledger.load(ctx.investigation_id).calculations[outcome.calculation_id]
        if calc.state is CalculationState.SPECIFIED and calc.examination is None:
            ctx.engine.examine(ctx.investigation_id, calc.calculation_id)
            calc = ctx.ledger.load(ctx.investigation_id).calculations[calc.calculation_id]
        origin = "an identical existing calculation, now linked to you" if outcome.reused else "a new calculation"
        return f"{_calculation_summary(ctx, calc)}\n({origin})\n{_next_step(calc)}"

    def submit_report(args: ReportArgs) -> ToolOutcome:
        report = AgentReport(
            outcome=ReportOutcome(args.outcome), summary=args.summary,
            findings=tuple(ReportFinding(statement=f.statement, evidence_ids=tuple(f.evidence_ids))
                           for f in args.findings),
            open_questions=tuple(args.open_questions),
            requests=tuple(ReportRequest(to=r.to, text=r.text) for r in args.requests),
            next_steps=tuple(args.next_steps), waiting_for=tuple(args.wait_for))
        state = ctx.ledger.load(ctx.investigation_id)
        unknown = sorted({e for f in report.findings for e in f.evidence_ids} - set(state.evidence))
        if unknown:
            raise ToolFailure(f"no evidence {', '.join(unknown)} in the record; cite recorded evidence ids only")
        if report.waiting_for:
            record(EventDraft(actor=ctx.actor, payload=HypothesisStatusChanged(
                hypothesis_id=ctx.hypothesis_id, status=HypothesisStatus.WAITING_FOR_JOB,
                reason=f"waiting for {', '.join(report.waiting_for)}", awaiting_calculation_ids=report.waiting_for)))
        ctx.report = report
        if report.waiting_for:
            return ToolOutcome("Report recorded. Your task ends here; you are woken when these calculations have "
                               "ended.", end_run=True, ends_as="waiting")
        return ToolOutcome("Report recorded. Your task ends here.", end_run=True, ends_as="reported")

    def run_outputs(args: CalculationArg) -> str:
        state, calc = _calculation(ctx, args.calculation_id)
        _require_results(ctx)
        if calc.workdir is None and not calc.registered:
            return f"{args.calculation_id} has no run directory yet ({calc.state.value})."
        lines = [_calculation_summary(ctx, calc, state)]
        if calc.run_check is None:
            lines.append("The run has not been checked yet; readings and analyses wait for the check.")
        else:
            stored = ", ".join(calc.run_check.outputs) or "none"
            lines.append(f"Stored outputs (cite these in readings; staged for analyses by default): {stored}")
        if calc.registered:
            lines.append(f"Stored inputs: {', '.join(calc.input_files)}")
        directory = ctx.validator.run_directory(ctx.investigation_id, args.calculation_id)
        if directory.is_dir():
            files, more = _files_in(directory, 200)
            lines.append(f"Files in {directory} (stage others for an analysis by their relative path):")
            lines += [f"  {p.relative_to(directory)}  ({p.stat().st_size} bytes)" for p in files]
            if more:
                lines.append(f"  … {more} more")
        return "\n".join(lines)

    def submit_reading(args: ReadingArgs) -> ToolOutcome:
        _calculation(ctx, args.calculation_id)
        _require_results(ctx)
        reading = OutputReading(
            reader=ctx.actor, value=Quantity(value=args.value, unit=args.unit), summary=args.summary,
            limitations=tuple(args.limitations),
            citations=tuple(FileCitation(file=c.file, line_start=c.line_start, line_end=c.line_end, quote=c.quote)
                            for c in args.citations))
        observable = Observable(name=args.observable_name, definition=args.observable_definition, unit=args.unit)
        outcome = ctx.validator.record_reading(ctx.investigation_id, args.calculation_id, reading, observable)
        if outcome.recorded is None:
            return ToolOutcome("Reading not accepted:\n" + "\n".join(f"- {p}" for p in outcome.problems),
                               is_error=True)
        (event,) = outcome.recorded.events
        ev = event.payload.evidence
        return (f"Reading recorded as evidence {event.payload.evidence_id}: {ev.quality.value}, "
                f"{ev.observable.name} = {ev.value.value} {ev.value.unit}.")

    def run_analysis(args: AnalysisArgs) -> ToolOutcome:
        for staged in args.inputs:
            _calculation(ctx, staged.calculation_id)
        _require_results(ctx)
        if not (args.inputs or args.materials or args.files):
            raise ToolFailure("name the calculations, materials or files the script reads")
        report = ctx.analysis.run(ctx.investigation_id, ctx.hypothesis_id, purpose=args.purpose, script=args.script,
                                  inputs={i.calculation_id: i.files for i in args.inputs},
                                  data=_data_files(ctx, args.materials, args.files))
        if report.analysis_id is None:
            return ToolOutcome("The script did not run:\n" + "\n".join(f"- {p}" for p in report.problems),
                               is_error=True)
        run = report.run
        lines = [f"Analysis {report.analysis_id}: {run.outcome.value} (exit code {run.exit_code}, "
                 f"{run.duration_seconds:g} s)"]
        def check_line(c) -> str:
            return f"    {'passed' if c.passed else 'FAILED'} {c.name}" + (f": {c.detail}" if c.detail else "")

        if run.checks:
            lines.append("Checks on every result:")
            lines += [check_line(c) for c in run.checks]
        if run.results:
            lines.append("Results (record one with record_result):")
            for r in run.results:
                lines.append(f"  {r.name} = {r.value.value!r} {r.value.unit}" + (f"  ({r.description})"
                                                                                if r.description else ""))
                lines += [check_line(c) for c in r.checks]
        lines += [f"Note: {p}" for p in report.problems]
        lines += _save_products(report.analysis_id, run, args.save_products)
        lines.append("Output:\n" + (report.shown_output or "(nothing printed)"))
        return ToolOutcome("\n".join(lines), is_error=run.outcome.value != "completed")

    def _save_products(analysis_id: str, run, wanted: list[str]) -> list[str]:
        if not wanted:
            return []
        if ctx.workspace is None:
            return ["Note: you have no workspace, so no product was saved."]
        lines = []
        for name in wanted:
            ref = run.products.get(name)
            if ref is None:
                lines.append(f"Note: the script wrote no {name}, so it was not saved.")
                continue
            target = workspace_path(ctx.workspace, f"analyses/{analysis_id}/{name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ctx.artifacts.resolve(ref), target)
            lines.append(f"Saved {name} to {target}")
        return lines

    def run_command(args: CommandArgs) -> ToolOutcome:
        if ctx.bench is None:
            raise ToolFailure("there is no workbench here; tell the Chief Scientist")
        report = ctx.bench.run(args.command, args.purpose, directory=args.directory, minutes=args.minutes,
                               cores=args.cores)
        if report.run_id is None:
            return ToolOutcome("The command did not run:\n" + "\n".join(f"- {p}" for p in report.problems),
                               is_error=True)
        run = report.run
        lines = [f"Workbench run {report.run_id}: {run.outcome.value}"
                 + (f" (exit code {run.exit_code})" if run.exit_code is not None else "")
                 + f", {run.duration_seconds:g} s on {run.cores} core(s) of {run.host}"]
        lines += [f"Note: {p}" for p in report.problems]
        if run.changed:
            names = sorted(run.changed)
            lines.append("Files created or changed: " + ", ".join(names[:50])
                         + (f" … and {len(names) - 50} more" if len(names) > 50 else ""))
        if run.deleted:
            lines.append("Files deleted: " + ", ".join(run.deleted[:50]))
        lines.append(f"Output (all of it is in {report.output_path}):\n" + (report.shown_output or "(nothing printed)"))
        return ToolOutcome("\n".join(lines), is_error=run.outcome.value != "completed")

    def workbench_runs(_: NoArgs) -> str:
        state = ctx.ledger.load(ctx.investigation_id)
        runs = [r for r in state.workbench_runs.values() if r.hypothesis_id == ctx.hypothesis_id]
        lines = [ctx.bench.where()]
        if not state.workbench_enabled:
            lines.append("The researcher has turned the workbench off for this investigation.")
        lines += [f"{r.run_id}: {r.run.outcome.value}, {r.run.duration_seconds:g} s: {r.run.command}"
                  for r in runs[-30:]] or ["No commands run yet."]
        return "\n".join(lines)

    def register_existing_run(args: RegisterArgs) -> ToolOutcome:
        if ctx.registrar is None:
            raise ToolFailure(NO_RESULTS)
        outcome = ctx.registrar.register(ctx.investigation_id, args.directory, requested_by=ctx.actor,
                                         description=args.description, hypothesis_id=ctx.hypothesis_id,
                                         output_file=args.output_file, system_label=args.system_label)
        if outcome.calculation_id is None:
            return ToolOutcome("Not registered:\n" + "\n".join(f"- {p}" for p in outcome.problems), is_error=True)
        state = ctx.ledger.load(ctx.investigation_id)
        calc = state.calculations[outcome.calculation_id]
        notes = [f"  [{n.severity.value}] {n.code}: {n.message}" for n in calc.registration.notes]
        return "\n".join([
            _calculation_summary(ctx, calc, state),
            "(already registered for you)" if outcome.reused else "(registered)",
            *notes,
            "Read it with run_outputs and the file tools; record a printed value with submit_reading, or compute "
            "one with run_analysis and record_result. Evidence from it is marked as computed outside CEDAR: it "
            "existed before your predictions, so it establishes or checks rather than tests a prediction."])

    def record_external_evidence(args: ExternalArgs) -> ToolOutcome:
        if ctx.external is None:
            raise ToolFailure("external evidence cannot be recorded here; tell the Chief Scientist")
        value = None
        if args.value is not None:
            if not args.unit:
                raise ToolFailure("give the unit of the value")
            uncertainties = () if args.uncertainty is None else (Uncertainty(
                kind=UncertaintyKind.EXPERIMENTAL, value=args.uncertainty,
                basis=args.uncertainty_basis or "as the source gives it"),)
            value = Quantity(value=args.value, unit=args.unit, uncertainties=uncertainties)
        fact = ExternalFact(kind=args.kind, summary=args.summary, source_kind=args.source_kind, source=args.source,
                            quote=args.quote, value=value, locator=args.locator, title=args.title,
                            conditions=args.conditions, assumptions=tuple(args.assumptions),
                            limitations=tuple(args.limitations),
                            observable=Observable(name=args.observable_name, definition=args.observable_definition,
                                                  unit=args.unit))
        outcome = ctx.external.record(ctx.investigation_id, fact, reader=ctx.actor)
        if outcome.recorded is None:
            return ToolOutcome("Not recorded:\n" + "\n".join(f"- {p}" for p in outcome.problems), is_error=True)
        (event,) = outcome.recorded.events
        ev = event.payload.evidence
        shown = f"{ev.value.value!r} {ev.value.unit}" if ev.value else "no value"
        note = (" Its source could not be checked, so it stays PROVISIONAL." if ev.quality.value == "PROVISIONAL"
                else "")
        return ToolOutcome(f"Recorded as evidence {event.payload.evidence_id}: {ev.quality.value}, "
                           f"{ev.observable.name} = {shown}.{note} It existed before any prediction: it can "
                           "establish the discrepancy or challenge a hypothesis, not support one.")

    def other_agents(args: AgentsArgs) -> str:
        return team_text(ctx, ctx.ledger.load(ctx.investigation_id), args.hypothesis_id)

    def record_result(args: ResultArgs) -> ToolOutcome:
        analysis = ctx.ledger.load(ctx.investigation_id).analyses.get(args.analysis_id)
        if analysis is None or analysis.hypothesis_id != ctx.hypothesis_id:
            raise ToolFailure(f"{args.analysis_id} is not one of your analyses")
        _require_results(ctx)
        result = analysis.run.result(args.result_name)
        unit = result.value.unit if result is not None else None
        outcome = ctx.validator.record_analysis_result(
            ctx.investigation_id, args.analysis_id, args.result_name,
            observable=Observable(name=args.observable_name, definition=args.observable_definition, unit=unit),
            summary=args.summary, limitations=tuple(args.limitations), reader=ctx.actor)
        if outcome.recorded is None:
            return ToolOutcome("Result not recorded:\n" + "\n".join(f"- {p}" for p in outcome.problems),
                               is_error=True)
        (event,) = outcome.recorded.events
        ev = event.payload.evidence
        failed = [c.name for c in ev.quality_checks if not c.passed]
        return (f"Recorded as evidence {event.payload.evidence_id}: {ev.quality.value}, {ev.observable.name} = "
                f"{ev.value.value!r} {ev.value.unit}." + (f" Failed checks: {', '.join(failed)}." if failed else ""))

    tools = [
        AgentTool("investigation_overview", "Show the investigation, your hypothesis, its predictions and every "
                  "calculation linked to it, with examination findings and evidence. Call this first.",
                  NoArgs, investigation_overview),
        AgentTool("describe_protocol", "Explain what a protocol accepts: observables, input files, how to declare "
                  "settings, and what its examiner checks. Call it before requesting a calculation.",
                  DescribeArgs, describe_protocol),
        AgentTool("set_hypothesis_status", "Move your hypothesis through its lifecycle, e.g. to PLANNING_TEST "
                  "before requesting calculations or ANALYZING when results are in.", StatusArgs,
                  set_hypothesis_status),
        AgentTool("register_prediction", "Commit a prediction, naming the calculations it predicts, before they are "
                  "submitted; any calculation of the investigation may be named. It cannot be edited later. Only "
                  "evidence from calculations predicted in time can support your hypothesis.", PredictionArgs,
                  register_prediction),
        AgentTool("request_calculation", "Request a calculation, optionally with input files you wrote yourself. "
                  "Declare every physical and numerical setting the protocol lists, and nothing else (the compute "
                  "profile, for one, comes from the investigation). The protocol's examiner checks the files at "
                  "once and its findings come back; a rejected request must be fixed and made again.",
                  CalculationArgs, request_calculation),
        AgentTool("submit_report", "End your task with your report to the Chief Scientist: how it ended, what you "
                  "found with the evidence ids, open questions, what you need from the researcher or the Chief "
                  "Scientist, and next steps. To wait for calculations, name them in wait_for: you are woken when "
                  "they have ended. Nothing runs after this call.", ReportArgs, submit_report),
        AgentTool("run_outputs", "Show a finished calculation's run check, its stored outputs, and every file "
                  "in its run directory, with absolute paths for reading with the file tools.", CalculationArg,
                  run_outputs),
        AgentTool("submit_reading", "Record a value printed in a run's stored output, citing the exact lines; the "
                  "quoted lines must contain the value and its unit (include a header line if the unit is only "
                  "there). It is checked against the stored file before it is recorded.", ReadingArgs,
                  submit_reading),
        AgentTool("run_analysis", "Run a Python script you wrote on files of checked runs, on the login node, "
                  "for quantities that are not printed as such (eigenvalue gaps, geometries, averages, "
                  "differences between runs, a comparison with measured data) or to explore outputs. Stage the "
                  "runs' files (inputs/<calculation_id>/) and any attached materials or other files "
                  "(inputs/data/). The script is examined and run isolated: it sees only its staged inputs, "
                  "writes only in its working directory, cannot start programs or use the network, and is limited "
                  "in time and memory. Every run is recorded with the hash of every file it read.", AnalysisArgs,
                  run_analysis),
        AgentTool("record_result", f"Record one result an analysis wrote to {RESULTS_FILE} as evidence. The value "
                  "is taken from the analysis record, never typed by you; its quality follows the health of the "
                  "runs it used.", ResultArgs, record_result),
        AgentTool("record_external_evidence", "Record a measured value or a fact about the experiment as evidence, "
                  "citing its source and quoting it: a researcher's message (often a single sentence such as 'the "
                  "measured adsorption height is 2.5 Å'), the investigation's recorded measurement, an attached "
                  "material, a file, or a web page. Quotes from text sources are checked; values read off figures "
                  "or from web pages stay PROVISIONAL.", ExternalArgs, record_external_evidence),
        AgentTool("other_agents", "Show the other agents' work: their hypotheses, calculations with the directories "
                  "their files are in, evidence and latest reports. You may read their files and use their "
                  "calculations in your readings and analyses when your own test needs them.", AgentsArgs,
                  other_agents),
        AgentTool("register_existing_run", "Register a finished FHI-aims run computed outside CEDAR, e.g. one of "
                  "the researcher's earlier calculations, for your hypothesis. Nothing runs: its files are stored by "
                  "hash, its health is checked from its output, and the settings its inputs used are recorded. You "
                  "can then read and analyse it like your own runs, and evidence from it is marked as computed "
                  "outside CEDAR.", RegisterArgs, register_existing_run),
    ]
    if ctx.workspace is not None:
        tools += workspace_tools(ctx.workspace)
    if ctx.bench is not None:
        tools += [
            AgentTool("run_command", "Run a shell command or script in your workspace, to check things: build or "
                      "inspect a structure, test a script on a sample, search outputs, run tests. It runs in a "
                      "sandbox: it can read what you can read, writes only in your workspace (and /tmp), has no "
                      "network, cannot reach the scheduler and cannot run the DFT codes; DFT runs only as a "
                      "requested calculation. It has a time limit and is recorded. What it computes is context "
                      "for your work, not evidence.", CommandArgs, run_command),
            AgentTool("workbench_runs", "Show where your commands run, with how many cores and minutes, and the "
                      "commands you ran.", NoArgs, workbench_runs),
        ]
    return tools
