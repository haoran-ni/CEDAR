"""Tools of the input reviewer: judge the scientific content of a calculation's inputs before they can run.

Deterministic checks have already covered what runs on the machine (the job script, its resources, the budget) and
that control.in sets what the task declares. The reviewer reads the files, the installed code's manual and source and
the code's documentation on the web, and decides: the inputs are fine (the researcher still authorizes submission),
the researcher should look at them first, or they go back to the agent with the reviewer's reasons. Claims about what
a keyword does cite their source; quotes from local files are checked. Its verdicts are its own, never evidence.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from cedar.agents.toolkits.files import FileAccess
from cedar.agents.tools import AgentTool, ToolFailure
from cedar.contracts.base import Actor, Contract, FileName, Identifier, NonEmptyStr, new_id
from cedar.contracts.events import CalculationStateChanged, EventDraft
from cedar.contracts.examination import ExaminationVerdict
from cedar.contracts.review import (
    InputReviewRecorded,
    ReviewDecision,
    ReviewFinding,
    ReviewSeverity,
    ReviewSource,
    SettingRole,
)
from cedar.contracts.status import CalculationState
from cedar.engine.inputs import PROTOCOL_ENGINE
from cedar.hpc.profile import ComputeProfile
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import (
    MAX_REVIEW_REJECTIONS,
    AgentTaskRecord,
    CalculationRecord,
    InvestigationState,
    awaits_reviewer,
    escalated,
    reviewer_rejections,
    same_request,
)
from cedar.store.artifacts import ArtifactStore
from cedar.summaries import rejection_reason, requested_by, run_directory

MAX_FILE_LINES = 400
"""Lines of each input file shown in full; the rest is read with the file tools."""
MAX_SOURCE_BYTES = 50_000_000
"""Largest local file a quote is checked against."""
DOC_MARKERS = ("doc/manual", "src/read_control.f90")
"""What an FHI-aims source tree holds: the manual's LaTeX sources and the parser of control.in."""


@dataclass
class ReviewerContext:
    ledger: ResearchLedger
    investigation_id: str
    actor: Actor
    artifacts: ArtifactStore
    task: AgentTaskRecord
    files: FileAccess
    """What the reviewer may read; quotes from local files are checked through it."""
    profile: ComputeProfile | None = None
    """The approved compute profile: where the code, its version and its data are."""


class NoArgs(Contract):
    pass


class SourceArgs(Contract):
    source: NonEmptyStr = Field(description="An absolute path of a local file (the installed manual or source), or "
                                            "a URL of a documentation page you read.")
    lines: str | None = Field(default=None, description="For a local file, the line or line range you quote, e.g. "
                                                        "'6512-6515'.")
    quote: NonEmptyStr = Field(description="The words of the source your finding relies on, copied exactly.")


class FindingArgs(Contract):
    severity: ReviewSeverity = Field(description="note: worth knowing; warning: the researcher should look before "
                                                 "it runs; problem: the inputs must change.")
    text: NonEmptyStr = Field(description="What you found and why it matters for this calculation.")
    file: FileName | None = Field(default=None, description="The input file it concerns, e.g. control.in.")
    line: int | None = Field(default=None, ge=1, description="The line of that file.")
    keyword: str | None = Field(default=None, description="The keyword it concerns, if any.")
    sources: list[SourceArgs] = Field(default_factory=list, description="The documentation it rests on. Cite a "
                                      "source for every claim about what a keyword does or which values it takes.")


class ReviewArgs(Contract):
    calculation_id: Identifier
    decision: ReviewDecision = Field(description="input_agent_approved (no warning or problem), input_agent_warned "
                                                 "(a warning is the most serious finding), input_agent_rejected (a "
                                                 "problem was found).")
    summary: NonEmptyStr = Field(description="Your verdict in two or three sentences, for the researcher and the "
                                             "agent.")
    findings: list[FindingArgs] = Field(default_factory=list)
    settings: dict[str, SettingRole] = Field(default_factory=dict, description="What every keyword set in control.in "
                                             "does: physical_model, numerical, convergence_aid, output or other.")


def installation_lines(profile: ComputeProfile | None) -> list[str]:
    """Where the installed code is, and its manual and source if they can be found next to it."""
    if profile is None or "fhi-aims" not in profile.codes:
        return ["No compute profile with an FHI-aims installation; rely on the web documentation."]
    code = profile.codes["fhi-aims"]
    lines = [f"Installed FHI-aims {code.version}: binary {code.binary}"]
    lines += [f"  {role}: {path}" for role, path in code.data_paths.items()]
    root = source_tree(*code.data_paths.values(), code.binary)
    if root is None:
        lines.append("  No source tree with doc/manual or src/read_control.f90 was found near these paths; look for "
                     "one with the file tools, or rely on the web documentation for this version.")
    else:
        lines += [f"  Source tree of this installation: {root}",
                  f"  Manual (LaTeX sources, for exactly this version): {root / 'doc' / 'manual'}"
                  if (root / "doc" / "manual").is_dir() else "  No manual in the source tree.",
                  f"  control.in parser (what each keyword reads; keywords missing from the manual are here): "
                  f"{root / 'src' / 'read_control.f90'}" if (root / "src" / "read_control.f90").is_file() else
                  "  No src/read_control.f90 in the source tree."]
    return lines


def source_tree(*paths: str) -> Path | None:
    """The nearest directory above one of ``paths``, or beside one of their parents, that holds an FHI-aims manual or
    the control.in parser; looks at most four levels up."""
    for path in paths:
        for candidate in list(Path(path).parents)[:4]:
            if candidate == candidate.parent:
                break
            try:
                if any((candidate / marker).exists() for marker in DOC_MARKERS):
                    return candidate
                siblings = [d for d in candidate.iterdir() if d.is_dir() and any((d / m).exists() for m in DOC_MARKERS)]
            except OSError:
                continue
            if len(siblings) == 1:
                return siblings[0]
    return None


def _file_text(ctx: ReviewerContext, calc: CalculationRecord, name: str) -> str:
    text = ctx.artifacts.read(calc.examination.files[name]).decode(errors="replace")
    lines = text.splitlines()
    numbered = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(lines[:MAX_FILE_LINES], start=1))
    more = (f"\n      … {len(lines) - MAX_FILE_LINES} more lines; read them with read_file"
            if len(lines) > MAX_FILE_LINES else "")
    return f"--- {name} ---\n{numbered}{more}"


def review_material(ctx: ReviewerContext, state: InvestigationState, calc: CalculationRecord) -> str:
    """Everything the reviewer needs to judge one calculation's inputs."""
    task, exam = calc.task, calc.examination
    lines = [f"Calculation {calc.calculation_id}, requested by {requested_by(state, calc)}",
             f"  protocol {task.protocol.name} {task.protocol.version}; runs "
             + ("locally on the interactive node" if task.execution == "local" else "through Slurm"),
             f"  observable: {task.observable.name} ({task.observable.definition})"
             + (f" [{task.observable.unit}]" if task.observable.unit else ""),
             f"  estimand: {task.estimand}; system: {task.system_label}",
             f"  declared interventions {task.interventions}; controls {task.controls}; numerical settings "
             f"{task.numerical_settings}"]
    for hypothesis_id in calc.hypothesis_ids:
        hyp = state.hypotheses[hypothesis_id]
        lines.append(f"  for hypothesis {hypothesis_id}: {hyp.spec.statement}")
    lines += [f"  predicted by {r.hypothesis_id} ({r.prediction_id}): {r.prediction.expected_outcome}; falsified if "
              f"{r.prediction.falsified_if}" for r in state.predictions.values()
              if calc.calculation_id in r.prediction.calculation_ids]
    directory = run_directory(calc, ctx.profile.runs_root if ctx.profile else None)
    lines.append(f"  files in {directory}" if directory else "  no run directory")
    lines.append("What the deterministic examination recorded (errors would have rejected the inputs already; "
                 "review findings concern the job script and need the researcher whatever you decide):")
    lines += [f"  [{f.severity.value}] {f.code}: {f.message}" + (f" ({f.file}" + (f":{f.line}" if f.line else "")
                                                                + ")" if f.file else "")
              for f in exam.findings if f.code != "generated"]
    generated = [f.file for f in exam.findings if f.code == "generated" and f.file]
    if generated:
        lines.append(f"  generated by the protocol: {', '.join(generated)}")
    lines.append(f"Keywords control.in sets (classify every one of them): {', '.join(exam.settings) or 'none'}")
    earlier = reviewer_rejections(state, calc)
    if earlier:
        lines.append(f"Earlier revisions of this request you sent back ({len(earlier)} of {MAX_REVIEW_REJECTIONS}; "
                     + ("you cannot send this one back: if you find a problem, the researcher decides" if
                        escalated(state, calc) else "check that your reasons were addressed") + "):")
        lines += [f"  {c}: {state.calculations[c].end_reason}" for c in earlier]
    siblings = [c for c in state.calculations.values() if c.calculation_id != calc.calculation_id
                and c.examination is not None and not c.registered and set(c.hypothesis_ids) & set(calc.hypothesis_ids)
                and c.state is not CalculationState.REJECTED]
    if siblings:
        lines.append("Other calculations of the same hypothesis, which its tests may compare with this one (keywords "
                     "whose setting differs from this one's):")
        for other in siblings:
            keys = sorted(set(other.examination.settings) | set(exam.settings))
            differ = [f"{k}: {other.examination.settings.get(k, '(absent)')!r} vs {exam.settings.get(k, '(absent)')!r}"
                      for k in keys if other.examination.settings.get(k) != exam.settings.get(k)]
            same = " (the same request)" if same_request(other, calc) else ""
            lines.append(f"  {other.calculation_id}{same}, {other.state.value}: {task_line(other)}; "
                         + ("; ".join(differ) if differ else "no keyword differs"))
    lines += ["", *installation_lines(ctx.profile), ""]
    lines += [_file_text(ctx, calc, name) for name in sorted(exam.files)]
    return "\n".join(lines)


def task_line(calc: CalculationRecord) -> str:
    return f"{calc.task.system_label}: {calc.task.estimand}"


def _normalized(text: str) -> str:
    return " ".join(text.split())


def check_source(ctx: ReviewerContext, source: SourceArgs) -> ReviewSource:
    """A cited source as recorded: a quote from a local file must be in it, at the lines given if any."""
    if re.match(r"^https?://", source.source):
        return ReviewSource(source=source.source, lines=source.lines, quote=source.quote, verified=False)
    path = ctx.files.regular_file(source.source)
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ToolFailure(f"{source.source} is too large to check a quote against; quote a smaller file")
    text = path.read_text(errors="replace")
    if source.lines:
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", source.lines)
        if match is None:
            raise ToolFailure(f"lines {source.lines!r}: give a line or a range such as 6512-6515")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        text = "\n".join(text.splitlines()[first - 1:last])
    if _normalized(source.quote) not in _normalized(text):
        where = f" at lines {source.lines}" if source.lines else ""
        raise ToolFailure(f"the quote is not in {source.source}{where}; copy the words exactly, or cite the lines "
                          "where they are")
    return ReviewSource(source=str(path), lines=source.lines, quote=source.quote, verified=True)


def reviewer_tools(ctx: ReviewerContext) -> list[AgentTool]:
    def load() -> InvestigationState:
        return ctx.ledger.load(ctx.investigation_id)

    def inputs_to_review(_: NoArgs) -> str:
        state = load()
        waiting = [state.calculations[c] for c in ctx.task.calculation_ids
                   if awaits_reviewer(state, state.calculations[c])]
        done = [c for c in ctx.task.calculation_ids if c not in {w.calculation_id for w in waiting}]
        parts = [review_material(ctx, state, calc) for calc in waiting]
        if done:
            parts.append(f"No longer waiting for you (reviewed, or decided by the researcher): {', '.join(done)}")
        return "\n\n".join(parts) if waiting else (parts[0] if parts else "Nothing waits for your review.")

    def record(args: ReviewArgs) -> str:
        if args.calculation_id not in ctx.task.calculation_ids:
            raise ToolFailure(f"your task reviews {', '.join(ctx.task.calculation_ids)}, not {args.calculation_id}")
        state = load()
        calc = state.calculations.get(args.calculation_id)
        if calc is None or not awaits_reviewer(state, calc):
            raise ToolFailure(f"the inputs of {args.calculation_id} no longer wait for your review")
        unclassified = sorted(set(calc.examination.settings) - set(args.settings))
        if unclassified:
            raise ToolFailure(f"classify every keyword control.in sets; missing: {', '.join(unclassified)}")
        findings = tuple(ReviewFinding(severity=f.severity, text=f.text, file=f.file, line=f.line, keyword=f.keyword,
                                       sources=tuple(check_source(ctx, s) for s in f.sources))
                         for f in args.findings)
        review = InputReviewRecorded(review_id=new_id("review"), calculation_id=args.calculation_id,
                                     decision=args.decision, summary=args.summary, findings=findings,
                                     settings=args.settings, task_id=ctx.task.task_id)
        outcome: list[str] = []

        def build(current: InvestigationState) -> list[EventDraft]:
            drafts = [EventDraft(actor=ctx.actor, payload=review)]
            now = current.calculations[args.calculation_id]
            if args.decision is ReviewDecision.APPROVED and now.examination.verdict is ExaminationVerdict.APPROVED:
                drafts.append(EventDraft(actor=PROTOCOL_ENGINE, payload=CalculationStateChanged(
                    calculation_id=args.calculation_id, state=CalculationState.VALIDATED_INPUT,
                    authorization_id=review.review_id)))
                outcome[:] = ["approved"]
            elif args.decision is ReviewDecision.REJECTED and not escalated(current, now):
                drafts.append(EventDraft(actor=PROTOCOL_ENGINE, payload=CalculationStateChanged(
                    calculation_id=args.calculation_id, state=CalculationState.REJECTED,
                    reason=rejection_reason(review), authorization_id=review.review_id)))
                outcome[:] = ["rejected"]
            else:
                outcome[:] = ["researcher"]
            return drafts

        ctx.ledger.record_with(ctx.investigation_id, build)
        unverified = sum(1 for f in findings for s in f.sources if not s.verified)
        checked = (f" {unverified} web source{'s' * (unverified != 1)} could not be checked and are recorded as "
                   "cited." if unverified else "")
        if outcome == ["approved"]:
            return (f"Recorded {review.review_id}: approved; the inputs wait for the researcher to authorize "
                    "submission." + checked)
        if outcome == ["rejected"]:
            return f"Recorded {review.review_id}: rejected; the agent is woken with your reasons." + checked
        why = ("you sent back the most revisions of this request you may" if args.decision is ReviewDecision.REJECTED
               else "the job script needs their review too" if args.decision is ReviewDecision.APPROVED
               else "you warned")
        return f"Recorded {review.review_id}: the researcher decides, since {why}." + checked

    return [
        AgentTool("inputs_to_review", "The inputs your task reviews: what each calculation is for, the deterministic "
                  "examination's facts, earlier rejected revisions, the hypothesis's other calculations, where the "
                  "installed code's manual and source are, and the files, numbered by line.", NoArgs,
                  inputs_to_review),
        AgentTool("record_input_review", "Record your verdict on one calculation's inputs, with every finding and its "
                  "sources, and what every keyword control.in sets does. Approved inputs wait for the researcher to "
                  "authorize submission; warned ones for the researcher to look at; rejected ones go back to the "
                  "agent with your reasons.", ReviewArgs, record),
    ]

