"""The input reviewer's verdict on a calculation's inputs, before anything runs.

Deterministic checks cover what runs on the machine: the job script against the approved
template, its resources and the task's budget. Whether the scientific inputs make sense
(what each keyword does in the installed code version, whether the settings fit the
estimand and the comparison they serve, whether the structure is what the task describes)
is judged by the input reviewer, an agent that reads the files and the code's documentation.
Its verdict is a model's judgement, recorded as its own and never as evidence.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from cedar.contracts.base import Contract, FileName, Identifier, NonEmptyStr


class ReviewDecision(StrEnum):
    APPROVED = "input_agent_approved"
    """The inputs are fine; submission still needs the researcher's authorization."""
    WARNED = "input_agent_warned"
    """The researcher should look at the inputs in person before they can run."""
    REJECTED = "input_agent_rejected"
    """The inputs must change: they go back to the agent with the reviewer's reasons."""


class ReviewSeverity(StrEnum):
    NOTE = "note"
    """Worth knowing; no action needed."""
    WARNING = "warning"
    """The researcher should look before the inputs run."""
    PROBLEM = "problem"
    """The inputs must change."""


class ReviewSource(Contract):
    """Documentation a finding rests on: a file of the installed code, or a web page."""

    source: NonEmptyStr = Field(description="An absolute path of a local file (the installed manual or source), or "
                                            "a URL.")
    lines: str | None = Field(default=None, description="For a local file, the line or line range quoted, e.g. "
                                                        "'6512-6515'.")
    quote: NonEmptyStr = Field(description="The words of the source the finding relies on, copied exactly.")
    verified: bool = False
    """The quote was found in the local file, at the given lines if any; web pages cannot be checked."""


class ReviewFinding(Contract):
    severity: ReviewSeverity
    text: NonEmptyStr
    file: FileName | None = None
    line: int | None = Field(default=None, ge=1)
    keyword: str | None = None
    sources: tuple[ReviewSource, ...] = ()


SettingRole = Literal["physical_model", "numerical", "convergence_aid", "output", "other"]
"""What a control.in keyword does: change the physical model or estimand, set numerical accuracy, only change how a
result is reached, only change what is printed, or something else."""

_DECISION_FOR = {ReviewSeverity.PROBLEM: ReviewDecision.REJECTED, ReviewSeverity.WARNING: ReviewDecision.WARNED}


class InputReviewRecorded(Contract):
    """The input reviewer's review of one calculation's inputs."""

    type: Literal["input_review_recorded"] = "input_review_recorded"
    review_id: Identifier
    calculation_id: Identifier
    decision: ReviewDecision
    summary: NonEmptyStr
    findings: tuple[ReviewFinding, ...] = ()
    settings: dict[NonEmptyStr, SettingRole] = {}
    """What each keyword set in control.in does, as the reviewer classified it."""
    task_id: Identifier | None = None
    """The reviewer's task that produced it."""

    @model_validator(mode="after")
    def _decision_follows_findings(self) -> "InputReviewRecorded":
        severities = {f.severity for f in self.findings}
        worst = next((s for s in (ReviewSeverity.PROBLEM, ReviewSeverity.WARNING) if s in severities), None)
        expected = _DECISION_FOR.get(worst, ReviewDecision.APPROVED)
        if self.decision is not expected:
            raise ValueError(f"a review whose most serious finding is {worst.value if worst else 'a note or none'} is "
                             f"{expected.value}, not {self.decision.value}")
        return self
