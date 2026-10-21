"""The Chief Scientist's recorded recommendation of what to do next, and why.

Candidates are compared qualitatively: what each would show under each hypothesis it bears
on, why that is informative, what it costs and whether it can be done. There are no
numerical information-gain scores or probabilities: those need outcome distributions and
priors the record does not hold. A step tells hypotheses apart only if they expect
different outcomes from it.
"""

from enum import StrEnum

from pydantic import Field, model_validator

from sim2real.contracts.base import Contract, Identifier, NonEmptyStr


class ActionKind(StrEnum):
    CALCULATION = "calculation"
    ANALYSIS = "analysis"
    ASSESSMENT = "assessment"
    """By the Evidence Judge."""
    MEASUREMENT = "measurement"
    """An experiment to suggest to the researcher; sim2real never carries it out."""
    CLOSE = "close"
    """Propose closing the investigation."""
    OTHER = "other"


class ActionPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExpectedOutcome(Contract):
    hypothesis_id: Identifier
    outcome: NonEmptyStr = Field(description="What the step would show if this hypothesis holds.")


class CandidateAction(Contract):
    candidate_id: Identifier = Field(description="Short and unique within the recommendation, e.g. 'a1'.")
    kind: ActionKind
    action: NonEmptyStr = Field(description="What would be done, specifically.")
    hypothesis_ids: tuple[Identifier, ...] = Field(default=(), description="The hypotheses it bears on; every "
                                                                            "kind but close bears on at least one.")
    expected_outcomes: tuple[ExpectedOutcome, ...] = Field(
        default=(), description="What it would show under each hypothesis it bears on, where that can be said.")
    rationale: NonEmptyStr = Field(description="Why it is informative: which hypotheses it tells apart and how, or "
                                               "what it checks.")
    cost: NonEmptyStr = Field(description="Stated estimate, e.g. calculations, core-hours, tokens, time.")
    feasibility: NonEmptyStr = Field(description="Whether it can be done now: protocol available, approvals needed.")
    priority: ActionPriority
    priority_reason: NonEmptyStr

    @model_validator(mode="after")
    def _bears_on_hypotheses(self) -> "CandidateAction":
        if self.kind is not ActionKind.CLOSE and not self.hypothesis_ids:
            raise ValueError(f"a {self.kind.value} candidate names the hypotheses it bears on")
        expected = [o.hypothesis_id for o in self.expected_outcomes]
        if len(set(expected)) != len(expected):
            raise ValueError("one expected outcome per hypothesis")
        if not set(expected) <= set(self.hypothesis_ids):
            raise ValueError("expected outcomes are for hypotheses the candidate bears on")
        return self
