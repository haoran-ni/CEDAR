"""Hypotheses and their preregistered predictions."""

from enum import StrEnum

from pydantic import Field, JsonValue

from cedar.contracts.base import Contract, Identifier, NonEmptyStr, Observable
from cedar.contracts.task import ProtocolRef


class Quantifiability(StrEnum):
    DIRECT = "directly_quantifiable"
    APPROXIMATE = "approximately_quantifiable"
    INDIRECT = "indirectly_constrained"
    NON_IDENTIFIABLE = "non_identifiable"
    UNKNOWN = "unknown"


class HypothesisSpec(Contract):
    title: NonEmptyStr
    statement: NonEmptyStr
    rationale: NonEmptyStr
    falsifiers: tuple[NonEmptyStr, ...] = Field(min_length=1)
    """Observations that would contradict the hypothesis; at least one is required."""
    discriminating_observations: tuple[NonEmptyStr, ...] = ()
    assumptions: tuple[NonEmptyStr, ...] = ()
    mechanism: str | None = None
    """Free-form label such as "xc" or "coverage"; not drawn from a fixed roster."""
    quantifiability: Quantifiability = Quantifiability.UNKNOWN
    depends_on: tuple[Identifier, ...] = ()
    interacts_with: tuple[Identifier, ...] = ()
    required_protocols: tuple[ProtocolRef, ...] = ()
    estimated_cost: str | None = None


class Prediction(Contract):
    """An expectation committed before the outcome is seen. Never edited after recording."""

    statement: NonEmptyStr
    observable: Observable
    expected_outcome: NonEmptyStr
    falsified_if: NonEmptyStr
    conditions: dict[str, JsonValue] = {}
    calculation_ids: tuple[Identifier, ...] = ()
    """Planned calculations whose outcome this prediction is about."""
