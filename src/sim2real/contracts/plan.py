"""Versioned research plans and user authorizations."""

from enum import StrEnum

from pydantic import Field

from sim2real.contracts.base import Contract, Identifier, NonEmptyStr
from sim2real.contracts.task import ResourceBudget


class Priority(StrEnum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    ON_HOLD = "on_hold"
    """No new calculations; work already submitted keeps running."""


class ResearchPlan(Contract):
    objective: NonEmptyStr
    hypothesis_priorities: dict[Identifier, Priority] = {}
    budget: ResourceBudget | None = None
    notes: tuple[NonEmptyStr, ...] = ()

    def priority(self, hypothesis_id: str) -> Priority:
        return self.hypothesis_priorities.get(hypothesis_id, Priority.NORMAL)


class PermissionAction(StrEnum):
    SUBMIT = "submit"
    CANCEL = "cancel"


class Authorization(Contract):
    """Explicit user permission for one action on named targets; no wildcards."""

    action: PermissionAction
    target_ids: tuple[Identifier, ...] = Field(min_length=1)
