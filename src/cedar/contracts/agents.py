"""Hypothesis agents as the Chief Scientist proposes them, before the researcher's review.

Each hypothesis has at most one agent at a time. The Chief Scientist decides which
hypotheses get one, what role each plays and what it does first; the researcher sees the
proposal before any agent starts.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from cedar.contracts.base import Contract, Identifier, NonEmptyStr


class AgentSpec(Contract):
    hypothesis_id: Identifier
    role: NonEmptyStr = Field(max_length=200)
    """The specialization the work needs, in the Chief Scientist's words, e.g. 'exchange-correlation specialist'."""
    brief: NonEmptyStr
    """The agent's first task."""
    budget_usd: float | None = Field(default=None, gt=0)
    """Dollars this agent may spend on models over its lifetime; None leaves only the investigation's budget. There
    is no limit on an agent's model calls."""


class ReportOutcome(StrEnum):
    """How a hypothesis agent's task ended, in its own judgement."""

    CONCLUDED = "concluded"
    """The task is done: the agent says what it found."""
    WAITING_FOR_CALCULATIONS = "waiting_for_calculations"
    """It waits for calculations; it is woken when they end."""
    NEEDS_APPROVAL = "needs_approval"
    """Its calculations wait for the researcher's approval."""
    NEEDS_RESEARCHER_DATA = "needs_researcher_data"
    """It needs data or a decision from the researcher."""
    NEEDS_OTHER_AGENT = "needs_other_agent"
    """It needs another agent's result first."""
    COULD_NOT_PROCEED = "could_not_proceed"
    """Something stopped it; the report says what."""


class ReportFinding(Contract):
    statement: NonEmptyStr
    evidence_ids: tuple[Identifier, ...] = ()
    """The recorded evidence it rests on; a finding without evidence is the agent's inference."""


class ReportRequest(Contract):
    to: Literal["researcher", "chief_scientist"]
    text: NonEmptyStr


class AgentReport(Contract):
    """A hypothesis agent's report at the end of a task, in fields the Chief Scientist, the panel and the notebook
    can act on. Its words are the agent's own; values count only where they are recorded evidence."""

    outcome: ReportOutcome
    summary: NonEmptyStr
    findings: tuple[ReportFinding, ...] = ()
    open_questions: tuple[NonEmptyStr, ...] = ()
    requests: tuple[ReportRequest, ...] = ()
    next_steps: tuple[NonEmptyStr, ...] = ()
    waiting_for: tuple[Identifier, ...] = ()
    """Calculations it waits for."""

    @model_validator(mode="after")
    def _waiting_names_calculations(self) -> "AgentReport":
        if (self.outcome is ReportOutcome.WAITING_FOR_CALCULATIONS) != bool(self.waiting_for):
            raise ValueError("a report waits for calculations exactly when its outcome is waiting_for_calculations")
        return self

    def text(self) -> str:
        """The report as plain text."""
        lines = [f"[{self.outcome.value.replace('_', ' ')}] {self.summary}"]
        lines += ["Findings:"] + [f"- {f.statement}" + (f" (evidence {', '.join(f.evidence_ids)})" if f.evidence_ids
                                                       else " (no recorded evidence)") for f in self.findings] \
            if self.findings else []
        lines += ["Open questions:"] + [f"- {q}" for q in self.open_questions] if self.open_questions else []
        lines += ["Requests:"] + [f"- to the {r.to.replace('_', ' ')}: {r.text}" for r in self.requests] \
            if self.requests else []
        lines += ["Next steps:"] + [f"- {n}" for n in self.next_steps] if self.next_steps else []
        if self.waiting_for:
            lines.append(f"Waiting for: {', '.join(self.waiting_for)}")
        return "\n".join(lines)
