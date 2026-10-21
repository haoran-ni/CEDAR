"""Hypothesis agents as the Chief Scientist proposes them, before the researcher's review.

Each hypothesis has at most one agent at a time. The Chief Scientist decides which
hypotheses get one, what role each plays and what it does first; the researcher sees the
proposal before any agent starts.
"""

from pydantic import Field

from sim2real.contracts.base import Contract, Identifier, NonEmptyStr


class AgentSpec(Contract):
    hypothesis_id: Identifier
    role: NonEmptyStr = Field(max_length=200)
    """The specialization the work needs, in the Chief Scientist's words, e.g. 'exchange-correlation specialist'."""
    brief: NonEmptyStr
    """The agent's first task."""
    token_budget: int | None = Field(default=None, gt=0)
    """Tokens this agent may process over its lifetime; None leaves only the investigation's budget."""
    max_turns: int | None = Field(default=None, ge=1, le=200)
    """Model calls per run; None leaves the dispatcher's limit."""
