"""The provider-neutral agent loop.

One invocation appends the new instruction to the session's transcript, then alternates
model turns and tool calls until the agent answers without calling a tool, a tool ends
the run (for example to wait for a job), the money it may spend is spent, or the caller asks
it to stop (checked before every model call, where the transcript is complete). There is no
limit on the number of model calls unless the caller sets one. What each call costs follows
from the model's prices (``models/prices.py``). Tools from a turn
that was cut off (``max_tokens``) or refused are never run; their calls are answered with
an error so the transcript stays valid for the next invocation.

With ``web``, the provider's own web search and fetch run inside a model turn. When the
provider pauses that loop, the paused reply is sent straight back so it can continue: up to
``MAX_PAUSED_CONTINUATIONS`` times in a row, even past the turn and token limits, since a
transcript left on a paused turn could not take the next instruction. A stop request still
stops it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from cedar.agents.tools import AgentTool, ToolOutcome
from cedar.agents.transcripts import TranscriptStore
from cedar.models.prices import prices_for
from cedar.models.types import (
    Message,
    ModelClient,
    ToolCall,
    ModelError,
    ModelSettings,
    TextPart,
    ToolResult,
    Usage,
    WebAccess,
)
from cedar.secrets import redact

Outcome = Literal["finished", "waiting", "reported", "turn_limit", "budget_limit", "truncated", "refused",
                  "model_error", "stopped"]

MAX_PAUSED_CONTINUATIONS = 5


@dataclass(frozen=True)
class RunLimits:
    max_turns: int | None = None
    """Model calls in one invocation; None for no limit."""
    max_cost_usd: float | None = None
    """Dollars one invocation may spend on the model; None for no limit."""


@dataclass
class RunResult:
    outcome: Outcome
    final_text: str = ""
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    served_models: list[str] = field(default_factory=list)
    error: str | None = None
    dropped_reasoning_blocks: int = 0
    context_tokens: int | None = None
    """The size of the last model call's input: how long the conversation has grown."""
    retryable: bool = False
    """For a model error: whether trying again later may work (a rate limit, an overloaded or unreachable service)."""
    cost_usd: float | None = None
    """What the invocation cost, at the model's prices; None when they are not known."""


def _add(total: Usage, more: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + more.input_tokens,
        output_tokens=total.output_tokens + more.output_tokens,
        cache_read_tokens=total.cache_read_tokens + more.cache_read_tokens,
        cache_write_tokens=total.cache_write_tokens + more.cache_write_tokens,
        web_searches=total.web_searches + more.web_searches,
        web_fetches=total.web_fetches + more.web_fetches,
    )


class AgentRunner:
    def __init__(self, client: ModelClient, settings: ModelSettings, transcripts: TranscriptStore, *,
                 secrets: Sequence[str] = ()) -> None:
        self._client = client
        self._settings = settings
        self._prices = prices_for(settings)
        self._transcripts = transcripts
        self._secrets = list(secrets)

    def run(self, session_id: str, *, system: str, tools: Sequence[AgentTool], instruction: str,
            limits: RunLimits = RunLimits(),
            observer: Callable[[ToolCall, ToolOutcome], None] | None = None,
            should_stop: Callable[[], str | None] | None = None, web: WebAccess | None = None,
            on_turn: Callable[["RunResult"], None] | None = None,
            spend_left: Callable[[float], float | None] | None = None) -> RunResult:
        """Run one invocation; ``observer`` sees each tool call and its outcome, e.g. to show progress,
        ``on_turn`` the result so far after each model call, ``should_stop`` is asked before every model call,
        returning why the run must stop, or None, and ``web`` gives the model the provider's web search and
        fetch. ``spend_left``, given what this run has spent, says what a budget it shares with other runs has
        left (None for no limit); it is asked before every model call too."""
        by_name = {tool.name: tool for tool in tools}
        specs = [tool.spec() for tool in tools]
        history = self._transcripts.load(session_id)
        self._append(session_id, history, Message(role="user", parts=(TextPart(text=instruction),)))
        result = RunResult(outcome="finished", cost_usd=0.0 if self._prices is not None else None)
        paused = 0
        if (limits.max_cost_usd is not None or spend_left is not None) and self._prices is None:
            result.outcome = "budget_limit"
            result.error = (f"the prices of {self._settings.provider} {self._settings.model} are not known, so its "
                            "spending cannot be kept within the budget; set them in models.toml")
            return result

        while True:
            reason = should_stop() if should_stop is not None else None
            if reason:
                result.outcome, result.error = "stopped", f"stopped: {reason}"
                return result
            if not paused and limits.max_turns is not None and result.turns >= limits.max_turns:
                result.outcome = "turn_limit"
                return result
            if not paused and limits.max_cost_usd is not None and result.cost_usd >= limits.max_cost_usd:
                result.outcome = "budget_limit"
                result.error = f"its budget of ${limits.max_cost_usd:.2f} is spent"
                return result
            shared = spend_left(result.cost_usd) if spend_left is not None and not paused else None
            if shared is not None and shared <= 0:
                result.outcome = "budget_limit"
                result.error = "the budget is spent"
                return result
            try:
                reply = self._client.complete(system=system, messages=history, tools=specs, settings=self._settings,
                                              web=web)
            except ModelError as exc:
                result.outcome, result.error = "model_error", redact(str(exc), self._secrets)
                result.retryable = exc.retryable
                return result
            result.turns += 1
            result.usage = _add(result.usage, reply.usage)
            if self._prices is not None:
                result.cost_usd = self._prices.cost(result.usage)
            result.context_tokens = (reply.usage.input_tokens + reply.usage.cache_read_tokens
                                     + reply.usage.cache_write_tokens)
            if on_turn is not None:
                on_turn(result)
            result.dropped_reasoning_blocks += reply.dropped_reasoning_blocks
            if reply.served_model not in result.served_models:
                result.served_models.append(reply.served_model)
            self._append(session_id, history, reply.message)
            calls = reply.message.tool_calls

            if reply.stop in ("max_tokens", "refusal"):
                result.outcome = "truncated" if reply.stop == "max_tokens" else "refused"
                result.error = (f"model refused (category {reply.refusal_category})" if reply.stop == "refusal"
                                else "model output hit max_tokens")
                if calls:
                    self._append(session_id, history, Message(role="user", parts=tuple(
                        ToolResult(call_id=c.id, content=f"Not run: {result.error}.", is_error=True) for c in calls)))
                return result
            if reply.stop == "pause_turn" and not calls:
                paused += 1
                if paused > MAX_PAUSED_CONTINUATIONS:
                    result.outcome = "turn_limit"
                    result.error = f"the provider paused its web search {paused} times in a row"
                    return result
                continue
            paused = 0
            if not calls:
                result.final_text = redact(reply.message.text, self._secrets)
                return result

            results, end_run, ends_as = [], False, "waiting"
            for call in calls:
                tool = by_name.get(call.name)
                outcome = tool.invoke(call.arguments) if tool else ToolOutcome(
                    f"Unknown tool {call.name!r}; available: {', '.join(sorted(by_name))}", is_error=True)
                outcome = ToolOutcome(redact(outcome.content, self._secrets), outcome.is_error, outcome.end_run,
                                      ends_as=outcome.ends_as, media=outcome.media)
                if observer is not None:
                    observer(call, outcome)
                results.append(ToolResult(call_id=call.id, content=outcome.content, is_error=outcome.is_error,
                                          media=outcome.media))
                if outcome.end_run and not end_run:
                    ends_as = outcome.ends_as
                end_run = end_run or outcome.end_run
            self._append(session_id, history, Message(role="user", parts=tuple(results)))
            if end_run:
                result.outcome = ends_as
                result.final_text = redact(reply.message.text, self._secrets)
                return result

    def _append(self, session_id: str, history: list[Message], message: Message) -> None:
        self._transcripts.append(session_id, message)
        history.append(message)
