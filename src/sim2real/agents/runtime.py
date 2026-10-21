"""The provider-neutral agent loop.

One invocation appends the new instruction to the session's transcript, then alternates
model turns and tool calls until the agent answers without calling a tool, a tool ends
the run (for example to wait for a job), or a limit is reached. Tools from a turn that
was cut off (``max_tokens``) or refused are never run; their calls are answered with an
error so the transcript stays valid for the next invocation.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sim2real.agents.tools import AgentTool, ToolOutcome
from sim2real.agents.transcripts import TranscriptStore
from sim2real.models.types import (
    Message,
    ModelClient,
    ToolCall,
    ModelError,
    ModelSettings,
    TextPart,
    ToolResult,
    Usage,
)
from sim2real.secrets import redact

Outcome = Literal["finished", "waiting", "turn_limit", "token_limit", "truncated", "refused", "model_error"]


@dataclass(frozen=True)
class RunLimits:
    max_turns: int = 40
    """Model calls in one invocation."""
    max_tokens: int | None = None
    """Tokens processed in one invocation (input, cached input and output); None for no limit."""


@dataclass
class RunResult:
    outcome: Outcome
    final_text: str = ""
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    served_models: list[str] = field(default_factory=list)
    error: str | None = None


def _add(total: Usage, more: Usage) -> Usage:
    return Usage(
        input_tokens=total.input_tokens + more.input_tokens,
        output_tokens=total.output_tokens + more.output_tokens,
        cache_read_tokens=total.cache_read_tokens + more.cache_read_tokens,
        cache_write_tokens=total.cache_write_tokens + more.cache_write_tokens,
    )


class AgentRunner:
    def __init__(self, client: ModelClient, settings: ModelSettings, transcripts: TranscriptStore, *,
                 secrets: Sequence[str] = ()) -> None:
        self._client = client
        self._settings = settings
        self._transcripts = transcripts
        self._secrets = list(secrets)

    def run(self, session_id: str, *, system: str, tools: Sequence[AgentTool], instruction: str,
            limits: RunLimits = RunLimits(),
            observer: Callable[[ToolCall, ToolOutcome], None] | None = None) -> RunResult:
        """Run one invocation; ``observer`` sees each tool call and its outcome, e.g. to show progress."""
        by_name = {tool.name: tool for tool in tools}
        specs = [tool.spec() for tool in tools]
        history = self._transcripts.load(session_id)
        self._append(session_id, history, Message(role="user", parts=(TextPart(text=instruction),)))
        result = RunResult(outcome="finished")

        while True:
            if result.turns >= limits.max_turns:
                result.outcome = "turn_limit"
                return result
            if limits.max_tokens is not None and result.usage.total >= limits.max_tokens:
                result.outcome = "token_limit"
                return result
            try:
                reply = self._client.complete(system=system, messages=history, tools=specs, settings=self._settings)
            except ModelError as exc:
                result.outcome, result.error = "model_error", redact(str(exc), self._secrets)
                return result
            result.turns += 1
            result.usage = _add(result.usage, reply.usage)
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
            if not calls:
                result.final_text = reply.message.text
                return result

            results, end_run = [], False
            for call in calls:
                tool = by_name.get(call.name)
                outcome = tool.invoke(call.arguments) if tool else ToolOutcome(
                    f"Unknown tool {call.name!r}; available: {', '.join(sorted(by_name))}", is_error=True)
                outcome = ToolOutcome(redact(outcome.content, self._secrets), outcome.is_error, outcome.end_run)
                if observer is not None:
                    observer(call, outcome)
                results.append(ToolResult(call_id=call.id, content=outcome.content, is_error=outcome.is_error))
                end_run = end_run or outcome.end_run
            self._append(session_id, history, Message(role="user", parts=tuple(results)))
            if end_run:
                result.outcome = "waiting"
                result.final_text = reply.message.text
                return result

    def _append(self, session_id: str, history: list[Message], message: Message) -> None:
        self._transcripts.append(session_id, message)
        history.append(message)
