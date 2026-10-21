"""TEST DOUBLE. A model that replays scripted replies and records every request it receives."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sim2real.models.types import Message, ModelReply, ModelSettings, TextPart, ToolCall, ToolSpec, Usage, WebAccess


@dataclass
class ScriptedCall:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    settings: ModelSettings
    web: WebAccess | None = None


Script = ModelReply | Callable[[ScriptedCall], ModelReply]


@dataclass
class ScriptedModel:
    script: list[Script]
    provider: str = "scripted"
    calls: list[ScriptedCall] = field(default_factory=list)

    def complete(self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec],
                 settings: ModelSettings, web: WebAccess | None = None) -> ModelReply:
        call = ScriptedCall(system, tuple(messages), tuple(tools), settings, web)
        self.calls.append(call)
        if not self.script:
            raise AssertionError("the scripted model ran out of replies")
        step = self.script.pop(0)
        return step(call) if callable(step) else step


def reply(*parts: TextPart | ToolCall, stop: str | None = None, model: str = "scripted-model") -> ModelReply:
    """Build a scripted assistant reply; ``stop`` defaults to tool_use when a tool is called."""
    has_calls = any(isinstance(p, ToolCall) for p in parts)
    return ModelReply(
        message=Message(role="assistant", parts=parts, provider="scripted"),
        stop=stop or ("tool_use" if has_calls else "end_turn"),
        usage=Usage(input_tokens=10, output_tokens=5),
        requested_model=model,
        served_model=model,
    )
