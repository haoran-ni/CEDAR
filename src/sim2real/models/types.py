"""Provider-neutral conversation types shared by every model adapter.

An assistant message keeps the provider's own content blocks (``native``) next to the
neutral parts. When a conversation goes back to the same provider, the native blocks are
sent unchanged: some providers require their own blocks, such as reasoning signatures, to
come back verbatim in an append-only history. Other providers see only the neutral parts.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, Protocol

from pydantic import Field, JsonValue

from sim2real.contracts.base import Contract, NonEmptyStr


class TextPart(Contract):
    type: Literal["text"] = "text"
    text: str


class ToolCall(Contract):
    type: Literal["tool_call"] = "tool_call"
    id: NonEmptyStr
    name: NonEmptyStr
    arguments: dict[str, JsonValue]


class ToolResult(Contract):
    type: Literal["tool_result"] = "tool_result"
    call_id: NonEmptyStr
    content: str
    is_error: bool = False


Part = Annotated[TextPart | ToolCall | ToolResult, Field(discriminator="type")]


class Message(Contract):
    role: Literal["user", "assistant"]
    parts: tuple[Part, ...]
    provider: str | None = None
    """For assistant messages: the provider that produced it."""
    native: tuple[dict[str, JsonValue], ...] | None = None
    """The provider's own content blocks, echoed back verbatim to that provider."""

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(p for p in self.parts if isinstance(p, ToolCall))


class ToolSpec(Contract):
    name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    description: NonEmptyStr
    input_schema: dict[str, JsonValue]


class Usage(Contract):
    input_tokens: int = 0
    """Uncached input; cached input is counted separately below."""
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        """Every token processed, cached or not."""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


class ModelReply(Contract):
    message: Message
    stop: StopReason
    usage: Usage
    requested_model: NonEmptyStr
    served_model: NonEmptyStr
    """The model that actually answered, which differs from the request after a fallback."""
    refusal_category: str | None = None


class ModelSettings(Contract):
    """Which model a role uses. Configuration, never hard-coded."""

    provider: NonEmptyStr
    model: NonEmptyStr
    max_output_tokens: int = Field(default=64000, gt=0)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """Reasoning effort, for providers that support it."""
    options: dict[str, JsonValue] = {}
    """Provider-specific extras, passed through by the adapter that understands them."""


class ModelError(RuntimeError):
    """A model call failed. ``retryable`` says whether trying again later may help."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class ModelClient(Protocol):
    provider: str

    def complete(self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec],
                 settings: ModelSettings) -> ModelReply: ...
