"""Provider-neutral conversation types shared by every model adapter.

An assistant message keeps the provider's own content blocks (``native``) next to the
neutral parts. When a conversation goes back to the same provider, the native blocks are
sent unchanged: some providers require their own blocks, such as reasoning signatures, to
come back verbatim in an append-only history. Other providers see only the neutral parts.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, Protocol

from pydantic import Field, JsonValue

from cedar.contracts.base import Contract, NonEmptyStr


class TextPart(Contract):
    type: Literal["text"] = "text"
    text: str


class ToolCall(Contract):
    type: Literal["tool_call"] = "tool_call"
    id: NonEmptyStr
    name: NonEmptyStr
    arguments: dict[str, JsonValue]


MediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"]


class MediaPart(Contract):
    """An image or a PDF a tool hands the model, base64-encoded."""

    type: Literal["media"] = "media"
    media_type: MediaType
    data: NonEmptyStr
    title: str | None = None


class ToolResult(Contract):
    type: Literal["tool_result"] = "tool_result"
    call_id: NonEmptyStr
    content: str
    is_error: bool = False
    media: tuple[MediaPart, ...] = ()
    """Images and PDFs shown to the model with the text, e.g. a figure the agent asked to view."""


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


class WebAccess(Contract):
    """Web search and web fetch for one request, run by the provider itself, never by CEDAR."""

    allowed_domains: tuple[NonEmptyStr, ...] = ()
    """Only these domains (and their subdomains); empty for the whole web."""
    max_searches: int = Field(default=5, ge=1)
    """Per model call."""
    max_fetches: int = Field(default=10, ge=1)
    """Per model call."""
    max_fetch_tokens: int = Field(default=50_000, ge=1)
    """Approximate limit on the text of one fetched page; PDFs are not cut."""


class Usage(Contract):
    input_tokens: int = 0
    """Uncached input; cached input is counted separately below."""
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    web_fetches: int = 0

    @property
    def total(self) -> int:
        """Every token processed, cached or not."""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "pause_turn", "other"]
"""``pause_turn``: the provider paused its own tool loop (e.g. web search) and continues when the reply is sent
back unchanged."""


class ModelReply(Contract):
    message: Message
    stop: StopReason
    usage: Usage
    requested_model: NonEmptyStr
    served_model: NonEmptyStr
    """The model that actually answered, which differs from the request after a fallback."""
    refusal_category: str | None = None
    dropped_reasoning_blocks: int = Field(default=0, ge=0)
    """Earlier reasoning blocks the provider left out because the conversation they belong to has
    changed since they were made, e.g. tool definitions updated by a new CEDAR version."""


class ModelPrices(Contract):
    """What a model's tokens cost, in US dollars per million tokens of each kind, and a web search in dollars."""

    input: float = Field(ge=0)
    """Fresh input."""
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    """Input read from the prompt cache."""
    cache_write: float = Field(ge=0)
    """Input written to the prompt cache."""
    web_search: float = Field(default=0.0, ge=0)
    """Per search the provider runs."""

    def cost(self, usage: "Usage") -> float:
        """What ``usage`` cost, in dollars."""
        tokens = (usage.input_tokens * self.input + usage.output_tokens * self.output
                  + usage.cache_read_tokens * self.cache_read + usage.cache_write_tokens * self.cache_write)
        return tokens / 1_000_000 + usage.web_searches * self.web_search


class ModelSettings(Contract):
    """Which model a role uses. Configuration, never hard-coded."""

    provider: NonEmptyStr
    model: NonEmptyStr
    prices: ModelPrices | None = None
    """What its tokens cost; None takes the prices published for the model, if CEDAR knows them."""
    max_output_tokens: int = Field(default=64000, gt=0)
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    """Reasoning effort, for providers that support it; None sends none, for models without it."""
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
                 settings: ModelSettings, web: WebAccess | None = None) -> ModelReply:
        """One model turn. ``web`` gives the model the provider's web search and fetch; an adapter whose provider
        has none raises a non-retryable ModelError rather than leaving them out silently."""
        ...
