"""Anthropic adapter: one Messages API request per call, through the official SDK.

Requests stream (and are read with ``get_final_message``) so long outputs do not hit HTTP
timeouts. Tool inputs are not streamed eagerly: only the final message is read, and the API
then checks each input before sending it, so a malformed one arrives as a value the runtime's
validation rejects and the agent corrects in the same run. Eagerly streamed, it would arrive
as JSON the SDK cannot parse, which loses the whole reply. Refusals are retried
server-side on Anthropic's recommended fallback model unless ``options.refusal_fallback``
is false; the model that actually answered is reported as ``served_model``.

Assistant turns are sent back with their reasoning blocks. The API checks that each block
still belongs to the conversation it was made in (same system prompt, tools and earlier
messages). A new CEDAR version can change tool definitions under a resumed session, so
``options.thinking_prefix_mismatch`` decides what happens to blocks that fail: "drop_block"
(the default) leaves them out and reports how many, "error" fails the request, and "off"
sends no thinking settings at all, for models without adaptive thinking.

Web search and web fetch are Anthropic's own server tools: Anthropic runs them, and their
blocks come back among the assistant's native blocks. When Anthropic pauses its tool loop
(``pause_turn``), sending the reply back unchanged lets it continue. The tool versions can
be set with ``options.web_search_tool`` and ``options.web_fetch_tool`` for models that need
older ones.
"""

from collections.abc import Sequence

import anthropic

from cedar.secrets import redact
from cedar.models.types import (
    Message,
    ModelError,
    ModelReply,
    ModelSettings,
    TextPart,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    WebAccess,
)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_BINDING_BETA = "thinking-binding-controls-2026-08-01"
_PREFIX_MISMATCH = ("drop_block", "error", "off")
_STOPS = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal",
          "pause_turn": "pause_turn"}
_WEB_SEARCH, _WEB_FETCH = "web_search_20260209", "web_fetch_20260209"
_RETRYABLE_ERRORS = {"rate_limit_error", "api_error", "overloaded_error"}


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str, *, max_retries: int = 4, timeout: float = 600.0,
                 client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._secrets = [api_key]

    def complete(self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec],
                 settings: ModelSettings, web: WebAccess | None = None) -> ModelReply:
        params: dict = {
            "model": settings.model,
            "max_tokens": settings.max_output_tokens,
            "system": system,
            "messages": [self._to_wire(m) for m in messages],
            "cache_control": {"type": "ephemeral"},
        }
        wire_tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                      for t in tools]
        if web is not None:
            wire_tools += self._web_tools(web, settings)
        if wire_tools:
            params["tools"] = wire_tools
        if settings.effort:
            params["output_config"] = {"effort": settings.effort}
        betas = []
        mismatch = settings.options.get("thinking_prefix_mismatch", "drop_block")
        if mismatch not in _PREFIX_MISMATCH:
            raise ModelError(f"options.thinking_prefix_mismatch must be one of {', '.join(_PREFIX_MISMATCH)}, "
                             f"not {mismatch!r}", retryable=False)
        if mismatch != "off":
            params["thinking"] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": mismatch}}
            betas.append(_BINDING_BETA)
        if settings.options.get("refusal_fallback", True):
            betas.append(_FALLBACK_BETA)
            params["fallbacks"] = "default"
        if betas:
            params["betas"] = betas
        try:
            with self._client.beta.messages.stream(**params) as stream:
                response = stream.get_final_message()
        except ValueError as exc:
            # The SDK could not parse a tool input at all, so the reply is lost; the turn must be re-issued.
            raise ModelError(self._clean(f"unparseable tool input from the model: {exc}"), retryable=True) from None
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ModelError(self._clean(f"Anthropic rejected the credentials: {exc}"), retryable=False) from None
        except (anthropic.BadRequestError, anthropic.NotFoundError) as exc:
            raise ModelError(self._clean(f"Anthropic rejected the request: {exc}"), retryable=False) from None
        except anthropic.RateLimitError as exc:
            raise ModelError(self._clean(f"Anthropic rate limit: {exc}"), retryable=True) from None
        except anthropic.APIStatusError as exc:
            # An error Anthropic sends after the stream has started comes with the stream's status, 200; only its
            # type says what it is.
            status = exc.status_code if exc.status_code >= 400 else "partway through the reply"
            raise ModelError(self._clean(f"Anthropic error {status}: {exc}"),
                             retryable=exc.status_code >= 500 or exc.type in _RETRYABLE_ERRORS) from None
        except anthropic.APIConnectionError as exc:
            raise ModelError(self._clean(f"cannot reach Anthropic: {exc}"), retryable=True) from None
        return self._from_wire(response, settings.model)

    def _clean(self, text: str) -> str:
        return redact(text, self._secrets)

    @staticmethod
    def _web_tools(web: WebAccess, settings: ModelSettings) -> list[dict]:
        domains = {"allowed_domains": list(web.allowed_domains)} if web.allowed_domains else {}
        return [
            {"type": settings.options.get("web_search_tool", _WEB_SEARCH), "name": "web_search",
             "max_uses": web.max_searches, **domains},
            {"type": settings.options.get("web_fetch_tool", _WEB_FETCH), "name": "web_fetch",
             "max_uses": web.max_fetches, "max_content_tokens": web.max_fetch_tokens, **domains},
        ]

    def _to_wire(self, message: Message) -> dict:
        if message.role == "assistant" and message.provider == self.provider and message.native is not None:
            return {"role": "assistant", "content": list(message.native)}
        content: list[dict] = []
        # Tool results lead a user turn; text follows.
        for part in sorted(message.parts, key=lambda p: not isinstance(p, ToolResult)):
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ToolCall):
                content.append({"type": "tool_use", "id": part.id, "name": part.name, "input": part.arguments})
            elif part.media:
                blocks = [{"type": "text", "text": part.content}] if part.content else []
                for media in part.media:
                    source = {"type": "base64", "media_type": media.media_type, "data": media.data}
                    if media.media_type == "application/pdf":
                        blocks.append({"type": "document", "source": source,
                                       **({"title": media.title} if media.title else {})})
                    else:
                        blocks.append({"type": "image", "source": source})
                content.append({"type": "tool_result", "tool_use_id": part.call_id, "content": blocks,
                                "is_error": part.is_error})
            else:
                content.append({"type": "tool_result", "tool_use_id": part.call_id, "content": part.content,
                                "is_error": part.is_error})
        return {"role": message.role, "content": content}

    def _from_wire(self, response, requested_model: str) -> ModelReply:
        parts: list = []
        for block in response.content:
            if block.type == "text":
                parts.append(TextPart(text=block.text))
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {"__unparsed__": str(block.input)}
                parts.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
        native = tuple(block.to_dict() for block in response.content)
        usage = response.usage
        server = getattr(usage, "server_tool_use", None)
        details = response.stop_details
        transformations = getattr(response, "input_transformations", None) or ()
        return ModelReply(
            message=Message(role="assistant", parts=tuple(parts), provider=self.provider, native=native),
            stop=_STOPS.get(response.stop_reason, "other"),
            usage=Usage(input_tokens=usage.input_tokens or 0, output_tokens=usage.output_tokens or 0,
                        cache_read_tokens=usage.cache_read_input_tokens or 0,
                        cache_write_tokens=usage.cache_creation_input_tokens or 0,
                        web_searches=(server.web_search_requests or 0) if server else 0,
                        web_fetches=(server.web_fetch_requests or 0) if server else 0),
            requested_model=requested_model,
            served_model=response.model,
            refusal_category=getattr(details, "category", None) if response.stop_reason == "refusal" else None,
            dropped_reasoning_blocks=sum(1 for t in transformations if getattr(t, "type", None) == "thinking_dropped"),
        )
