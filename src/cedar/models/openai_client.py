"""OpenAI adapter: one Responses API request per call, through the official SDK.

Requests stream so long outputs do not hit HTTP timeouts. The adapter reads the stream's
events itself and keeps the response the last one carries, so that a reply cut off by
``max_output_tokens`` comes back like any other. Requests are stateless (``store`` false):
the whole conversation goes with every call, as it does to Anthropic, and OpenAI keeps no
copy to serve later requests. Prompt caching is OpenAI's default and needs no setting.

Assistant turns are sent back as the output items they were, reasoning included: GPT-5.6
models read earlier turns' reasoning, which their reasoning items carry encrypted. A tool's
output has no error flag at OpenAI; an error result says in its text what went wrong.

OpenAI has no fallback model for refusals. A response that holds a refusal, is cut short by
the content filter, or fails for a usage policy (``bio_policy``,
``misalignment_policy_violation``) comes back as a refusal.

Web search is OpenAI's own tool, run by OpenAI. It opens pages as actions of a search call
(``open_page``, ``find_in_page``), not with a tool of their own, and only searches are billed.
The allowed domains go in its filters; the limits on searches and on fetches per model call
become one limit on all its calls (``max_tool_calls``). OpenAI has no limit on the text of
one page, so ``max_fetch_tokens`` is not applied. ``options.web_search_tool`` sets the tool
type for models that need an older one.
"""

import json
from collections.abc import Sequence

import openai

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

_FINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}
_POLICY_FAILURES = {"bio_policy", "misalignment_policy_violation"}
_RETRYABLE_FAILURES = {"server_error", "rate_limit_exceeded"}
_INCOMPLETE = {"max_output_tokens": ("max_tokens", None), "content_filter": ("refusal", "content_filter")}
_FETCH_ACTIONS = {"open_page", "find_in_page"}
_WEB_SEARCH = "web_search"


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str, *, max_retries: int = 4, timeout: float = 600.0,
                 client: openai.OpenAI | None = None) -> None:
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._secrets = [api_key]

    def complete(self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec],
                 settings: ModelSettings, web: WebAccess | None = None) -> ModelReply:
        params: dict = {
            "model": settings.model,
            "max_output_tokens": settings.max_output_tokens,
            "instructions": system,
            "input": [item for message in messages for item in self._to_wire(message)],
            "store": False,
        }
        wire_tools = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.input_schema,
                       "strict": False} for t in tools]
        if web is not None:
            wire_tools.append(self._web_tool(web, settings))
            params["max_tool_calls"] = web.max_searches + web.max_fetches
        if wire_tools:
            params["tools"] = wire_tools
        if settings.effort:
            params["reasoning"] = {"effort": settings.effort}
        try:
            response, output = self._stream(params)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise ModelError(self._clean(f"OpenAI rejected the credentials: {exc}"), retryable=False) from None
        except (openai.BadRequestError, openai.NotFoundError) as exc:
            raise ModelError(self._clean(f"OpenAI rejected the request: {exc}"), retryable=False) from None
        except openai.RateLimitError as exc:
            # A spent quota or spending limit comes as a 429 too, and waiting does not lift it.
            if "insufficient_quota" in (exc.code, exc.type):
                raise ModelError(self._clean(f"OpenAI quota or spending limit reached: {exc}"), retryable=False) from None
            raise ModelError(self._clean(f"OpenAI rate limit: {exc}"), retryable=True) from None
        except openai.APIStatusError as exc:
            raise ModelError(self._clean(f"OpenAI error {exc.status_code}: {exc}"),
                             retryable=exc.status_code >= 500) from None
        except openai.APIConnectionError as exc:
            raise ModelError(self._clean(f"cannot reach OpenAI: {exc}"), retryable=True) from None
        except openai.APIError as exc:
            # An error OpenAI sends after the stream has started has no status; its code says what it is.
            raise ModelError(self._clean(f"OpenAI error partway through the reply: {exc}"),
                             retryable=bool({exc.code, exc.type} & _RETRYABLE_FAILURES)) from None
        return self._from_wire(response, output, settings.model)

    def _stream(self, params: dict) -> tuple:
        """The response the stream ends with (completed, cut off or failed) and its output items, taken from the
        items the stream finished where the final event leaves them out."""
        finished, final = {}, None
        with self._client.responses.create(stream=True, **params) as stream:
            for event in stream:
                if event.type == "response.output_item.done":
                    finished[event.output_index] = event.item
                elif event.type in _FINAL_EVENTS:
                    final = event.response
                elif event.type == "error":
                    raise ModelError(self._clean(f"OpenAI error partway through the reply: {event.code}: "
                                                 f"{event.message}"), retryable=event.code in _RETRYABLE_FAILURES)
        if final is None:
            raise ModelError("the OpenAI stream ended without a response", retryable=True)
        output = getattr(final, "output", None)
        return final, output if output is not None else [finished[i] for i in sorted(finished)]

    def _clean(self, text: str) -> str:
        return redact(text, self._secrets)

    @staticmethod
    def _web_tool(web: WebAccess, settings: ModelSettings) -> dict:
        tool: dict = {"type": settings.options.get("web_search_tool", _WEB_SEARCH)}
        if web.allowed_domains:
            tool["filters"] = {"allowed_domains": list(web.allowed_domains)}
        return tool

    def _to_wire(self, message: Message) -> list[dict]:
        if message.role == "assistant" and message.provider == self.provider and message.native is not None:
            return list(message.native)
        items: list[dict] = []
        # Tool outputs lead a user turn; text follows.
        for part in sorted(message.parts, key=lambda p: not isinstance(p, ToolResult)):
            if isinstance(part, TextPart):
                items.append({"role": message.role, "content": part.text})
            elif isinstance(part, ToolCall):
                items.append({"type": "function_call", "call_id": part.id, "name": part.name,
                              "arguments": json.dumps(part.arguments)})
            else:
                items.append({"type": "function_call_output", "call_id": part.call_id, "output": self._output(part)})
        return items

    @staticmethod
    def _output(result: ToolResult) -> str | list[dict]:
        if not result.media:
            return result.content
        blocks = [{"type": "input_text", "text": result.content}] if result.content else []
        for media in result.media:
            data = f"data:{media.media_type};base64,{media.data}"
            if media.media_type == "application/pdf":
                name = media.title or "document"
                blocks.append({"type": "input_file", "file_data": data,
                               "filename": name if name.lower().endswith(".pdf") else f"{name}.pdf"})
            else:
                blocks.append({"type": "input_image", "image_url": data})
        return blocks

    def _from_wire(self, response, output: list, requested_model: str) -> ModelReply:
        category = None
        if response.status == "failed":
            error = getattr(response, "error", None)
            code = getattr(error, "code", None)
            if code not in _POLICY_FAILURES:
                raise ModelError(self._clean(f"OpenAI could not answer: {code}: {getattr(error, 'message', '')}"),
                                 retryable=code in _RETRYABLE_FAILURES)
            output, category = [], code  # what it had written before failing is not a turn to send back
        parts: list = []
        refused = False
        for item in output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        parts.append(TextPart(text=content.text))
                    refused = refused or content.type == "refusal"
            elif item.type == "function_call":
                try:
                    arguments = json.loads(item.arguments)
                except json.JSONDecodeError:
                    arguments = None
                parts.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments
                                      if isinstance(arguments, dict) else {"__unparsed__": item.arguments}))
        if response.status == "failed":
            stop = "refusal"
        elif response.status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            stop, category = _INCOMPLETE.get(reason, ("other", None))
        elif refused:
            stop = "refusal"
        else:
            stop = "tool_use" if any(isinstance(p, ToolCall) for p in parts) else "end_turn"
        searches = [item for item in output if item.type == "web_search_call"]
        fetches = sum(1 for s in searches if getattr(getattr(s, "action", None), "type", None) in _FETCH_ACTIONS)
        usage = getattr(response, "usage", None)
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        written = getattr(details, "cache_write_tokens", 0) or 0
        return ModelReply(
            message=Message(role="assistant", parts=tuple(parts), provider=self.provider,
                            native=tuple(item.to_dict() for item in output)),
            stop=stop,
            usage=Usage(input_tokens=(usage.input_tokens - cached - written) if usage else 0,
                        output_tokens=usage.output_tokens if usage else 0,
                        cache_read_tokens=cached, cache_write_tokens=written,
                        web_searches=len(searches) - fetches, web_fetches=fetches),
            requested_model=requested_model,
            served_model=response.model,
            refusal_category=category if stop == "refusal" else None,
        )
