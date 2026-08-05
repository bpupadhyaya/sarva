"""sarva.providers.openai_provider — the OpenAI adapter.

Thin: translates GenerateRequest into the Chat Completions API shape and
translates streamed OpenAI chunk events into ProviderEvent, the same
contract the Anthropic and Ollama adapters already implement (§3.1: "the
heart of newer models keep coming" — one interface, one adapter per
backend, no code anywhere else needs to know which backend is talking).

NOTE: written to the documented `openai` Python SDK's Chat Completions
streaming shape but not yet exercised against a live API key in this
environment — mark its conformance tests `@pytest.mark.live` (skipped
without OPENAI_API_KEY) until a real run validates it, same discipline
as anthropic_provider.py and ollama_provider.py before their first live
runs. See BUILD-JOURNAL.md.

Deliberately NOT adding entries to `providers/data/models.yaml` in this
change: that file's own header states it's "re-validated at every
release," and this project's honesty principle (no fabricated content
anywhere, degraders included) applies just as much to a registry file as
to model output — this session has no verified-current OpenAI model
catalog (IDs, capabilities, per-token pricing) to add responsibly.
Wiring a specific model in is a one-entry config change for whoever has
that data, not a code change (the entire point of the registry design)
- the adapter itself is what needed writing.

Reasoning-effort control (`GenerateConfig.effort`) is also deliberately
left unmapped for now: OpenAI's `reasoning_effort` parameter only
applies to reasoning-capable models and rejects the request on models
that don't support it, unlike Anthropic's `output_config.effort` which
Claude's adaptive-thinking models accept uniformly — mapping it here
without knowing which registry models are reasoning-capable would risk
breaking non-reasoning models with a real 400, not a hypothetical one.
"""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

import openai

from sarva.multimodal.content import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sarva.multimodal.fetch import resolve_media_bytes
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    ProviderEvent,
    StopReason,
    StreamErrorEvent,
    TextDeltaEvent,
    ToolCallEvent,
    Usage,
)

_STOP_REASON_MAP = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
    # A real bug found by a fresh-eyes sweep, the third instance of a
    # bug class already fixed in both sibling adapters (google_
    # provider.py's FinishReason mapping, anthropic_provider.py's
    # pause_turn): the real OpenAI SDK's own finish_reason type
    # (openai.types.chat.chat_completion_chunk.Choice) also includes
    # "function_call" -- the deprecated legacy single-function-calling
    # API's own stop reason, still a real, documented value the SDK's
    # type still carries, not a hypothetical one. This adapter only
    # ever parses `delta.tool_calls` (the modern API), never the
    # legacy `delta.function_call` field, so a response using the old
    # API produces no ToolCallBlock at all -- confirmed live, a
    # duck-typed fake stream yielding finish_reason="function_call"
    # fell through the map's own `.get(..., StopReason.END_TURN)`
    # default straight into the success path, reporting what was
    # actually an unparsed function call as a normal, complete answer.
    # Mapped to REFUSAL rather than silently claiming success, matching
    # the honest treatment `AgentLoop` already gives every other
    # "not a genuine complete answer" case.
    "function_call": StopReason.REFUSAL,
}


async def _to_openai_messages(m: Message) -> list[dict[str, Any]]:
    """One Sarva `Message` can become *several* OpenAI messages: unlike
    Anthropic, which lets multiple tool_result blocks live inside one
    role="user" content array, OpenAI requires one dedicated
    role="tool" message per tool_call_id. Assistant text and tool_calls
    stay combined in a single message (OpenAI's own shape for that case)."""
    content_parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []

    for b in m.content:
        if isinstance(b, TextBlock):
            content_parts.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            # resolve_media_bytes (not b.resolve_bytes()) so url-sourced
            # images work too, not just data/path.
            image_bytes = await resolve_media_bytes(b)
            b64 = base64.standard_b64encode(image_bytes).decode()
            data_url = f"data:{b.media_type};base64,{b64}"
            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        elif isinstance(b, ToolCallBlock):
            tool_calls.append(
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.arguments)},
                }
            )
        elif isinstance(b, ToolResultBlock):
            # A real bug found by actually constructing a ToolResultBlock
            # carrying an ImageBlock (e.g. a screenshot/image-generation
            # tool's result -- ToolResultBlock.content's own type comment
            # already names this as an anticipated shape, not a
            # hypothetical): the plain `"".join(... TextBlock)` here
            # silently dropped it with no error. Unlike Anthropic/Gemini,
            # this genuinely can't just be fixed by sending the image
            # along instead -- OpenAI's own SDK type
            # (`ChatCompletionToolMessageParam.content`) only accepts
            # `str | Iterable[ChatCompletionContentPartTextParam]`, text
            # parts only, confirmed by reading it directly. Silently
            # dropping non-text content would still send Claude/GPT a
            # tool result missing content the caller believes is
            # present -- the same "materially misleading response" the
            # `else: raise` a few lines below this exists to prevent for
            # top-level blocks -- so this raises instead of guessing at a
            # wire shape OpenAI doesn't actually support. Text-only
            # results (the overwhelming common case) are completely
            # unaffected.
            for c in b.content:
                if not isinstance(c, TextBlock):
                    raise ValueError(
                        f"OpenAIProvider cannot translate a {type(c).__name__!r} content "
                        "block inside a tool result (OpenAI's tool-message content field "
                        "only accepts text; no wire-format mapping exists for anything else)"
                    )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": b.tool_call_id,
                    "content": "".join(c.text for c in b.content if isinstance(c, TextBlock)),
                }
            )
        elif isinstance(b, ThinkingBlock):
            # Deliberately, explicitly dropped -- not silently: OpenAI's
            # reasoning models don't accept a caller-supplied reasoning
            # trace back on the next turn the way this block would imply,
            # so there's nothing meaningful to round-trip yet. Explicit
            # here so it's a named, intentional skip rather than an
            # unhandled type quietly falling through with no case at all.
            continue
        else:
            # A block type this adapter has no translation for at all
            # (e.g. DocumentBlock reaching this adapter directly,
            # unconverted -- it has a degrader now, but only
            # degrade_message()'s opt-in fallback path uses it; a caller
            # that skips degradation, or a model whose registry entry
            # claims document support it doesn't actually have wire-level
            # code for, still reaches here). Raising here is deliberate:
            # silently omitting it would send the request missing content
            # the caller believes is present, and the model would answer
            # as if it had read something it never received -- a
            # materially misleading response, not a cosmetic gap. See
            # docs/multimodal.md for the fuller story.
            raise ValueError(
                f"OpenAIProvider cannot translate a {type(b).__name__!r} content block "
                "(no wire-format mapping exists for it yet)"
            )

    messages: list[dict[str, Any]] = []
    if content_parts or tool_calls:
        main: dict[str, Any] = {"role": m.role, "content": content_parts or None}
        if tool_calls:
            main["tool_calls"] = tool_calls
        messages.append(main)
    messages.extend(tool_messages)
    return messages


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: openai.AsyncOpenAI | None = None):
        self._client = client or openai.AsyncOpenAI()

    async def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for m in request.messages:
            messages.extend(await _to_openai_messages(m))

        kwargs: dict[str, Any] = dict(
            model=request.model,
            messages=messages,
            max_completion_tokens=request.config.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]

        text_acc = ""
        # OpenAI streams tool-call arguments incrementally, keyed by the
        # chunk's `index` -- accumulate id/name/arguments-so-far per index
        # rather than assuming any single chunk carries a complete call.
        tool_call_parts: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"id": "", "name": "", "arguments": ""}
        )
        # A real bug found by a fresh-eyes sweep, the identical gap
        # already found and fixed in google_provider.py/ollama_provider.py,
        # just never propagated here: without this, text was accumulated
        # into ONE running string across the whole stream and only ever
        # spliced into `blocks` once, unconditionally first, after the
        # loop ended -- tool calls were already assembled in true
        # chronological (first-seen) order relative to each other (Python
        # dict insertion order on `tool_call_parts`), but any text
        # occurring between or after them got silently pulled forward and
        # merged ahead of every tool call. Confirmed live: an ordinary
        # sequential-tool-calling turn (reasoning text, a call, more
        # reasoning text, another call -- ordinary ReAct-style behavior,
        # not contrived) produced one TextBlock with both segments
        # concatenated, hoisted ahead of both tool calls, corrupting the
        # persisted Message AgentLoop appends straight to transcript_out/
        # SessionStore and resends as history. Genuinely trickier to fix
        # here than in the sibling adapters: OpenAI streams a tool call's
        # arguments incrementally across MANY chunks sharing one `index`,
        # so a call isn't complete (and can't become a real ToolCallBlock)
        # until the stream ends -- there's no single "this chunk completed
        # a call" moment to flush against the way Ollama/Gemini's
        # atomically-complete-per-chunk tool calls have. Fixed instead by
        # recording an ORDERED list of markers as things first appear --
        # a flushed text segment, or the first-seen index of a tool call
        # (its actual ToolCallBlock is still only built once, at the end,
        # from `tool_call_parts`) -- and replaying that order when
        # assembling `blocks`, rather than reconstructing position from
        # `tool_call_parts` and a single trailing text blob.
        order: list[tuple[str, str | int]] = []
        seen_indices: set[int] = set()
        finish_reason: str | None = None
        usage_tokens = (0, 0, 0)

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.usage:
                    usage_tokens = (
                        chunk.usage.prompt_tokens,
                        chunk.usage.completion_tokens,
                        (chunk.usage.prompt_tokens_details.cached_tokens or 0)
                        if chunk.usage.prompt_tokens_details
                        else 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    text_acc += delta.content
                    yield TextDeltaEvent(text=delta.content)
                for tc in delta.tool_calls or []:
                    if tc.index not in seen_indices:
                        seen_indices.add(tc.index)
                        if text_acc:
                            order.append(("text", text_acc))
                            text_acc = ""
                        order.append(("tool_call", tc.index))
                    part = tool_call_parts[tc.index]
                    if tc.id:
                        part["id"] = tc.id
                    if tc.function and tc.function.name:
                        part["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        part["arguments"] += tc.function.arguments
        except openai.RateLimitError as e:
            yield StreamErrorEvent(code="rate_limit", detail=str(e), retryable=True)
            return
        except openai.APIConnectionError as e:
            yield StreamErrorEvent(code="network", detail=str(e), retryable=True)
            return
        except openai.APIStatusError as e:
            yield StreamErrorEvent(code="provider", detail=str(e), retryable=e.status_code >= 500)
            return
        except openai.APIError as e:
            # A real bug found by reading the SDK's own exception
            # hierarchy, the same way as the identical gap fixed in
            # anthropic_provider.py: openai.APIResponseValidationError
            # (raised when the SDK can't parse a malformed response) is a
            # direct sibling of RateLimitError/APIConnectionError/
            # APIStatusError under the SDK's own APIError base, not a
            # subclass of any of the three above, so it propagated
            # uncaught. Catching the SDK's own common base here, after the
            # three specific/more-informative handlers already had their
            # chance, closes this for APIResponseValidationError
            # specifically and for any other still-unnamed APIError
            # subtype the SDK might add in the future.
            yield StreamErrorEvent(code="provider", detail=str(e), retryable=True)
            return

        # Trailing text (the ordinary case: a plain END_TURN reply, or
        # reasoning text after the last tool call) belongs after every
        # block that chronologically preceded it -- appended to `order`
        # here, not spliced onto the front of `blocks` below.
        if text_acc:
            order.append(("text", text_acc))
            text_acc = ""

        blocks: list[object] = []
        for kind, value in order:
            if kind == "text":
                blocks.append(TextBlock(text=value))
                continue
            part = tool_call_parts[value]
            try:
                arguments = json.loads(part["arguments"]) if part["arguments"] else {}
            except json.JSONDecodeError as e:
                # A real bug found by actually feeding a stream malformed
                # tool-call-arguments JSON (a real, documented GPT
                # tool-calling failure mode, independent of max_tokens
                # truncation -- confirmed live with a genuinely
                # malformed accumulated string, unescaped embedded
                # quotes): silently substituting {} here let AgentLoop
                # dispatch a corrupted, empty arguments dict to whatever
                # tool the model actually meant to call with real
                # arguments -- no StreamErrorEvent, no signal anywhere.
                # A built-in tool's required-key access then raises a
                # confusing bare KeyError the model has no way to
                # connect back to its own dropped arguments; an MCP tool
                # (McpToolAdapter.run) forwards {} straight to the
                # remote server with zero local validation, silently
                # executing the WRONG action for any tool with optional/
                # defaulted parameters -- genuinely undetectable
                # corruption, not just a confusing message. Every other
                # failure path in this function already signals a real
                # problem via StreamErrorEvent (retryable=True re-calls
                # the model fresh, the same recovery a rate limit or a
                # network blip already gets) -- this was the one
                # instance that silently pressed on instead.
                yield StreamErrorEvent(
                    code="malformed_tool_arguments",
                    detail=f"tool call {part['name']!r} arguments were not valid JSON: {e}",
                    retryable=True,
                )
                return
            call = ToolCallBlock(id=part["id"], name=part["name"], arguments=arguments)
            blocks.append(call)
            yield ToolCallEvent(call=call)

        prompt_tokens, completion_tokens, cached_tokens = usage_tokens
        yield DoneEvent(
            # Default REFUSAL, not END_TURN -- see _STOP_REASON_MAP's
            # own comment on "function_call". "stop" is the only value
            # that legitimately means success, and it's already
            # explicitly mapped (also covering the `finish_reason is
            # None` case via the `or "stop"` above, unchanged from
            # before), so any other value, known today or added by a
            # future SDK release, now fails safe instead of silently
            # claiming a complete, successful answer.
            stop_reason=_STOP_REASON_MAP.get(finish_reason or "stop", StopReason.REFUSAL),
            message=Message(role="assistant", content=blocks),
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cache_read_tokens=cached_tokens,
                # Real per-token pricing needs a verified-current entry in
                # models.yaml (see module docstring) -- reporting cost_usd=0
                # here rather than a guessed/fabricated number.
                cost_usd=0.0,
            ),
        )

    async def close(self) -> None:
        await self._client.close()
