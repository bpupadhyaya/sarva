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
            # (e.g. DocumentBlock, which has neither a degrader nor
            # adapter support yet). Raising here is deliberate: silently
            # omitting it would send the request missing content the
            # caller believes is present, and the model would answer as
            # if it had read something it never received -- a materially
            # misleading response, not a cosmetic gap. See
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

        blocks: list[object] = []
        if text_acc:
            blocks.append(TextBlock(text=text_acc))
        for part in tool_call_parts.values():
            try:
                arguments = json.loads(part["arguments"]) if part["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            call = ToolCallBlock(id=part["id"], name=part["name"], arguments=arguments)
            blocks.append(call)
            yield ToolCallEvent(call=call)

        prompt_tokens, completion_tokens, cached_tokens = usage_tokens
        yield DoneEvent(
            stop_reason=_STOP_REASON_MAP.get(finish_reason or "stop", StopReason.END_TURN),
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
