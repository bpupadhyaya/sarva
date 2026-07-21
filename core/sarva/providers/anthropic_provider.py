"""sarva.providers.anthropic_provider — the Anthropic adapter.

Thin: translates GenerateRequest into the Messages API shape and translates
streamed Anthropic events into ProviderEvent. Uses adaptive thinking by
default (matching current Claude 4.6+ model guidance) and the `effort`
parameter for cost/quality control.

NOTE: this adapter is written to the documented Anthropic Python SDK
streaming pattern but has not yet been exercised against a live API key in
this environment — mark its conformance tests `@pytest.mark.live` (skipped
without ANTHROPIC_API_KEY) until a real run validates it. See BUILD-JOURNAL.md.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from sarva.multimodal.content import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    ProviderEvent,
    StopReason,
    StreamErrorEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    Usage,
)

_STOP_REASON_MAP = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "stop_sequence": StopReason.END_TURN,
}

# 2026-07 pricing snapshot (USD / 1M tokens) — keep in sync with providers/data/models.yaml
_PRICE = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _to_anthropic_message(m: Message) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            blocks.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": b.media_type,
                        "data": base64.standard_b64encode(b.resolve_bytes()).decode(),
                    },
                }
            )
        elif isinstance(b, ToolCallBlock):
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments})
        elif isinstance(b, ToolResultBlock):
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.tool_call_id,
                    "content": "".join(c.text for c in b.content if isinstance(c, TextBlock)),
                    "is_error": b.is_error,
                }
            )
        # ThinkingBlock round-trip (required when continuing on the same
        # model) lands when the agent loop starts threading provider_data
        # back through GenerateRequest — tracked for T2.
    return {"role": m.role, "content": blocks}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: anthropic.AsyncAnthropic | None = None):
        self._client = client or anthropic.AsyncAnthropic()

    async def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]:
        tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in request.tools
        ]
        messages = [_to_anthropic_message(m) for m in request.messages]
        kwargs: dict[str, Any] = dict(
            model=request.model,
            max_tokens=request.config.max_tokens,
            messages=messages,
        )
        if request.system:
            kwargs["system"] = request.system
        if tools:
            kwargs["tools"] = tools
        if request.config.thinking is not False:
            kwargs["thinking"] = {"type": "adaptive"}
        if request.config.effort:
            kwargs["output_config"] = {"effort": request.config.effort}

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield TextDeltaEvent(text=event.delta.text)
                        elif event.delta.type == "thinking_delta":
                            yield ThinkingDeltaEvent(text=event.delta.thinking)
                final = await stream.get_final_message()
        except anthropic.RateLimitError as e:
            yield StreamErrorEvent(code="rate_limit", detail=str(e), retryable=True)
            return
        except anthropic.APIConnectionError as e:
            yield StreamErrorEvent(code="network", detail=str(e), retryable=True)
            return
        except anthropic.APIStatusError as e:
            yield StreamErrorEvent(code="provider", detail=str(e), retryable=e.status_code >= 500)
            return

        blocks: list[object] = []
        for b in final.content:
            if b.type == "text":
                blocks.append(TextBlock(text=b.text))
            elif b.type == "thinking":
                blocks.append(
                    ThinkingBlock(
                        text=b.thinking,
                        provider_data={"signature": getattr(b, "signature", None)},
                    )
                )
            elif b.type == "tool_use":
                call = ToolCallBlock(id=b.id, name=b.name, arguments=b.input)
                blocks.append(call)
                yield ToolCallEvent(call=call)

        in_price, out_price = _PRICE.get(request.model, (0.0, 0.0))
        usage = Usage(
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
            cost_usd=(final.usage.input_tokens * in_price + final.usage.output_tokens * out_price)
            / 1_000_000,
        )
        yield DoneEvent(
            stop_reason=_STOP_REASON_MAP.get(final.stop_reason, StopReason.END_TURN),
            message=Message(role="assistant", content=blocks),
            usage=usage,
        )

    async def close(self) -> None:
        await self._client.close()
