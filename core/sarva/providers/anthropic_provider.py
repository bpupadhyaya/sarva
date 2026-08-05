"""sarva.providers.anthropic_provider — the Anthropic adapter.

Thin: translates GenerateRequest into the Messages API shape and translates
streamed Anthropic events into ProviderEvent. Uses adaptive thinking by
default (matching current Claude 4.6+ model guidance) and the `effort`
parameter for cost/quality control.

NOTE: this adapter is written to the documented Anthropic Python SDK
streaming pattern but has not yet been exercised against a live API key in
this environment — mark its conformance tests `@pytest.mark.live` (skipped
without ANTHROPIC_API_KEY) until a real run validates it. See BUILD-JOURNAL.md.

Round-trips extended-thinking blocks back to the API when a real
signature is present (Anthropic's anti-tampering check on reused
thinking history) — `generate()` below stores the SDK's own signature
in `ThinkingBlock.provider_data` the moment one is produced, and
`AgentLoop` already threads that exact `Message` into the next turn's
history unmodified, so no further plumbing was needed to make this
work once `provider_data` existed on the block itself. A block with no
signature (never passed through this adapter) still drops rather than
sending a fabricated one Anthropic would reject.
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
from sarva.multimodal.fetch import resolve_media_bytes
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
    # A real bug found by a fresh-eyes sweep, the identical bug class
    # already fixed once in google_provider.py's FinishReason mapping,
    # never propagated to this sibling adapter: the real Anthropic SDK's
    # `StopReason` type also includes `"pause_turn"` -- a real,
    # documented, non-error state ("we paused a long-running turn. You
    # may provide the response back as-is in a subsequent request to
    # let the model continue," per the SDK's own docstring, returned for
    # long-running server-side tool use such as web search/code
    # execution), not a hypothetical value. Left unmapped, it fell
    # through the lookup's own `.get(..., StopReason.END_TURN)` default
    # straight into the *success* path -- confirmed live, a real
    # `pause_turn` response was reported to AgentLoop/CLI/server as a
    # normal, complete answer, silently dropping the unfinished
    # long-running turn instead of surfacing it as incomplete. This
    # project has no turn-resumption mechanism wired in yet, so there's
    # no accurate "paused, resume me" StopReason to map to -- REFUSAL is
    # the closest honest fit among the ones that exist: it's not a
    # successful, complete answer, and `AgentLoop` already treats it as
    # a clean FAILED rather than crashing or silently succeeding, the
    # same "fail safe, never silently succeed" choice the Google fix
    # made for its own unmapped reasons.
    "pause_turn": StopReason.REFUSAL,
}

# 2026-07 pricing snapshot (USD / 1M tokens) — keep in sync with providers/data/models.yaml
_PRICE = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


async def _to_anthropic_message(m: Message) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            blocks.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            # resolve_media_bytes (not b.resolve_bytes()) so url-sourced
            # images work too, not just data/path — see
            # sarva.multimodal.fetch's module docstring.
            image_bytes = await resolve_media_bytes(b)
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": b.media_type,
                        "data": base64.standard_b64encode(image_bytes).decode(),
                    },
                }
            )
        elif isinstance(b, ToolCallBlock):
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments})
        elif isinstance(b, ToolResultBlock):
            # A real bug found by actually constructing a ToolResultBlock
            # carrying an ImageBlock (e.g. a screenshot/image-generation
            # tool's result -- ToolResultBlock.content's own type comment
            # already names this as an anticipated shape, not a
            # hypothetical): the plain `"".join(... TextBlock)` here
            # silently dropped it with no error, sending Claude a tool
            # result missing content the caller believes is present --
            # the exact "materially misleading response" the `else:
            # raise` a few lines below this exists to prevent for
            # top-level blocks, just never applied one level down inside
            # a tool result. Anthropic's own SDK type
            # (`ToolResultBlockParam.content`) genuinely accepts a list
            # mixing text and image blocks, confirmed by reading it
            # directly, not assumed -- so this is a real capability gap,
            # not a wire-format limitation. Text-only results (the
            # overwhelming common case) still send a plain string exactly
            # as before, so no existing behavior changes for them.
            tool_result_content: str | list[dict[str, Any]]
            if all(isinstance(c, TextBlock) for c in b.content):
                tool_result_content = "".join(c.text for c in b.content if isinstance(c, TextBlock))
            else:
                parts: list[dict[str, Any]] = []
                for c in b.content:
                    if isinstance(c, TextBlock):
                        parts.append({"type": "text", "text": c.text})
                    elif isinstance(c, ImageBlock):
                        c_bytes = await resolve_media_bytes(c)
                        parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": c.media_type,
                                    "data": base64.standard_b64encode(c_bytes).decode(),
                                },
                            }
                        )
                    else:
                        raise ValueError(
                            f"AnthropicProvider cannot translate a {type(c).__name__!r} "
                            "content block inside a tool result (no wire-format mapping "
                            "exists for it yet)"
                        )
                tool_result_content = parts
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.tool_call_id,
                    "content": tool_result_content,
                    "is_error": b.is_error,
                }
            )
        elif isinstance(b, ThinkingBlock):
            # Round-tripped when possible, not unconditionally dropped:
            # Anthropic requires the ORIGINAL signature back (an
            # anti-tampering check) to accept a thinking block as
            # genuine history rather than reject the turn -- generate()
            # below already stores it in provider_data the moment a
            # ThinkingBlock is produced, and AgentLoop already threads
            # the same Message object straight into the next turn's
            # history (`messages.append(done.message)` in agent/loop.py)
            # with nothing stripped along the way, so no further
            # plumbing was actually needed once provider_data existed on
            # the block itself. A block with no signature (never passed
            # through this adapter -- e.g. a hand-built session, or one
            # predating this field) can't be reconstructed safely, so it
            # still drops exactly as before rather than sending a
            # fabricated one Anthropic would reject anyway.
            #
            # A redacted_thinking block (see generate()'s own comment on
            # the parsing side) round-trips the same way, just under a
            # different wire shape: the opaque encrypted blob goes back
            # as `data` on a `redacted_thinking` block, never as
            # `thinking` text (there is none -- redaction's whole point)
            # or a `signature` (redacted blocks don't carry one).
            redacted_data = (b.provider_data or {}).get("redacted_data")
            if redacted_data is not None:
                blocks.append({"type": "redacted_thinking", "data": redacted_data})
            else:
                signature = (b.provider_data or {}).get("signature")
                if signature:
                    blocks.append({"type": "thinking", "thinking": b.text, "signature": signature})
        else:
            # A block type this adapter has no translation for at all
            # (e.g. DocumentBlock reaching this adapter directly,
            # unconverted -- it has a degrader now, but only
            # degrade_message()'s opt-in fallback path uses it; a caller
            # that skips degradation, or a model whose registry entry
            # claims document support it doesn't actually have wire-level
            # code for, still reaches here). Raising here is deliberate:
            # silently omitting it would send the request to Claude
            # missing content the caller believes is present, and the
            # model would answer as if it had read something it never
            # received -- a materially misleading response, not a
            # cosmetic gap. See docs/multimodal.md for the fuller story.
            raise ValueError(
                f"AnthropicProvider cannot translate a {type(b).__name__!r} content block "
                "(no wire-format mapping exists for it yet)"
            )
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
        messages = [await _to_anthropic_message(m) for m in request.messages]
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
        except anthropic.APIError as e:
            # A real bug found by actually raising APIResponseValidationError
            # through a duck-typed fake client: it's a direct sibling of
            # APIStatusError/APIConnectionError under the SDK's own APIError
            # base (not a subclass of either), raised when the SDK can't
            # parse a malformed response body -- the same "server sent
            # something we can't make sense of" shape as the Ollama
            # streaming-JSON bug fixed elsewhere in this project -- and it
            # propagated uncaught since none of the three specific handlers
            # above cover it. Catching the SDK's own common base here, after
            # the three specific/more-informative handlers already had their
            # chance, closes this for APIResponseValidationError specifically
            # and for any other still-unnamed APIError subtype the SDK might
            # add in the future.
            yield StreamErrorEvent(code="provider", detail=str(e), retryable=True)
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
            elif b.type == "redacted_thinking":
                # A real bug found by giving this adapter's thinking
                # round-trip its own fresh-eyes sweep, one layer deeper
                # than the plain `thinking` case it was already fixed
                # for: Anthropic's real SDK also defines
                # `RedactedThinkingBlock` (`type="redacted_thinking"`,
                # `data: str`), returned whenever the model's own safety
                # classifier flags part of its reasoning -- a normal,
                # documented, non-adversarial occurrence with adaptive
                # thinking on (this adapter's own default), not a rare
                # edge case. With no branch for it here, it matched none
                # of the `if`/`elif`s above and was silently dropped --
                # never surfaced as a ThinkingBlock, never yielded.
                # Confirmed live: when a redacted_thinking block preceded
                # a tool_use block (the ordinary "reason, then call a
                # tool" shape), the resulting Message threaded into the
                # next turn's history started with `tool_use` alone --
                # Anthropic's extended-thinking + tool-use contract
                # requires the leading thinking/redacted_thinking block
                # be replayed verbatim on the follow-up request, so the
                # real API would reject that turn. Represented as a
                # `ThinkingBlock` with empty visible text (redaction's
                # whole point is no visible text) and the opaque
                # encrypted blob preserved in `provider_data` under a
                # distinct key from the signed case's `signature`, so
                # `_to_anthropic_message` below can tell the two apart
                # and reconstruct the correct wire shape for each.
                blocks.append(ThinkingBlock(text="", provider_data={"redacted_data": b.data}))
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
            # Default REFUSAL, not END_TURN -- see _STOP_REASON_MAP's
            # own comment on `pause_turn`. Every value that legitimately
            # means success is already explicitly mapped above; any
            # other value, known today or added by a future SDK
            # release, now fails safe instead of silently claiming a
            # complete, successful answer.
            stop_reason=_STOP_REASON_MAP.get(final.stop_reason, StopReason.REFUSAL),
            message=Message(role="assistant", content=blocks),
            usage=usage,
        )

    async def close(self) -> None:
        await self._client.close()
