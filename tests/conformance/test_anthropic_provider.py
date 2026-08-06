"""Unit tests for the Anthropic adapter's translation function, plus
`generate()`'s own streaming-error handling (below, via a duck-typed
fake client -- the same discipline test_openai_provider_streaming.py
already established for SDK-based adapters, since this environment has
no real Anthropic API key to test the streaming path live).

No network, no API key — every block here carries an in-memory `data`
source, so `_to_anthropic_message`'s only await (`resolve_media_bytes`,
for url-sourced images) never actually runs. It's `async def` now purely
because that's what letting url sources resolve via
`sarva.multimodal.fetch` requires (see that module) — not because this
test exercises any I/O. Live end-to-end behavior is covered by
tests/live/test_live_providers.py.
"""

from __future__ import annotations

import base64

import anthropic
import httpx
import pytest
from sarva.multimodal.content import (
    DocumentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sarva.providers.anthropic_provider import AnthropicProvider, _to_anthropic_message
from sarva.providers.base import (
    DoneEvent,
    GenerateConfig,
    GenerateRequest,
    StopReason,
    StreamErrorEvent,
)


async def test_text_block_translation():
    m = Message(role="user", content=[TextBlock(text="hello")])
    out = await _to_anthropic_message(m)
    assert out == {"role": "user", "content": [{"type": "text", "text": "hello"}]}


async def test_image_block_translation_base64_round_trips():
    raw = b"\x89PNG\r\n\x1a\n"
    m = Message(role="user", content=[ImageBlock(media_type="image/png", data=raw)])
    out = await _to_anthropic_message(m)

    block = out["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(block["source"]["data"]) == raw


async def test_tool_call_and_result_translation():
    call = ToolCallBlock(id="t1", name="get_weather", arguments={"city": "Paris"})
    m1 = Message(role="assistant", content=[call])
    out1 = await _to_anthropic_message(m1)
    assert out1["content"][0] == {
        "type": "tool_use",
        "id": "t1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }

    result = ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="sunny")])
    m2 = Message(role="user", content=[result])
    out2 = await _to_anthropic_message(m2)
    assert out2["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "sunny",
        "is_error": False,
    }


async def test_tool_result_with_an_image_sends_it_instead_of_silently_dropping_it():
    # A real bug found by actually constructing a ToolResultBlock
    # carrying an ImageBlock (e.g. a screenshot tool's result --
    # ToolResultBlock.content's own type comment already names this as
    # an anticipated shape): the plain `"".join(... TextBlock)` this
    # adapter used to build tool_result content silently dropped
    # anything that wasn't a TextBlock, with no error. Anthropic's own
    # SDK type genuinely accepts a list mixing text and image blocks
    # inside a tool result, confirmed by reading it directly.
    raw = b"\x89PNG\r\n\x1a\n"
    result = ToolResultBlock(
        tool_call_id="t1",
        content=[
            TextBlock(text="here's the screenshot:"),
            ImageBlock(media_type="image/png", data=raw),
        ],
    )
    m = Message(role="user", content=[result])

    out = await _to_anthropic_message(m)

    tool_result = out["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"][0] == {"type": "text", "text": "here's the screenshot:"}
    image_part = tool_result["content"][1]
    assert image_part["type"] == "image"
    assert base64.standard_b64decode(image_part["source"]["data"]) == raw


async def test_thinking_block_with_a_signature_round_trips_to_the_wire_shape():
    # Anthropic requires the ORIGINAL signature back to accept a
    # thinking block as genuine history (an anti-tampering check) --
    # generate() already stores it in provider_data the moment a
    # ThinkingBlock is produced (see the "signature" key below), and
    # AgentLoop already threads the same Message straight into the next
    # turn's history unmodified, so a real signature reaching here is
    # the normal case for any thinking turn this adapter itself produced.
    m = Message(
        role="assistant",
        content=[
            ThinkingBlock(text="pondering", provider_data={"signature": "sig-abc123"}),
            TextBlock(text="hi"),
        ],
    )
    out = await _to_anthropic_message(m)
    assert out["content"] == [
        {"type": "thinking", "thinking": "pondering", "signature": "sig-abc123"},
        {"type": "text", "text": "hi"},
    ]


async def test_thinking_block_with_no_signature_is_dropped_not_fabricated():
    # A ThinkingBlock with no signature (never passed through this
    # adapter -- e.g. hand-built, or from before this field existed)
    # can't be reconstructed safely: sending a fabricated signature
    # would be rejected by Anthropic anyway, so this stays a named,
    # explicit skip rather than an invented value.
    m = Message(role="assistant", content=[ThinkingBlock(text="pondering"), TextBlock(text="hi")])
    out = await _to_anthropic_message(m)
    assert out["content"] == [{"type": "text", "text": "hi"}]


async def test_unsupported_block_type_raises_instead_of_silently_dropping():
    # DocumentBlock has no wire-format mapping in this adapter yet.
    # Silently omitting it would send the request missing content the
    # caller believes is present -- must raise loudly instead.
    m = Message(
        role="user",
        content=[
            TextBlock(text="see attached"),
            DocumentBlock(media_type="application/pdf", data=b"x"),
        ],
    )
    with pytest.raises(ValueError, match="DocumentBlock"):
        await _to_anthropic_message(m)


class _RaisingStreamContext:
    """A duck-typed stand-in for `client.messages.stream(**kwargs)`'s
    real async context manager, raising a chosen exception from
    `__aenter__` -- exactly where the real SDK raises one of its own
    `APIError` subtypes when a request fails before streaming even
    starts."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *args):
        return False


class _FakeMessages:
    def __init__(self, exc: Exception):
        self._exc = exc

    def stream(self, **kwargs):
        return _RaisingStreamContext(self._exc)


class _FakeClient:
    def __init__(self, exc: Exception):
        self.messages = _FakeMessages(exc)


def _req() -> GenerateRequest:
    return GenerateRequest(
        model="claude-x", messages=[Message(role="user", content=[TextBlock(text="hi")])]
    )


def _real_response() -> httpx.Response:
    return httpx.Response(
        200, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


async def test_generate_yields_a_clean_stream_error_on_a_malformed_sdk_response():
    # A real bug found by actually raising anthropic.APIResponseValidationError
    # through this fake client: it's a direct sibling of
    # APIStatusError/APIConnectionError under the SDK's own APIError base
    # (not a subclass of either), raised when the SDK can't parse a
    # malformed response body -- the same "server sent something we
    # can't make sense of" shape as the Ollama streaming-JSON bug fixed
    # elsewhere in this project. None of this adapter's three specific
    # except clauses (RateLimitError/APIConnectionError/APIStatusError)
    # covered it, so it propagated straight out of generate()'s async
    # generator uncaught before this fix.
    exc = anthropic.APIResponseValidationError(
        response=_real_response(), body=None, message="could not parse response"
    )
    provider = AnthropicProvider(client=_FakeClient(exc))

    events = [e async for e in provider.generate(_req())]

    assert len(events) == 1
    assert isinstance(events[0], StreamErrorEvent)
    assert events[0].code == "provider"
    assert "could not parse response" in events[0].detail


async def test_generate_still_handles_a_plain_rate_limit_error_as_before():
    # Regression guard: the new broad `except anthropic.APIError` catch-all
    # sits AFTER the three specific handlers, so it must never intercept
    # what RateLimitError's own more-specific handler already covers.
    exc = anthropic.RateLimitError("rate limited", response=_real_response(), body=None)
    provider = AnthropicProvider(client=_FakeClient(exc))

    events = [e async for e in provider.generate(_req())]

    assert len(events) == 1
    assert events[0].code == "rate_limit"
    assert events[0].retryable is True


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeFinalMessage:
    def __init__(self, stop_reason: str, content=None):
        self.content = content or []
        self.usage = _FakeUsage()
        self.stop_reason = stop_reason


class _SucceedingStreamContext:
    """Duck-typed stand-in for client.messages.stream(**kwargs)'s async
    context manager where the stream succeeds and produces a real final
    message -- unlike _RaisingStreamContext above, which only covers the
    error path."""

    def __init__(self, final):
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        async def _gen():
            return
            yield  # pragma: no cover

        return _gen()

    async def get_final_message(self):
        return self._final


class _FakeMessagesWithFinal:
    def __init__(self, final):
        self._final = final

    def stream(self, **kwargs):
        return _SucceedingStreamContext(self._final)


class _FakeClientWithFinal:
    def __init__(self, final):
        self.messages = _FakeMessagesWithFinal(final)


async def test_a_paused_long_running_turn_is_not_reported_as_a_silent_success():
    # A real bug found by a fresh-eyes sweep, the identical bug class
    # already fixed once in google_provider.py's FinishReason mapping,
    # never propagated to this sibling adapter: the real Anthropic SDK's
    # StopReason type also includes "pause_turn" (see anthropic.types.
    # stop_reason.StopReason) -- a real, documented, non-error state
    # returned for a long-running server-side tool call (web search/code
    # execution) that must be resumed with a follow-up request, not
    # treated as a complete answer. Left unmapped, it fell through
    # _STOP_REASON_MAP's own `.get(..., StopReason.END_TURN)` default
    # straight into the success path -- confirmed live before this fix.
    final = _FakeFinalMessage(stop_reason="pause_turn")
    provider = AnthropicProvider(client=_FakeClientWithFinal(final))

    events = [e async for e in provider.generate(_req())]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason != StopReason.END_TURN
    assert done.stop_reason == StopReason.REFUSAL


async def test_an_unrecognized_stop_reason_fails_safe_as_refusal_not_end_turn():
    # The other half of the same fix: the map's own default was changed
    # from StopReason.END_TURN to StopReason.REFUSAL, so any FUTURE
    # stop_reason value this map hasn't explicitly named yet also fails
    # safe -- a clean REFUSAL a caller can see, never a silently
    # "successful" response just because the exact value wasn't
    # enumerated.
    final = _FakeFinalMessage(stop_reason="some_future_reason")
    provider = AnthropicProvider(client=_FakeClientWithFinal(final))

    events = [e async for e in provider.generate(_req())]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.REFUSAL


class _FakeMessagesCapturingKwargs:
    def __init__(self, final):
        self._final = final
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _SucceedingStreamContext(self._final)


class _FakeClientCapturingKwargs:
    def __init__(self, final):
        self.messages = _FakeMessagesCapturingKwargs(final)


async def test_generate_translates_stop_sequences_to_the_real_sdk_kwarg():
    # A real bug found by a fresh-eyes sweep, applying the "one sibling
    # has the feature, others silently diverge" lens that already caught
    # round 149's DPO-schedule bug: GenerateConfig.stop_sequences is a
    # shared, provider-agnostic field every adapter is supposed to honor
    # (docs/providers.md names it as part of the contract), but only
    # GoogleProvider ever actually translated it -- this adapter (and
    # OpenAI's, and Ollama's) silently dropped it, with no exception or
    # warning, even though the real Anthropic SDK genuinely supports
    # `stop_sequences` as a real messages.stream() kwarg. Confirmed live
    # before this fix: a request with stop_sequences=["STOP_HERE"] sent
    # no stop_sequences kwarg at all.
    final = _FakeFinalMessage(stop_reason="end_turn")
    client = _FakeClientCapturingKwargs(final)
    provider = AnthropicProvider(client=client)
    request = GenerateRequest(
        model="claude-x",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        config=GenerateConfig(stop_sequences=["STOP_HERE"]),
    )

    [e async for e in provider.generate(request)]

    assert client.messages.calls[0]["stop_sequences"] == ["STOP_HERE"]
