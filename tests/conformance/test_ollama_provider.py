"""Unit tests for the Ollama adapter's translation function --
`_to_ollama_message` had zero conformance coverage until now (only
exercised by `tests/live/test_live_providers.py`, skipped without a
real running server), unlike every other adapter's own dedicated
translation-unit test file. No network, no server needed here.
"""

from __future__ import annotations

import base64

import httpx
import pytest
from sarva.multimodal.content import (
    DocumentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    StopReason,
    StreamErrorEvent,
    TextDeltaEvent,
)
from sarva.providers.ollama_provider import OllamaProvider, _strip_local_prefix, _to_ollama_message


async def test_text_block_translation():
    m = Message(role="user", content=[TextBlock(text="hello")])
    out = await _to_ollama_message(m)
    assert out == {"role": "user", "content": "hello"}


async def test_tool_call_translation():
    call = ToolCallBlock(id="t1", name="get_weather", arguments={"city": "Paris"})
    m = Message(role="assistant", content=[TextBlock(text="checking..."), call])
    out = await _to_ollama_message(m)
    assert out["role"] == "assistant"
    assert out["content"] == "checking..."
    assert out["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]


async def test_tool_result_renders_as_plain_text_content():
    # Ollama has no dedicated tool-result role -- the text just becomes
    # this message's own content.
    m = Message(
        role="user",
        content=[ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="sunny, 22C")])],
    )
    out = await _to_ollama_message(m)
    assert out == {"role": "user", "content": "sunny, 22C"}


async def test_multiple_tool_results_do_not_fuse_into_one_misleading_value():
    # A real bug found by giving this function its own fresh-eyes sweep:
    # `"".join(text_parts)` had no separator between entries -- harmless
    # for the common one-block message, but AgentLoop builds exactly
    # this shape (one ToolResultBlock per concurrently-dispatched tool
    # call) every time a model turn issues more than one tool call,
    # ordinary use, not an edge case. Confirmed live before the fix: two
    # tool results ("4" and "2") glued into a single wire-format string
    # "42" -- not just losing which result came from which tool (an
    # honest, already-documented limitation of this backend having no
    # dedicated tool-result role at all), but actively fusing two
    # distinct values into a different, misleading one.
    m = Message(
        role="user",
        content=[
            ToolResultBlock(tool_call_id="call_1", content=[TextBlock(text="4")]),
            ToolResultBlock(tool_call_id="call_2", content=[TextBlock(text="2")]),
        ],
    )
    out = await _to_ollama_message(m)
    assert out["content"] == "4\n2"
    assert out["content"] != "42"


async def test_tool_result_with_an_image_sends_it_via_the_real_images_field_not_silently_dropped():
    # A real bug found by a fresh-eyes sweep, the identical gap already
    # found and fixed in the Anthropic/OpenAI/Google adapters, just
    # never applied here: the plain text-only join silently dropped any
    # non-text content inside a tool result -- ImageBlock most
    # importantly (a screenshot/image-generation tool's result,
    # ToolResultBlock.content's own type comment already names this as
    # an anticipated shape). Confirmed live before this fix: a
    # ToolResultBlock with a TextBlock and an ImageBlock produced a wire
    # message with the image gone, no trace anywhere. Ollama's own
    # /api/chat wire format has no per-tool-result content shape at all
    # -- fixed by extracting the image into the same message-level
    # `images` array a top-level ImageBlock already populates (the only
    # shape Ollama's own API actually supports), rather than dropping it.
    raw_bytes = b"\x89PNG\r\n\x1a\nfake but real bytes for this test"
    m = Message(
        role="user",
        content=[
            ToolResultBlock(
                tool_call_id="call_1",
                content=[
                    TextBlock(text="here is the screenshot"),
                    ImageBlock(media_type="image/png", data=raw_bytes),
                ],
            )
        ],
    )
    out = await _to_ollama_message(m)
    assert out["content"] == "here is the screenshot"
    assert out["images"] == [base64.standard_b64encode(raw_bytes).decode()]


async def test_tool_result_with_an_unsupported_block_still_raises_not_silently_dropped():
    # A sibling of the image case above: a genuinely untranslatable
    # block type inside a tool result (no image, no text mapping at
    # all) must still raise, matching every other adapter's own loud-
    # failure discipline for exactly this case -- not every non-text
    # block inside a tool result is now silently accepted.
    from sarva.multimodal.content import ThinkingBlock

    m = Message(
        role="user",
        content=[ToolResultBlock(tool_call_id="call_1", content=[ThinkingBlock(text="reasoning")])],
    )
    with pytest.raises(ValueError, match="ThinkingBlock"):
        await _to_ollama_message(m)


async def test_message_with_no_tool_calls_omits_the_tool_calls_key():
    m = Message(role="user", content=[TextBlock(text="hi")])
    out = await _to_ollama_message(m)
    assert "tool_calls" not in out


async def test_image_block_translates_to_the_real_ollama_images_field():
    # Ollama's own /api/chat wire format wants raw base64 (no data: URI
    # prefix, no media_type) in a per-message `images` array -- the
    # exact shape confirmed against a real running server with a real
    # vision-capable model (moondream) before writing this adapter code.
    raw_bytes = b"\x89PNG\r\n\x1a\nfake but real bytes for this test"
    m = Message(
        role="user",
        content=[
            TextBlock(text="what's in this image?"),
            ImageBlock(media_type="image/png", data=raw_bytes),
        ],
    )
    out = await _to_ollama_message(m)
    assert out["role"] == "user"
    assert out["content"] == "what's in this image?"
    assert out["images"] == [base64.standard_b64encode(raw_bytes).decode()]


async def test_message_with_no_images_omits_the_images_key():
    m = Message(role="user", content=[TextBlock(text="hi")])
    out = await _to_ollama_message(m)
    assert "images" not in out


async def test_multiple_images_in_one_message_all_collected():
    m = Message(
        role="user",
        content=[
            ImageBlock(media_type="image/png", data=b"first"),
            TextBlock(text="compare these"),
            ImageBlock(media_type="image/png", data=b"second"),
        ],
    )
    out = await _to_ollama_message(m)
    assert out["images"] == [
        base64.standard_b64encode(b"first").decode(),
        base64.standard_b64encode(b"second").decode(),
    ]


async def test_document_block_still_raises_not_silently_dropped():
    # Real bug this pins: the loop over m.content had no `else` branch
    # at all, so an unhandled block type (still true for DocumentBlock --
    # ImageBlock is now genuinely supported, above) was silently
    # skipped -- the model would answer as if it had never received it.
    # Matches the Anthropic/OpenAI/Google/Foundry adapters' own
    # loud-failure guard for exactly this case.
    m = Message(
        role="user",
        content=[DocumentBlock(media_type="application/pdf", data=b"x")],
    )
    with pytest.raises(ValueError, match="DocumentBlock"):
        await _to_ollama_message(m)


def test_strip_local_prefix_removes_the_provider_namespace():
    assert _strip_local_prefix("ollama/qwen3:8b") == "qwen3:8b"


def test_strip_local_prefix_is_a_no_op_without_a_slash():
    assert _strip_local_prefix("qwen3:8b") == "qwen3:8b"


def _req() -> GenerateRequest:
    return GenerateRequest(
        model="qwen3:8b", messages=[Message(role="user", content=[TextBlock(text="hi")])]
    )


def _provider(body: bytes) -> OllamaProvider:
    # httpx.MockTransport -- the same "no network, no server needed"
    # discipline test_fetch.py already established for hermetic httpx
    # testing, and the most direct way to exercise generate()'s own
    # streaming/parsing logic: unlike the OpenAI/Google adapters (which
    # go through their SDKs, unit-tested with duck-typed chunk
    # stand-ins in test_openai_provider_streaming.py), Ollama talks to
    # httpx directly, so generate() had zero conformance coverage of
    # its own before this -- only ever exercised live, skipped without
    # a real running server.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaProvider(client=client)


@pytest.mark.asyncio
async def test_generate_streams_real_text_deltas_from_ndjson_lines():
    body = (
        b'{"message": {"content": "Hel"}, "done": false}\n'
        b'{"message": {"content": "lo"}, "done": true}\n'
    )
    events = [e async for e in _provider(body).generate(_req())]

    deltas = [e.text for e in events if isinstance(e, TextDeltaEvent)]
    assert deltas == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_generate_yields_a_clean_stream_error_on_a_malformed_ndjson_line():
    # A real bug found by actually feeding a genuinely truncated NDJSON
    # line through a mocked transport (the real-world trigger: a
    # network glitch or proxy cutting the stream mid-line): the raw
    # json.JSONDecodeError propagated straight out of generate()'s
    # async generator uncaught instead of the clean StreamErrorEvent
    # the status_code >= 400 branch already produces for a malformed
    # *response*. AgentLoop's own `except Exception` around the
    # provider call happens to catch this too (so a single AgentLoop
    # turn wouldn't crash), but sarva.eval's benchmark harness and
    # sarva.distill only catch ProviderError per-case -- an uncaught
    # JSONDecodeError from one bad line would abort either of those
    # entirely, losing every remaining case's results.
    body = b'{"message": {"content": "Hel"}, "done": false}\n' + b'{"message": {"content": "lo'

    events = [e async for e in _provider(body).generate(_req())]

    assert events[0] == TextDeltaEvent(text="Hel")
    assert isinstance(events[-1], StreamErrorEvent)
    assert events[-1].retryable is True
    assert "malformed streaming response" in events[-1].detail


@pytest.mark.asyncio
async def test_interleaved_text_and_tool_calls_keep_their_chronological_order():
    # A real bug found by a fresh-eyes sweep, the identical gap already
    # found and fixed in google_provider.py, just never propagated here:
    # text was accumulated into one running string across the WHOLE
    # stream and only ever spliced into the final message ONCE,
    # unconditionally first -- tool calls were already appended in true
    # chronological order, but text occurring between or after them got
    # silently pulled forward and merged ahead of every tool call.
    # Confirmed live before this fix: an ordinary sequential-tool-calling
    # turn (reasoning text, a call, more reasoning text, another call --
    # ordinary ReAct-style behavior for local tool-using models served
    # via Ollama, not contrived) produced ONE TextBlock with both
    # segments concatenated, hoisted ahead of BOTH tool calls --
    # corrupting the persisted Message AgentLoop appends straight to
    # transcript_out/SessionStore and resends as history on the next turn.
    import json

    lines = [
        {"message": {"content": "Let me check the weather. "}, "done": False},
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}],
            },
            "done": False,
        },
        {"message": {"content": "Now let me also check traffic."}, "done": False},
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "get_traffic", "arguments": {"city": "NYC"}}}],
            },
            "done": True,
        },
    ]
    body = b"\n".join(json.dumps(x).encode() for x in lines) + b"\n"

    events = [e async for e in _provider(body).generate(_req())]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    shapes = [
        (type(b).__name__, getattr(b, "text", None) or getattr(b, "name", None))
        for b in done.message.content
    ]
    assert shapes == [
        ("TextBlock", "Let me check the weather. "),
        ("ToolCallBlock", "get_weather"),
        ("TextBlock", "Now let me also check traffic."),
        ("ToolCallBlock", "get_traffic"),
    ]


@pytest.mark.asyncio
async def test_consecutive_text_deltas_with_no_call_between_them_still_merge_into_one_block():
    # A sibling check for the fix above: text arriving as multiple
    # streamed deltas with nothing else interleaved must still collapse
    # into a single TextBlock, not fragment into one block per delta.
    body = (
        b'{"message": {"content": "Hel"}, "done": false}\n'
        b'{"message": {"content": "lo"}, "done": false}\n'
        b'{"message": {"content": ", world."}, "done": true}\n'
    )
    events = [e async for e in _provider(body).generate(_req())]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    assert len(done.message.content) == 1
    assert done.message.content[0].text == "Hello, world."


@pytest.mark.asyncio
async def test_generate_reports_max_tokens_when_ollama_truncates_by_length():
    # A real bug found by a fresh-eyes sweep, the identical "sibling
    # adapter has a real stop/finish-reason mapping, this one never got
    # one" gap already found and fixed in all three OTHER provider
    # adapters (anthropic_provider.py's pause_turn, openai_provider.py's
    # function_call, google_provider.py's RECITATION/etc.): Ollama's
    # real streaming /api/chat response includes a done_reason field on
    # its final chunk ("length" when generation was cut off by hitting
    # num_predict/the context window -- documented Ollama wire
    # behavior), but nothing here ever read it. Confirmed live before
    # this fix: this exact mocked response reported StopReason.END_TURN
    # instead, silently indistinguishable from a genuinely complete
    # response.
    body = (
        b'{"message": {"content": "this is a very long"}, "done": false}\n'
        b'{"message": {"content": ""}, "done": true, "done_reason": "length"}\n'
    )
    events = [e async for e in _provider(body).generate(_req())]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    assert done.stop_reason == StopReason.MAX_TOKENS


@pytest.mark.asyncio
async def test_generate_still_reports_end_turn_for_a_normal_stop():
    # Regression guard for the fix above: an ordinary, complete
    # response (done_reason="stop") must not be affected.
    body = b'{"message": {"content": "hello"}, "done": true, "done_reason": "stop"}\n'
    events = [e async for e in _provider(body).generate(_req())]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    assert done.stop_reason == StopReason.END_TURN


@pytest.mark.asyncio
async def test_generate_reports_real_token_usage_not_a_hardcoded_zero():
    # A real bug found by a later fresh-eyes sweep: this adapter always
    # yielded usage=Usage() (0 input, 0 output tokens), with a comment
    # claiming Ollama's /api/chat "doesn't return token usage" --
    # factually wrong. The same final chunk this adapter already reads
    # done_reason off also carries prompt_eval_count/eval_count,
    # documented Ollama wire fields, confirmed live against a real
    # running Ollama server. AgentLoop's own `spend.total_tokens +=
    # done.usage.input_tokens + done.usage.output_tokens` means a
    # hardcoded-zero Usage silently made Budget.max_total_tokens
    # unenforceable for every Ollama-routed call.
    body = (
        b'{"message": {"content": "hello"}, "done": true, "done_reason": "stop", '
        b'"prompt_eval_count": 35, "eval_count": 3}\n'
    )
    events = [e async for e in _provider(body).generate(_req())]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    assert done.usage.input_tokens == 35
    assert done.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_generate_tool_use_is_not_overridden_by_a_stop_done_reason():
    # Ollama reports done_reason="stop" even for a turn that ends in a
    # tool call (it has no distinct "made a tool call" reason of its
    # own) -- presence of a tool call must still win over the raw
    # done_reason, the same rule every sibling adapter already applies.
    body = (
        b'{"message": {"content": "", "tool_calls": '
        b'[{"function": {"name": "f", "arguments": {}}}]}, "done": false}\n'
        b'{"message": {"content": ""}, "done": true, "done_reason": "stop"}\n'
    )
    events = [e async for e in _provider(body).generate(_req())]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    assert done.stop_reason == StopReason.TOOL_USE
