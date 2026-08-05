"""Hermetic tests for the one piece of genuinely novel, adapter-specific
logic in google_provider.py: inferring TOOL_USE from the *presence* of a
function_call part rather than trusting Gemini's own `finish_reason`.

Unlike Anthropic/OpenAI, whose finish reason says "tool_use"/"tool_calls"
directly, Gemini reports `STOP` even when the response includes
function_call parts -- there is no distinct "made a tool call" finish
reason at all. Getting this wrong (trusting finish_reason alone) would
silently misreport every Gemini tool-use turn as END_TURN, which would
in turn make the agent loop treat a turn that actually requested a tool
call as if the model were simply done -- a real, structural bug a
live-only test might not surface immediately if the first few manual
runs happened to be text-only. Everything else in this adapter
(translation) follows the established pattern of "unit-test pure
translation, verify the rest live" -- see test_google_provider.py and
tests/live/test_live_providers.py.

Uses duck-typed `SimpleNamespace` stand-ins for the google-genai SDK's
response objects rather than constructing real `GenerateContentResponse`
instances: this test's job is proving our own stop-reason inference is
correct, not re-verifying the SDK's own wire parsing.
"""

from __future__ import annotations

from types import SimpleNamespace

from google.genai import errors
from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    StopReason,
    StreamErrorEvent,
    ToolCallEvent,
)
from sarva.providers.google_provider import GoogleProvider


def _part(text=None, thought=False, function_call=None):
    return SimpleNamespace(text=text, thought=thought, function_call=function_call)


def _function_call(id, name, args):
    return SimpleNamespace(id=id, name=name, args=args)


def _chunk(parts=None, finish_reason=None, usage=None):
    content = SimpleNamespace(parts=parts) if parts else None
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _usage(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        cached_content_token_count=0,
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _FakeErrorStream:
    """Raises `exc` on the first `__anext__` -- simulates the SDK's own
    `_load_json_from_response()` failing to parse a malformed streaming
    chunk mid-stream."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._exc


class _FakeClient:
    def __init__(self, chunks=None, error: Exception | None = None):
        async def generate_content_stream(**kwargs):
            if error is not None:
                return _FakeErrorStream(error)
            return _FakeStream(chunks or [])

        models = SimpleNamespace(generate_content_stream=generate_content_stream)
        self.aio = SimpleNamespace(models=models)


def _simple_request(model: str = "gemini-x") -> GenerateRequest:
    return GenerateRequest(
        model=model, messages=[Message(role="user", content=[TextBlock(text="hi")])]
    )


async def test_tool_call_infers_tool_use_despite_stop_finish_reason():
    # Gemini's real behavior: finish_reason is "STOP" even when the
    # candidate made a function call. Trusting finish_reason alone would
    # wrongly report END_TURN here.
    chunks = [
        _chunk(
            parts=[_part(function_call=_function_call("t1", "get_weather", {"city": "Paris"}))],
            finish_reason="STOP",
            usage=_usage(10, 5),
        ),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_events) == 1
    assert tool_events[0].call.name == "get_weather"
    assert tool_events[0].call.arguments == {"city": "Paris"}

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.TOOL_USE
    assert done.usage.input_tokens == 10
    assert done.usage.output_tokens == 5


async def test_text_only_stream_produces_end_turn():
    chunks = [
        _chunk(parts=[_part(text="Hello")]),
        _chunk(parts=[_part(text=", world")], finish_reason="STOP", usage=_usage(3, 2)),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.END_TURN
    assert done.message.content[0].text == "Hello, world"


async def test_a_blocked_generation_is_reported_as_refusal_not_a_silent_empty_success():
    # A real bug found by a fresh-eyes sweep: the real google-genai
    # FinishReason enum has 17 members, and only 5 were ever mapped in
    # _STOP_REASON_MAP -- everything else, including RECITATION (Gemini
    # refusing because the answer recites training-data text too
    # closely, e.g. song lyrics -- a well-documented, ordinary
    # occurrence, not contrived) fell through the map's own `.get(...,
    # StopReason.END_TURN)` default straight into the *success* path.
    # Gemini sends no content parts on a blocked candidate, so this
    # reported a blocked generation as a normal, successful END_TURN
    # with a silently EMPTY message -- confirmed live before this fix.
    chunks = [_chunk(parts=None, finish_reason="RECITATION", usage=_usage(10, 0))]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.REFUSAL
    assert done.message.content == []


async def test_an_unrecognized_finish_reason_fails_safe_as_refusal_not_end_turn():
    # The other half of the same fix: the map's own default was changed
    # from StopReason.END_TURN to StopReason.REFUSAL, so any FUTURE
    # FinishReason value this map hasn't explicitly named yet (a new SDK
    # release, an image-out-specific variant) also fails safe -- a clean
    # REFUSAL a caller can see, never a silently "successful" empty
    # response just because the exact enum member wasn't enumerated.
    chunks = [_chunk(parts=None, finish_reason="SOME_FUTURE_REASON", usage=_usage(10, 0))]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.REFUSAL


async def test_thought_parts_become_thinking_delta_not_text_delta():
    from sarva.providers.base import ThinkingDeltaEvent

    chunks = [
        _chunk(parts=[_part(text="pondering...", thought=True)]),
        _chunk(parts=[_part(text="the answer is 4")], finish_reason="STOP", usage=_usage(5, 5)),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    thinking_events = [e for e in events if isinstance(e, ThinkingDeltaEvent)]
    assert len(thinking_events) == 1
    assert thinking_events[0].text == "pondering..."
    done = events[-1]
    assert done.message.content[0].text == "the answer is 4"


async def test_parallel_calls_to_the_same_tool_get_distinct_ids_when_gemini_omits_them():
    # A real bug found by giving this adapter's tool-call id handling
    # its own fresh-eyes sweep: google-genai's own FunctionCall.id is
    # documented Optional, and Gemini frequently leaves it unset. The
    # previous fallback (`id or name`) collapsed every call sharing a
    # name onto the SAME id whenever id was missing -- confirmed live
    # before this fix, two ordinary parallel calls to the same tool in
    # one turn (a completely ordinary agentic pattern, not contrived)
    # both got id="get_weather". That collision reaches the wire: `_to_
    # gemini_content` echoes tool_call_id into FunctionResponse.id, a
    # field Gemini's own docs say exists so the model can match a
    # response back to the call that produced it -- both responses would
    # have carried the identical id, defeating that correlation for
    # exactly the two calls that most needed it disambiguated.
    chunks = [
        _chunk(
            parts=[
                _part(function_call=_function_call(None, "get_weather", {"city": "NYC"})),
                _part(function_call=_function_call(None, "get_weather", {"city": "LA"})),
            ],
            finish_reason="STOP",
            usage=_usage(10, 5),
        ),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 2
    assert calls[0].id != calls[1].id
    assert calls[0].arguments == {"city": "NYC"}
    assert calls[1].arguments == {"city": "LA"}


async def test_a_real_function_call_id_from_gemini_is_used_as_is():
    # The common case (Gemini DOES supply an id) must stay exactly as
    # given, not overwritten by the synthetic fallback.
    chunks = [
        _chunk(
            parts=[_part(function_call=_function_call("real-id-123", "get_weather", {}))],
            finish_reason="STOP",
            usage=_usage(5, 5),
        ),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert calls[0].id == "real-id-123"


async def test_interleaved_text_and_tool_calls_keep_their_chronological_order():
    # A real bug found by giving this adapter's block ordering its own
    # fresh-eyes sweep: text was accumulated into one running string
    # across the WHOLE stream and only ever spliced into `blocks` once,
    # at the very front, after the loop ended -- unlike ToolCallBlock/
    # ImageBlock, which were already appended in true chronological
    # order as each part arrived. Confirmed live before this fix: an
    # ordinary sequential-tool-calling turn (reasoning text, a call,
    # more reasoning text, another call -- documented Gemini behavior)
    # produced ONE TextBlock with both text segments concatenated,
    # hoisted ahead of BOTH tool calls -- misrepresenting which
    # reasoning text justified which call in the persisted Message, the
    # exact thing AgentLoop appends to transcript_out/SessionStore and
    # re-sends as history on the next turn.
    chunks = [
        _chunk(parts=[_part(text="Let me check the weather in NYC.")]),
        _chunk(
            parts=[_part(function_call=_function_call("call-1", "get_weather", {"city": "NYC"}))]
        ),
        _chunk(parts=[_part(text="Now let me check the weather in LA.")]),
        _chunk(
            parts=[_part(function_call=_function_call("call-2", "get_weather", {"city": "LA"}))],
            finish_reason="STOP",
            usage=_usage(10, 10),
        ),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    done = [e for e in events if isinstance(e, DoneEvent)][0]
    shapes = [
        (type(b).__name__, getattr(b, "text", None) or getattr(b, "name", None))
        for b in done.message.content
    ]
    assert shapes == [
        ("TextBlock", "Let me check the weather in NYC."),
        ("ToolCallBlock", "get_weather"),
        ("TextBlock", "Now let me check the weather in LA."),
        ("ToolCallBlock", "get_weather"),
    ]


async def test_consecutive_text_deltas_with_no_call_between_them_still_merge_into_one_block():
    # A sibling check for the fix above: text arriving as multiple
    # streamed deltas with nothing else interleaved must still collapse
    # into a single TextBlock, not fragment into one block per delta.
    chunks = [
        _chunk(parts=[_part(text="Hel")]),
        _chunk(parts=[_part(text="lo")]),
        _chunk(parts=[_part(text=", world.")], finish_reason="STOP", usage=_usage(5, 5)),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    done = [e for e in events if isinstance(e, DoneEvent)][0]
    assert len(done.message.content) == 1
    assert done.message.content[0].text == "Hello, world."


async def test_generate_yields_a_clean_stream_error_on_a_malformed_sdk_response():
    # A real bug found by reading google-genai's own source, the same
    # way as the identical gap fixed in ollama_provider.py: the SDK's
    # `_load_json_from_response()` wraps a failed `json.loads()` on a
    # streaming chunk in `errors.UnknownApiResponseError` -- a
    # `ValueError` subclass, NOT an `errors.APIError` subclass like
    # ClientError/ServerError, so this adapter's existing two `except`
    # clauses never touched it and it propagated uncaught.
    exc = errors.UnknownApiResponseError("Failed to parse response as JSON. Raw response: garbage")
    provider = GoogleProvider(client=_FakeClient(error=exc))

    events = [e async for e in provider.generate(_simple_request())]

    assert len(events) == 1
    assert isinstance(events[0], StreamErrorEvent)
    assert events[0].code == "provider"
    assert events[0].retryable is True
    assert "Failed to parse response as JSON" in events[0].detail
