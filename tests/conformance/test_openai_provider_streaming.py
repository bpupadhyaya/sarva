"""Hermetic tests for the one piece of genuinely novel, adapter-specific
logic in openai_provider.py: incremental tool-call-argument accumulation.

Unlike Anthropic (whose SDK hands back a single, already-assembled
`get_final_message()`) or Ollama (whose chat API sends each tool call
complete in one chunk), OpenAI streams a tool call's `arguments` string
as fragments across many chunks, keyed by `index` -- real, non-trivial,
easy-to-get-subtly-wrong logic (an index bug would cross-contaminate two
concurrent tool calls' argument fragments) that a live-only test
wouldn't reliably force, since a live model might never happen to
interleave two tool calls' chunks in one run. Everything else in this
adapter (translation, error mapping) follows the established
Anthropic/Ollama pattern of "unit-test pure translation, verify the rest
live" -- see test_openai_provider.py and tests/live/test_live_providers.py.

Uses duck-typed `SimpleNamespace` stand-ins for the openai SDK's chunk
objects rather than constructing real `ChatCompletionChunk` instances:
this test's job is proving our own accumulation logic is correct, not
re-verifying the SDK's own (pydantic-validated) wire parsing.
"""

from __future__ import annotations

from types import SimpleNamespace

from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    StopReason,
    StreamErrorEvent,
    ToolCallEvent,
)
from sarva.providers.openai_provider import OpenAIProvider


def _chunk(content=None, tool_call_deltas=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_call_deltas)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tc_delta(index, id=None, name=None, arguments=None):
    function = (
        SimpleNamespace(name=name, arguments=arguments)
        if (name is not None or arguments is not None)
        else None
    )
    return SimpleNamespace(index=index, id=id, function=function)


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _FakeClient:
    def __init__(self, chunks):
        self._chunks = chunks

        async def create(**kwargs):
            return _FakeStream(self._chunks)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    async def close(self):
        pass


def _usage(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=None,
    )


def _simple_request(model: str = "gpt-x") -> GenerateRequest:
    return GenerateRequest(
        model=model, messages=[Message(role="user", content=[TextBlock(text="hi")])]
    )


async def test_tool_call_arguments_reassemble_across_incremental_chunks():
    # Two concurrent tool calls (index 0 and 1) with their argument
    # fragments interleaved chunk-by-chunk -- proves index-keyed
    # accumulation doesn't cross-contaminate them.
    chunks = [
        _chunk(tool_call_deltas=[_tc_delta(0, id="call_a", name="get_weather", arguments='{"ci')]),
        _chunk(tool_call_deltas=[_tc_delta(1, id="call_b", name="get_time", arguments='{"tz')]),
        _chunk(tool_call_deltas=[_tc_delta(0, arguments='ty": "Paris"}')]),
        _chunk(tool_call_deltas=[_tc_delta(1, arguments='": "UTC"}')]),
        _chunk(finish_reason="tool_calls", usage=_usage(10, 5)),
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]

    tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
    calls_by_id = {e.call.id: e.call for e in tool_events}
    assert set(calls_by_id) == {"call_a", "call_b"}
    assert calls_by_id["call_a"].name == "get_weather"
    assert calls_by_id["call_a"].arguments == {"city": "Paris"}
    assert calls_by_id["call_b"].name == "get_time"
    assert calls_by_id["call_b"].arguments == {"tz": "UTC"}

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.TOOL_USE
    assert done.usage.input_tokens == 10
    assert done.usage.output_tokens == 5


async def test_text_only_stream_produces_end_turn():
    chunks = [
        _chunk(content="Hello"),
        _chunk(content=", world"),
        _chunk(finish_reason="stop", usage=_usage(3, 2)),
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.END_TURN
    assert done.message.content[0].text == "Hello, world"


class _FakeClientCapturingKwargs:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls: list[dict] = []

        async def create(**kwargs):
            self.calls.append(kwargs)
            return _FakeStream(self._chunks)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    async def close(self):
        pass


async def test_generate_translates_stop_sequences_to_the_real_sdk_kwarg():
    # A real bug found by a fresh-eyes sweep, applying the "one sibling
    # has the feature, others silently diverge" lens that already caught
    # round 149's DPO-schedule bug: GenerateConfig.stop_sequences is a
    # shared, provider-agnostic field every adapter is supposed to honor
    # (docs/providers.md names it as part of the contract), but only
    # GoogleProvider ever actually translated it -- this adapter (and
    # Anthropic's, and Ollama's) silently dropped it, even though the
    # real OpenAI SDK genuinely supports a `stop` kwarg (str or
    # list[str]) on chat.completions.create(). Confirmed live before
    # this fix: a request with stop_sequences=["STOP_HERE"] sent no
    # `stop` kwarg at all.
    from sarva.providers.base import GenerateConfig

    chunks = [_chunk(content="hi"), _chunk(finish_reason="stop", usage=_usage(1, 1))]
    client = _FakeClientCapturingKwargs(chunks)
    provider = OpenAIProvider(client=client)
    request = GenerateRequest(
        model="gpt-x",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        config=GenerateConfig(stop_sequences=["STOP_HERE"]),
    )

    [e async for e in provider.generate(request)]

    assert client.calls[0]["stop"] == ["STOP_HERE"]


async def test_a_legacy_function_call_finish_reason_is_not_reported_as_a_silent_success():
    # A real bug found by a fresh-eyes sweep, the third instance of a bug
    # class already fixed in both sibling adapters (Google's FinishReason
    # mapping, Anthropic's pause_turn): the real OpenAI SDK's own
    # finish_reason type also includes "function_call" -- the deprecated
    # legacy single-function-calling API's own stop reason, still a real,
    # documented value the SDK's type carries. This adapter only ever
    # parses delta.tool_calls (the modern API), never the legacy
    # delta.function_call field, so a response using the old API produces
    # no ToolCallBlock at all. Confirmed live before this fix: it fell
    # through _STOP_REASON_MAP's own `.get(..., StopReason.END_TURN)`
    # default straight into the success path, reporting an unparsed
    # function call as a normal, complete answer.
    chunks = [
        _chunk(
            content="unparsed legacy function call",
            finish_reason="function_call",
            usage=_usage(10, 5),
        )
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason != StopReason.END_TURN
    assert done.stop_reason == StopReason.REFUSAL


async def test_an_unrecognized_finish_reason_fails_safe_as_refusal_not_end_turn():
    # The other half of the same fix: the map's own default was changed
    # from StopReason.END_TURN to StopReason.REFUSAL, so any FUTURE
    # finish_reason value this map hasn't explicitly named yet also
    # fails safe -- a clean REFUSAL a caller can see, never a silently
    # "successful" response just because the exact value wasn't
    # enumerated.
    chunks = [_chunk(content="x", finish_reason="some_future_reason", usage=_usage(10, 5))]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.stop_reason == StopReason.REFUSAL


async def test_interleaved_text_and_tool_calls_keep_their_chronological_order():
    # A real bug found by a fresh-eyes sweep, the identical gap already
    # found and fixed in google_provider.py/ollama_provider.py, just
    # never propagated here: text was accumulated into ONE running
    # string across the WHOLE stream and only ever spliced into the
    # final message once, unconditionally first -- tool calls were
    # already assembled in true chronological order relative to each
    # other, but text occurring between or after them got silently
    # pulled forward and merged ahead of every tool call. Confirmed
    # live before this fix: an ordinary sequential-tool-calling turn
    # (reasoning text, a call, more reasoning text, another call --
    # ordinary ReAct-style behavior, not contrived) produced ONE
    # TextBlock with both segments concatenated, hoisted ahead of BOTH
    # tool calls -- corrupting the persisted Message AgentLoop appends
    # straight to transcript_out/SessionStore and resends as history.
    chunks = [
        _chunk(content="Let me check the weather first.\n"),
        _chunk(tool_call_deltas=[_tc_delta(0, id="call_1", name="get_weather", arguments="{}")]),
        _chunk(content="Now let me check the time.\n"),
        _chunk(
            tool_call_deltas=[_tc_delta(1, id="call_2", name="get_time", arguments="{}")],
            finish_reason="tool_calls",
            usage=_usage(10, 5),
        ),
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    shapes = [
        (type(b).__name__, getattr(b, "text", None) or getattr(b, "name", None))
        for b in done.message.content
    ]
    assert shapes == [
        ("TextBlock", "Let me check the weather first.\n"),
        ("ToolCallBlock", "get_weather"),
        ("TextBlock", "Now let me check the time.\n"),
        ("ToolCallBlock", "get_time"),
    ]


async def test_consecutive_text_deltas_with_no_call_between_them_still_merge_into_one_block():
    # A sibling check for the fix above: text arriving as multiple
    # streamed deltas with nothing else interleaved must still collapse
    # into a single TextBlock, not fragment into one block per delta.
    chunks = [
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(content=", world.", finish_reason="stop", usage=_usage(3, 2)),
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]
    done = [e for e in events if isinstance(e, DoneEvent)][0]

    assert len(done.message.content) == 1
    assert done.message.content[0].text == "Hello, world."


async def test_malformed_tool_call_arguments_do_not_crash_the_adapter():
    # A real bug found by a fresh-eyes sweep, not by this test as
    # originally written: a tool call whose accumulated argument
    # fragments never form valid JSON (truncated stream, a real,
    # documented GPT tool-calling failure mode) used to silently
    # degrade to an empty dict here -- no error, no signal anywhere.
    # AgentLoop would then dispatch that corrupted {} to whatever tool
    # the model actually meant to call with real arguments: a built-in
    # tool's required-key access raises a confusing bare KeyError with
    # no way to trace it back to dropped arguments; an MCP tool forwards
    # {} straight to the remote server with zero local validation,
    # silently executing the WRONG action for any tool with optional/
    # defaulted parameters -- undetectable corruption, not just a
    # confusing message. "Must not crash the adapter" was the right
    # instinct (a raw JSONDecodeError propagating out would be worse),
    # but "silently substitute {}" was the wrong way to satisfy it --
    # every other failure path in this same function already signals a
    # real problem via StreamErrorEvent (retryable=True re-calls the
    # model fresh, the same recovery a rate limit or network blip
    # already gets), and this is that same treatment, not a crash.
    chunks = [
        _chunk(tool_call_deltas=[_tc_delta(0, id="call_a", name="broken", arguments="{not json")]),
        _chunk(finish_reason="tool_calls", usage=_usage(1, 1)),
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]  # must not raise

    assert len(events) == 1
    assert isinstance(events[0], StreamErrorEvent)
    assert events[0].code == "malformed_tool_arguments"
    assert events[0].retryable is True
    assert "broken" in events[0].detail


async def test_well_formed_tool_call_arguments_still_parse_normally():
    # Regression guard: the malformed-JSON branch above must never
    # trigger on ordinary, valid tool-call arguments -- the
    # overwhelmingly common case this function handles on every real
    # tool-using turn.
    chunks = [
        _chunk(tool_call_deltas=[_tc_delta(0, id="call_a", name="get_weather", arguments="")]),
        _chunk(tool_call_deltas=[_tc_delta(0, arguments='{"city": "Paris"}')]),
        _chunk(finish_reason="tool_calls", usage=_usage(1, 1)),
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]

    tool_event = next(e for e in events if isinstance(e, ToolCallEvent))
    assert tool_event.call.arguments == {"city": "Paris"}
