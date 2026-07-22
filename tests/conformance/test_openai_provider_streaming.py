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
from sarva.providers.base import DoneEvent, GenerateRequest, StopReason, ToolCallEvent
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


async def test_malformed_tool_call_arguments_do_not_crash_the_adapter():
    # A tool call whose accumulated argument fragments never form valid
    # JSON (truncated stream, provider bug) must degrade to an empty
    # dict, not raise out of the adapter.
    chunks = [
        _chunk(tool_call_deltas=[_tc_delta(0, id="call_a", name="broken", arguments="{not json")]),
        _chunk(finish_reason="tool_calls", usage=_usage(1, 1)),
    ]
    provider = OpenAIProvider(client=_FakeClient(chunks))
    req = _simple_request()

    events = [e async for e in provider.generate(req)]

    tool_event = next(e for e in events if isinstance(e, ToolCallEvent))
    assert tool_event.call.arguments == {}
