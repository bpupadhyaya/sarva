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

from sarva.multimodal.content import Message, TextBlock
from sarva.providers.base import DoneEvent, GenerateRequest, StopReason, ToolCallEvent
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


class _FakeClient:
    def __init__(self, chunks):
        async def generate_content_stream(**kwargs):
            return _FakeStream(chunks)

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
