"""Hermetic, end-to-end proof that an extended-thinking + tool-use turn
survives a real multi-turn round trip through `AnthropicProvider`, not
just that `_to_anthropic_message` translates one block correctly in
isolation (see test_anthropic_provider.py for that unit-level proof).

Anthropic requires the ORIGINAL signature back on a reused thinking
block (an anti-tampering check) -- get it wrong and either the turn
gets silently degraded or the API rejects it. The property that
actually matters isn't "the translation function returns the right
dict when called directly," it's "a real two-turn conversation (model
thinks + calls a tool, caller sends the tool result back) produces a
second request whose `messages` payload contains the exact thinking
block Anthropic returned the first time, byte-for-byte." This drives
`AnthropicProvider.generate()` twice against a fake SDK client and
inspects the actual second call's kwargs, the same "prove the real
pipeline, not a mocked shortcut" bar test_google_provider_streaming.py
and test_openai_provider_streaming.py already hold themselves to.

Uses a duck-typed fake `client.messages.stream(...)` async context
manager rather than the real SDK's typed response objects -- this
test's job is proving Sarva's own history-threading and translation are
correct, not re-verifying the SDK's own wire parsing.
"""

from __future__ import annotations

from types import SimpleNamespace

from sarva.multimodal.content import Message, TextBlock, ToolResultBlock
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva.providers.base import DoneEvent, GenerateRequest


def _content_block(type_, **kwargs):
    return SimpleNamespace(type=type_, **kwargs)


def _usage(input_tokens, output_tokens):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class _FakeStream:
    def __init__(self, final_message):
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        return
        yield  # pragma: no cover -- makes this an async generator with no events

    async def get_final_message(self):
        return self._final_message


class _FakeMessagesAPI:
    def __init__(self, final_messages):
        self._final_messages = list(final_messages)
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self._final_messages.pop(0))


class _FakeClient:
    def __init__(self, final_messages):
        self.messages = _FakeMessagesAPI(final_messages)


async def test_a_signed_thinking_block_survives_a_real_two_turn_round_trip():
    turn_1_response = SimpleNamespace(
        content=[
            _content_block("thinking", thinking="the user wants the weather", signature="sig-xyz"),
            _content_block("tool_use", id="t1", name="get_weather", input={"city": "Paris"}),
        ],
        stop_reason="tool_use",
        usage=_usage(20, 10),
    )
    turn_2_response = SimpleNamespace(
        content=[_content_block("text", text="It's sunny in Paris.")],
        stop_reason="end_turn",
        usage=_usage(30, 8),
    )
    client = _FakeClient([turn_1_response, turn_2_response])
    provider = AnthropicProvider(client=client)

    request_1 = GenerateRequest(
        model="claude-opus-4-8",
        messages=[Message(role="user", content=[TextBlock(text="what's the weather in Paris?")])],
    )
    events_1 = [e async for e in provider.generate(request_1)]
    done_1 = events_1[-1]
    assert isinstance(done_1, DoneEvent)

    # Exactly what AgentLoop itself does (agent/loop.py: `messages.append(
    # done.message)`, then a fresh user turn carrying the tool result) --
    # no shortcut, the real history-threading shape.
    history = [*request_1.messages, done_1.message]
    history.append(
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_call_id="t1", content=[TextBlock(text="sunny, 22C")], is_error=False
                )
            ],
        )
    )
    request_2 = GenerateRequest(model="claude-opus-4-8", messages=history)
    events_2 = [e async for e in provider.generate(request_2)]
    done_2 = events_2[-1]
    assert isinstance(done_2, DoneEvent)

    second_call_messages = client.messages.calls[1]["messages"]
    assistant_turn = next(m for m in second_call_messages if m["role"] == "assistant")
    assert assistant_turn["content"][0] == {
        "type": "thinking",
        "thinking": "the user wants the weather",
        "signature": "sig-xyz",
    }
    assert assistant_turn["content"][1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }
