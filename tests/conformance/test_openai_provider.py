"""Unit tests for the OpenAI adapter's translation function, plus
`generate()`'s own streaming-error handling (below, via a duck-typed
fake client -- the same discipline test_anthropic_provider.py already
established, since this environment has no real OpenAI API key to test
the streaming path live).

No network, no API key — every block here carries an in-memory `data`
source, so `_to_openai_messages`'s only await (`resolve_media_bytes`, for
url-sourced images) never actually runs. Live end-to-end behavior is
covered by tests/live/test_live_providers.py.
"""

from __future__ import annotations

import base64

import httpx
import openai
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
from sarva.providers.base import GenerateRequest, StreamErrorEvent
from sarva.providers.openai_provider import OpenAIProvider, _to_openai_messages


async def test_text_block_translation():
    m = Message(role="user", content=[TextBlock(text="hello")])
    out = await _to_openai_messages(m)
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


async def test_image_block_translation_becomes_a_data_url():
    raw = b"\x89PNG\r\n\x1a\n"
    m = Message(role="user", content=[ImageBlock(media_type="image/png", data=raw)])
    out = await _to_openai_messages(m)

    part = out[0]["content"][0]
    assert part["type"] == "image_url"
    prefix = "data:image/png;base64,"
    assert part["image_url"]["url"].startswith(prefix)
    encoded = part["image_url"]["url"][len(prefix) :]
    assert base64.standard_b64decode(encoded) == raw


async def test_assistant_text_and_tool_call_combine_into_one_message():
    call = ToolCallBlock(id="t1", name="get_weather", arguments={"city": "Paris"})
    m = Message(role="assistant", content=[TextBlock(text="Let me check."), call])

    out = await _to_openai_messages(m)

    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == [{"type": "text", "text": "Let me check."}]
    assert out[0]["tool_calls"] == [
        {
            "id": "t1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]


async def test_tool_results_become_separate_tool_role_messages():
    # Unlike Anthropic, which lets multiple tool_result blocks live inside
    # one role="user" content array, OpenAI needs one dedicated
    # role="tool" message per tool_call_id -- the entire reason this
    # function returns a list instead of a single dict.
    m = Message(
        role="user",
        content=[
            ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="sunny")]),
            ToolResultBlock(tool_call_id="t2", content=[TextBlock(text="rainy")]),
        ],
    )

    out = await _to_openai_messages(m)

    assert out == [
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "t2", "content": "rainy"},
    ]


async def test_pure_tool_result_message_produces_no_leftover_main_message():
    m = Message(
        role="user", content=[ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="ok")])]
    )
    out = await _to_openai_messages(m)
    assert len(out) == 1
    assert out[0]["role"] == "tool"


async def test_tool_result_with_an_image_raises_instead_of_silently_dropping_it():
    # A real bug found by actually constructing a ToolResultBlock
    # carrying an ImageBlock: the plain `"".join(... TextBlock)` this
    # adapter used to build tool-message content silently dropped
    # anything that wasn't a TextBlock, with no error. Unlike Anthropic/
    # Gemini, this genuinely can't be fixed by sending the image along
    # instead -- OpenAI's own SDK type
    # (ChatCompletionToolMessageParam.content) only accepts text parts,
    # confirmed by reading it directly -- so this must raise rather than
    # guess at an unsupported wire shape.
    m = Message(
        role="user",
        content=[
            ToolResultBlock(
                tool_call_id="t1", content=[ImageBlock(media_type="image/png", data=b"\x89PNG")]
            )
        ],
    )

    with pytest.raises(ValueError, match="ImageBlock"):
        await _to_openai_messages(m)


async def test_thinking_block_is_explicitly_dropped_not_translated():
    # Deliberate, named skip -- OpenAI has no documented way to accept a
    # caller-supplied reasoning trace back on the next turn. Verifies it
    # doesn't appear in translated output and doesn't raise.
    m = Message(role="assistant", content=[ThinkingBlock(text="pondering"), TextBlock(text="hi")])
    out = await _to_openai_messages(m)
    assert out == [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]


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
        await _to_openai_messages(m)


class _FakeCompletions:
    def __init__(self, exc: Exception):
        self._exc = exc

    async def create(self, **kwargs):
        raise self._exc


class _FakeChat:
    def __init__(self, exc: Exception):
        self.completions = _FakeCompletions(exc)


class _FakeClient:
    def __init__(self, exc: Exception):
        self.chat = _FakeChat(exc)


def _req() -> GenerateRequest:
    return GenerateRequest(
        model="gpt-x", messages=[Message(role="user", content=[TextBlock(text="hi")])]
    )


def _real_response() -> httpx.Response:
    return httpx.Response(
        200, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )


async def test_generate_yields_a_clean_stream_error_on_a_malformed_sdk_response():
    # A real bug found by reading the SDK's own exception hierarchy, the
    # same way as the identical gap fixed in anthropic_provider.py:
    # openai.APIResponseValidationError (raised when the SDK can't parse
    # a malformed response) is a direct sibling of RateLimitError/
    # APIConnectionError/APIStatusError under the SDK's own APIError
    # base, not a subclass of any of them, so it propagated uncaught
    # before this fix.
    exc = openai.APIResponseValidationError(
        response=_real_response(), body=None, message="could not parse response"
    )
    provider = OpenAIProvider(client=_FakeClient(exc))

    events = [e async for e in provider.generate(_req())]

    assert len(events) == 1
    assert isinstance(events[0], StreamErrorEvent)
    assert events[0].code == "provider"
    assert "could not parse response" in events[0].detail


async def test_generate_still_handles_a_plain_rate_limit_error_as_before():
    # Regression guard: the new broad `except openai.APIError` catch-all
    # sits AFTER the three specific handlers, so it must never intercept
    # what RateLimitError's own more-specific handler already covers.
    exc = openai.RateLimitError("rate limited", response=_real_response(), body=None)
    provider = OpenAIProvider(client=_FakeClient(exc))

    events = [e async for e in provider.generate(_req())]

    assert len(events) == 1
    assert events[0].code == "rate_limit"
    assert events[0].retryable is True
