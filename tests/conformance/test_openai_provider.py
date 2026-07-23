"""Unit tests for the OpenAI adapter's translation function.

No network, no API key — every block here carries an in-memory `data`
source, so `_to_openai_messages`'s only await (`resolve_media_bytes`, for
url-sourced images) never actually runs. Live end-to-end behavior is
covered by tests/live/test_live_providers.py.
"""

from __future__ import annotations

import base64

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
from sarva.providers.openai_provider import _to_openai_messages


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
