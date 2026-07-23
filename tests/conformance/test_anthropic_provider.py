"""Unit tests for the Anthropic adapter's translation function.

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
from sarva.providers.anthropic_provider import _to_anthropic_message


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


async def test_thinking_block_is_explicitly_dropped_not_translated():
    # Deliberate, named skip -- not yet round-tripped back to the API.
    # Verifies it doesn't appear in translated output and doesn't raise.
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
