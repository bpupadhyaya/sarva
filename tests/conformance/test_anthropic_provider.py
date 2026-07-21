"""Unit tests for the Anthropic adapter's pure translation function.

No network, no API key — `_to_anthropic_message` is a pure function and
worth testing in isolation from the streaming call. Live end-to-end
behavior is covered by tests/live/test_live_providers.py.
"""

from __future__ import annotations

import base64

from sarva.multimodal.content import ImageBlock, Message, TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.anthropic_provider import _to_anthropic_message


def test_text_block_translation():
    m = Message(role="user", content=[TextBlock(text="hello")])
    out = _to_anthropic_message(m)
    assert out == {"role": "user", "content": [{"type": "text", "text": "hello"}]}


def test_image_block_translation_base64_round_trips():
    raw = b"\x89PNG\r\n\x1a\n"
    m = Message(role="user", content=[ImageBlock(media_type="image/png", data=raw)])
    out = _to_anthropic_message(m)

    block = out["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(block["source"]["data"]) == raw


def test_tool_call_and_result_translation():
    call = ToolCallBlock(id="t1", name="get_weather", arguments={"city": "Paris"})
    m1 = Message(role="assistant", content=[call])
    out1 = _to_anthropic_message(m1)
    assert out1["content"][0] == {
        "type": "tool_use",
        "id": "t1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }

    result = ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="sunny")])
    m2 = Message(role="user", content=[result])
    out2 = _to_anthropic_message(m2)
    assert out2["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "sunny",
        "is_error": False,
    }
