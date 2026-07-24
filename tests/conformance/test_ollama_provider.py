"""Unit tests for the Ollama adapter's translation function --
`_to_ollama_message` had zero conformance coverage until now (only
exercised by `tests/live/test_live_providers.py`, skipped without a
real running server), unlike every other adapter's own dedicated
translation-unit test file. No network, no server needed here.
"""

from __future__ import annotations

import base64

import pytest
from sarva.multimodal.content import (
    DocumentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sarva.providers.ollama_provider import _strip_local_prefix, _to_ollama_message


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
