"""Unit tests for the Ollama adapter's translation function --
`_to_ollama_message` had zero conformance coverage until now (only
exercised by `tests/live/test_live_providers.py`, skipped without a
real running server), unlike every other adapter's own dedicated
translation-unit test file. No network, no server needed here.
"""

from __future__ import annotations

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


def test_text_block_translation():
    m = Message(role="user", content=[TextBlock(text="hello")])
    out = _to_ollama_message(m)
    assert out == {"role": "user", "content": "hello"}


def test_tool_call_translation():
    call = ToolCallBlock(id="t1", name="get_weather", arguments={"city": "Paris"})
    m = Message(role="assistant", content=[TextBlock(text="checking..."), call])
    out = _to_ollama_message(m)
    assert out["role"] == "assistant"
    assert out["content"] == "checking..."
    assert out["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]


def test_tool_result_renders_as_plain_text_content():
    # Ollama has no dedicated tool-result role -- the text just becomes
    # this message's own content.
    m = Message(
        role="user",
        content=[ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="sunny, 22C")])],
    )
    out = _to_ollama_message(m)
    assert out == {"role": "user", "content": "sunny, 22C"}


def test_message_with_no_tool_calls_omits_the_tool_calls_key():
    m = Message(role="user", content=[TextBlock(text="hi")])
    out = _to_ollama_message(m)
    assert "tool_calls" not in out


def test_unsupported_block_type_raises_instead_of_silently_dropping():
    # Real bug this pins: the loop over m.content had no `else` branch
    # at all, so an ImageBlock (or any other unhandled type) was
    # silently skipped -- the model would answer as if it had never
    # received the image. Matches the Anthropic/OpenAI/Google/Foundry
    # adapters' own loud-failure guard for exactly this case.
    m = Message(
        role="user",
        content=[
            TextBlock(text="what's in this image?"),
            ImageBlock(media_type="image/png", data=b"\x89PNG\r\n\x1a\n"),
        ],
    )
    with pytest.raises(ValueError, match="ImageBlock"):
        _to_ollama_message(m)


def test_document_block_also_raises_not_just_image():
    m = Message(
        role="user",
        content=[DocumentBlock(media_type="application/pdf", data=b"x")],
    )
    with pytest.raises(ValueError, match="DocumentBlock"):
        _to_ollama_message(m)


def test_strip_local_prefix_removes_the_provider_namespace():
    assert _strip_local_prefix("ollama/qwen3:8b") == "qwen3:8b"


def test_strip_local_prefix_is_a_no_op_without_a_slash():
    assert _strip_local_prefix("qwen3:8b") == "qwen3:8b"
