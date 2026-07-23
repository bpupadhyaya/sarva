"""Unit tests for the Google Gemini adapter's translation function.

No network, no API key — every block here carries an in-memory `data`
source, so `_to_gemini_content`'s only await (`resolve_media_bytes`, for
url-sourced images) never actually runs. Live end-to-end behavior is
covered by tests/live/test_live_providers.py.
"""

from __future__ import annotations

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
from sarva.providers.google_provider import _to_gemini_content, _tool_call_names


async def test_text_block_translation():
    m = Message(role="user", content=[TextBlock(text="hello")])
    out = await _to_gemini_content(m, {})
    assert out.role == "user"
    assert out.parts[0].text == "hello"


async def test_assistant_role_maps_to_model():
    m = Message(role="assistant", content=[TextBlock(text="hi there")])
    out = await _to_gemini_content(m, {})
    assert out.role == "model"


async def test_image_block_translation_round_trips_raw_bytes():
    raw = b"\x89PNG\r\n\x1a\n"
    m = Message(role="user", content=[ImageBlock(media_type="image/png", data=raw)])
    out = await _to_gemini_content(m, {})

    part = out.parts[0]
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == raw


async def test_tool_call_translation():
    call = ToolCallBlock(id="t1", name="get_weather", arguments={"city": "Paris"})
    m = Message(role="assistant", content=[call])
    out = await _to_gemini_content(m, {})

    part = out.parts[0]
    assert part.function_call.id == "t1"
    assert part.function_call.name == "get_weather"
    assert part.function_call.args == {"city": "Paris"}


async def test_tool_call_names_scans_all_messages():
    call = ToolCallBlock(id="t1", name="get_weather", arguments={})
    messages = [
        Message(role="user", content=[TextBlock(text="hi")]),
        Message(role="assistant", content=[call]),
    ]
    assert _tool_call_names(messages) == {"t1": "get_weather"}


async def test_tool_result_resolves_name_from_earlier_tool_call():
    # Gemini's FunctionResponse requires `name`, which ToolResultBlock
    # doesn't carry -- must be resolved from the tool_call_id -> name map
    # built from the *other* messages in the same request.
    call_names = {"t1": "get_weather"}
    m = Message(
        role="user", content=[ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="sunny")])]
    )
    out = await _to_gemini_content(m, call_names)

    part = out.parts[0]
    assert part.function_response.id == "t1"
    assert part.function_response.name == "get_weather"
    assert part.function_response.response == {"output": "sunny"}


async def test_tool_result_error_uses_the_error_key():
    m = Message(
        role="user",
        content=[
            ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="boom")], is_error=True)
        ],
    )
    out = await _to_gemini_content(m, {"t1": "explode"})
    assert out.parts[0].function_response.response == {"error": "boom"}


async def test_thinking_block_is_explicitly_dropped_not_translated():
    # Deliberate, named skip -- Gemini surfaces "thought" parts on the
    # way out (ThinkingDeltaEvent) but there's no documented way to feed
    # one back in as request content yet. Verifies it doesn't appear in
    # translated output and doesn't raise.
    m = Message(role="assistant", content=[ThinkingBlock(text="pondering"), TextBlock(text="hi")])
    out = await _to_gemini_content(m, {})
    assert len(out.parts) == 1
    assert out.parts[0].text == "hi"


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
        await _to_gemini_content(m, {})
