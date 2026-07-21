"""Conformance tests for sarva.multimodal.content — see spec-02 invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sarva.multimodal.content import (
    ImageBlock,
    Message,
    Modality,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    UnsupportedModalityError,
    degrade_message,
)


def _sample_message() -> Message:
    return Message(
        role="user",
        content=[
            TextBlock(text="hello"),
            ThinkingBlock(text="pondering", provider_data={"sig": "abc"}),
            ImageBlock(media_type="image/png", data=b"\x89PNG\r\n"),
            ToolCallBlock(id="t1", name="get_weather", arguments={"city": "Paris"}),
            ToolResultBlock(tool_call_id="t1", content=[TextBlock(text="sunny")], is_error=False),
        ],
    )


def test_immutability():
    b = TextBlock(text="hi")
    with pytest.raises(ValidationError):
        b.text = "bye"  # frozen


def test_round_trip():
    m = _sample_message()
    restored = Message.model_validate_json(m.model_dump_json())
    assert restored == m


def test_single_source_enforced():
    with pytest.raises(ValidationError):
        ImageBlock(media_type="image/png")  # zero sources
    with pytest.raises(ValidationError):
        ImageBlock(media_type="image/png", data=b"x", path="a.png")  # two sources


def test_discriminator_resolves_to_concrete_type():
    m = Message.model_validate(
        {"role": "user", "content": [{"type": "image", "media_type": "image/png", "data": "eA=="}]}
    )
    assert isinstance(m.content[0], ImageBlock)
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": [{"type": "not_a_type"}]})


class _EchoDegrader:
    """Degrades one modality into a single TextBlock describing it."""

    def __init__(self, source: Modality):
        self.source = source

    async def degrade(self, block):
        return [TextBlock(text=f"[{self.source.value} converted to text]")]


@pytest.mark.asyncio
async def test_degradation_produces_supported_blocks():
    from sarva.multimodal.content import VideoBlock

    msg = Message(
        role="user",
        content=[VideoBlock(media_type="video/mp4", data=b"\x00\x01")],
    )
    degraders = {
        Modality.VIDEO: _EchoDegrader(Modality.VIDEO),
    }
    out = await degrade_message(msg, supported={Modality.TEXT}, degraders=degraders)
    assert all(b.type == "text" for b in out.content)
    assert len(out.content) >= 1


@pytest.mark.asyncio
async def test_degradation_raises_without_a_path():
    from sarva.multimodal.content import VideoBlock

    msg = Message(role="user", content=[VideoBlock(media_type="video/mp4", data=b"\x00")])
    with pytest.raises(UnsupportedModalityError):
        await degrade_message(msg, supported={Modality.TEXT}, degraders={})


def test_thinking_opacity_round_trip():
    original = ThinkingBlock(text="reasoning...", provider_data={"signature": "xyz-123"})
    restored = ThinkingBlock.model_validate_json(original.model_dump_json())
    assert restored.provider_data == original.provider_data
