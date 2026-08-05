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
    required_modalities,
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


def test_required_modalities_sees_media_nested_inside_a_tool_result():
    # A real bug found by a fresh-eyes sweep: MODALITY_OF hard-maps
    # "tool_result" -> Modality.TEXT unconditionally, so a plain
    # modality_of() scan over top-level blocks always reported a
    # ToolResultBlock as text-only, completely blind to what's actually
    # nested inside it -- an MCP server's real ImageContent result
    # (screenshot/browser-automation/chart-generation tools all produce
    # this) becomes a genuine ImageBlock inside ToolResultBlock.content
    # (mcp_client.py's own _convert_content), invisible to any caller
    # that only checks the top-level modality.
    blocks = [
        ToolResultBlock(
            tool_call_id="t1",
            content=[
                TextBlock(text="here's the screenshot"),
                ImageBlock(media_type="image/png", data=b"x"),
            ],
        )
    ]
    assert required_modalities(blocks) == {Modality.TEXT, Modality.IMAGE}


@pytest.mark.asyncio
async def test_degrade_message_recurses_into_a_tool_results_nested_content():
    # The degradation-side counterpart to the required_modalities fix
    # above: without recursing into ToolResultBlock.content,
    # _degrade_block's own modality_of() check always saw "text" for the
    # whole block and returned it completely unmodified -- any nested
    # media sailed straight past degradation, directly contradicting
    # this module's own docstring guarantee ("content is never silently
    # dropped"). Confirmed live before this fix: degrading a
    # ToolResultBlock containing an ImageBlock against a text-only
    # model left the raw ImageBlock completely untouched.
    msg = Message(
        role="user",
        content=[
            ToolResultBlock(
                tool_call_id="t1",
                content=[
                    TextBlock(text="a screenshot"),
                    ImageBlock(media_type="image/png", data=b"x"),
                ],
            )
        ],
    )
    degraders = {Modality.IMAGE: _EchoDegrader(Modality.IMAGE)}
    out = await degrade_message(msg, supported={Modality.TEXT}, degraders=degraders)

    assert len(out.content) == 1
    result = out.content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.tool_call_id == "t1"
    assert all(b.type == "text" for b in result.content)
    assert any("image converted to text" in b.text for b in result.content)


@pytest.mark.asyncio
async def test_degrade_message_still_raises_when_a_tool_results_nested_media_has_no_path():
    msg = Message(
        role="user",
        content=[
            ToolResultBlock(
                tool_call_id="t1",
                content=[ImageBlock(media_type="image/png", data=b"x")],
            )
        ],
    )
    with pytest.raises(UnsupportedModalityError):
        await degrade_message(msg, supported={Modality.TEXT}, degraders={})


def test_thinking_opacity_round_trip():
    original = ThinkingBlock(text="reasoning...", provider_data={"signature": "xyz-123"})
    restored = ThinkingBlock.model_validate_json(original.model_dump_json())
    assert restored.provider_data == original.provider_data
