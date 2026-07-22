"""Conformance tests for sarva.multimodal.degraders.image — the
degradation registry's first real converter."""

from __future__ import annotations

import io

import pytest
from PIL import Image
from sarva.multimodal.content import ImageBlock, Message, Modality, TextBlock, degrade_message
from sarva.multimodal.degraders.image import ImageDecodeError, ImageToTextDegrader


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


async def test_degrade_produces_one_text_block_with_correct_dimensions_and_format():
    raw = _png_bytes(64, 32)
    block = ImageBlock(media_type="image/png", data=raw)

    out = await ImageToTextDegrader().degrade(block)

    assert len(out) == 1
    assert isinstance(out[0], TextBlock)
    assert "64x32" in out[0].text
    assert "PNG" in out[0].text


async def test_degrade_reports_the_actual_byte_size_not_a_guess():
    raw = _png_bytes(200, 200)
    block = ImageBlock(media_type="image/png", data=raw)
    out = await ImageToTextDegrader().degrade(block)
    expected_kb = len(raw) / 1024
    assert f"~{expected_kb:.0f}KB" in out[0].text


async def test_degrade_does_not_fabricate_visual_content():
    # The design principle this test pins: report verifiable metadata,
    # never invent a caption for content the degrader has no way to see.
    raw = _png_bytes(10, 10)
    block = ImageBlock(media_type="image/png", data=raw)
    out = await ImageToTextDegrader().degrade(block)
    assert "could not be described" in out[0].text


async def test_degrade_raises_a_clear_error_on_corrupt_image_bytes():
    block = ImageBlock(media_type="image/png", data=b"not a real image, just garbage bytes")
    with pytest.raises(ImageDecodeError):
        await ImageToTextDegrader().degrade(block)


async def test_degrade_works_from_a_path_source(tmp_path):
    raw = _png_bytes(50, 50)
    path = tmp_path / "test.png"
    path.write_bytes(raw)
    block = ImageBlock(media_type="image/png", path=str(path))
    out = await ImageToTextDegrader().degrade(block)
    assert "50x50" in out[0].text


async def test_wired_into_degrade_message_end_to_end():
    # Proves the concrete implementation actually satisfies the Degrader
    # protocol and works through the real recursive dispatcher in
    # content.py -- the framework was already proven with a fake
    # (_EchoDegrader in test_content.py); this proves it with something
    # real, not just when ImageToTextDegrader is called directly.
    raw = _png_bytes(16, 16)
    msg = Message(role="user", content=[ImageBlock(media_type="image/png", data=raw)])

    result = await degrade_message(
        msg,
        supported={Modality.TEXT},
        degraders={Modality.IMAGE: ImageToTextDegrader()},
    )

    assert len(result.content) == 1
    assert isinstance(result.content[0], TextBlock)
    assert "16x16" in result.content[0].text
