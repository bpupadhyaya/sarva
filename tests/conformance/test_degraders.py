"""Conformance tests for sarva.multimodal.degraders — the degradation
registry's concrete converters."""

from __future__ import annotations

import io
import wave

import pytest
from PIL import Image
from sarva.multimodal.content import (
    AudioBlock,
    ImageBlock,
    Message,
    Modality,
    TextBlock,
    VideoBlock,
    degrade_message,
)
from sarva.multimodal.degraders import AudioToTextDegrader, VideoToTextDegrader, default_degraders
from sarva.multimodal.degraders.image import ImageDecodeError, ImageToTextDegrader


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _wav_bytes(duration_s: float, framerate: int = 8000) -> bytes:
    n_frames = int(duration_s * framerate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b"\x00\x00" * n_frames)
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


# ---------- AudioToTextDegrader ----------


async def test_audio_degrade_decodes_real_wav_duration_from_bytes():
    # Unlike the image degrader (which always decodes via Pillow), this
    # is the one format AudioToTextDegrader *can* genuinely decode --
    # proves it actually reads the bytes rather than only ever trusting
    # declared metadata.
    raw = _wav_bytes(duration_s=2.5, framerate=8000)
    block = AudioBlock(media_type="audio/wav", data=raw)

    out = await AudioToTextDegrader().degrade(block)

    assert len(out) == 1
    assert "2.5s" in out[0].text
    assert "audio/wav" in out[0].text


async def test_audio_degrade_falls_back_to_declared_duration_for_undecodable_formats():
    # Real-world audio is overwhelmingly compressed (MP3/AAC/...), which
    # stdlib `wave` cannot parse -- that must fall back to whatever the
    # block already declares, not raise (the deliberate difference from
    # ImageToTextDegrader's corrupt-bytes behavior).
    block = AudioBlock(media_type="audio/mp3", data=b"ID3 not a real mp3 either", duration_s=42.0)

    out = await AudioToTextDegrader().degrade(block)

    assert "42.0s" in out[0].text
    assert "audio/mp3" in out[0].text


async def test_audio_degrade_reports_unknown_duration_when_nothing_is_knowable():
    block = AudioBlock(media_type="audio/mp3", data=b"not real audio data at all")
    out = await AudioToTextDegrader().degrade(block)
    assert "unknown duration" in out[0].text


async def test_audio_degrade_does_not_fabricate_content():
    raw = _wav_bytes(duration_s=1.0)
    block = AudioBlock(media_type="audio/wav", data=raw)
    out = await AudioToTextDegrader().degrade(block)
    assert "could not be transcribed" in out[0].text


async def test_audio_wired_into_degrade_message_end_to_end():
    raw = _wav_bytes(duration_s=3.0)
    msg = Message(role="user", content=[AudioBlock(media_type="audio/wav", data=raw)])

    result = await degrade_message(
        msg,
        supported={Modality.TEXT},
        degraders={Modality.AUDIO: AudioToTextDegrader()},
    )

    assert len(result.content) == 1
    assert "3.0s" in result.content[0].text


# ---------- VideoToTextDegrader ----------


async def test_video_degrade_reports_declared_duration():
    # No stdlib module can decode a real video container at all (unlike
    # audio's one WAV special case) -- this degrader always reports
    # declared metadata, never attempts byte-level decoding.
    block = VideoBlock(media_type="video/mp4", data=b"not real video data", duration_s=12.5)
    out = await VideoToTextDegrader().degrade(block)
    assert len(out) == 1
    assert "12.5s" in out[0].text
    assert "video/mp4" in out[0].text


async def test_video_degrade_reports_unknown_duration_when_not_declared():
    block = VideoBlock(media_type="video/mp4", data=b"not real video data")
    out = await VideoToTextDegrader().degrade(block)
    assert "unknown duration" in out[0].text


async def test_video_degrade_reports_actual_byte_size():
    raw = b"x" * 4096
    block = VideoBlock(media_type="video/mp4", data=raw, duration_s=1.0)
    out = await VideoToTextDegrader().degrade(block)
    expected_kb = len(raw) / 1024
    assert f"~{expected_kb:.0f}KB" in out[0].text


async def test_video_degrade_does_not_fabricate_content():
    block = VideoBlock(media_type="video/mp4", data=b"x", duration_s=1.0)
    out = await VideoToTextDegrader().degrade(block)
    assert "could not be described" in out[0].text


async def test_video_wired_into_degrade_message_end_to_end():
    block = VideoBlock(media_type="video/mp4", data=b"x", duration_s=7.0)
    msg = Message(role="user", content=[block])

    result = await degrade_message(
        msg,
        supported={Modality.TEXT},
        degraders={Modality.VIDEO: VideoToTextDegrader()},
    )

    assert len(result.content) == 1
    assert "7.0s" in result.content[0].text


# ---------- default_degraders ----------


async def test_default_degraders_covers_image_audio_and_video():
    degraders = default_degraders()
    assert set(degraders) == {Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
    assert isinstance(degraders[Modality.IMAGE], ImageToTextDegrader)
    assert isinstance(degraders[Modality.AUDIO], AudioToTextDegrader)
    assert isinstance(degraders[Modality.VIDEO], VideoToTextDegrader)


async def test_default_degraders_returns_fresh_instances_each_call():
    # Stateless degraders, but the dict itself shouldn't be a shared
    # mutable singleton a caller could accidentally corrupt across
    # unrelated AgentLoop instances.
    a = default_degraders()
    b = default_degraders()
    assert a is not b
