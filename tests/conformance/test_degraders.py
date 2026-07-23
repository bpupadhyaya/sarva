"""Conformance tests for sarva.multimodal.degraders — the degradation
registry's concrete converters."""

from __future__ import annotations

import io
import wave

import av
import pytest
from PIL import Image
from sarva.multimodal.content import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    Modality,
    TextBlock,
    VideoBlock,
    degrade_message,
)
from sarva.multimodal.degraders import (
    AudioToTextDegrader,
    DocumentToTextDegrader,
    VideoToTextDegrader,
    default_degraders,
)
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


def _minimal_pdf_bytes(text: str) -> bytes:
    """A real, hand-built, valid single-page PDF whose content stream
    literally contains `text` -- constructed with correct byte offsets
    (not a fixture file checked into the repo, and not trusting pypdf's
    xref-recovery leniency), so the test proves a real
    write-bytes-then-extract round trip rather than a fabricated one."""
    content_stream = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length "
        + str(len(content_stream)).encode()
        + b">>\nstream\n"
        + content_stream
        + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(buf.tell())
        buf.write(f"{i} 0 obj\n".encode())
        buf.write(obj)
        buf.write(b"\nendobj\n")
    xref_offset = buf.tell()
    n = len(objects) + 1
    buf.write(f"xref\n0 {n}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for off in offsets:
        buf.write(f"{off:010d} 00000 n \n".encode())
    buf.write(b"trailer\n")
    buf.write(f"<</Size {n}/Root 1 0 R>>\n".encode())
    buf.write(b"startxref\n")
    buf.write(f"{xref_offset}\n".encode())
    buf.write(b"%%EOF")
    return buf.getvalue()


def _synthetic_video_bytes(n_frames: int, fps: int = 10, size: tuple[int, int] = (32, 24)) -> bytes:
    """A real, tiny, genuinely PyAV-decodable mp4 -- encoded with PyAV
    itself, not a fixture file checked into the repo, so the test proves
    a real encode+decode round trip rather than trusting a byte blob no
    one can easily regenerate or verify."""
    width, height = size
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            # A distinct solid color per frame so a test can tell frames
            # apart, not just count them.
            shade = (i * 40) % 256
            img = Image.new("RGB", size, color=(shade, 0, 255 - shade))
            frame = av.VideoFrame.from_image(img).reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
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
    # Genuinely undecodable bytes (not a real container at all) fall back
    # to the declared-metadata-only report -- this exercises that
    # fallback path specifically, distinct from the real-decode tests
    # below which exercise a genuine PyAV-decodable video.
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


async def test_video_degrade_samples_real_frames_from_a_decodable_video():
    raw = _synthetic_video_bytes(n_frames=20, fps=10)
    block = VideoBlock(media_type="video/mp4", data=raw)

    out = await VideoToTextDegrader().degrade(block)

    text_blocks = [b for b in out if isinstance(b, TextBlock)]
    image_blocks = [b for b in out if isinstance(b, ImageBlock)]
    assert len(text_blocks) == 1
    assert 1 <= len(image_blocks) <= 4
    # 20 frames @ 10fps == a real, decoded 2.0s duration -- not the
    # declared-metadata fallback text ("unknown duration" or whatever the
    # caller happened to pass in duration_s, which is None here).
    assert "2.0s" in text_blocks[0].text
    # Each sampled frame is real, Pillow-decodable PNG data, not a stub.
    for img_block in image_blocks:
        with Image.open(io.BytesIO(img_block.data)) as decoded:
            assert decoded.size == (32, 24)
            assert decoded.format == "PNG"


async def test_video_degrade_caps_sampled_frames_regardless_of_video_length():
    raw = _synthetic_video_bytes(n_frames=50, fps=25)
    block = VideoBlock(media_type="video/mp4", data=raw)

    out = await VideoToTextDegrader().degrade(block)

    image_blocks = [b for b in out if isinstance(b, ImageBlock)]
    assert len(image_blocks) == 4  # capped, not one per decoded frame


async def test_video_degrade_prefers_real_decoded_duration_over_declared():
    # block.duration_s deliberately wrong -- the real decode should win,
    # proving this isn't just echoing back whatever the caller claimed.
    raw = _synthetic_video_bytes(n_frames=20, fps=10)
    block = VideoBlock(media_type="video/mp4", data=raw, duration_s=999.0)

    out = await VideoToTextDegrader().degrade(block)

    text_blocks = [b for b in out if isinstance(b, TextBlock)]
    assert "2.0s" in text_blocks[0].text
    assert "999.0s" not in text_blocks[0].text


async def test_video_frames_recursively_degrade_to_text_for_a_text_only_target():
    # Proves the full documented chain: video -> sampled image frames ->
    # (model still can't see images either) -> text, via degrade_message's
    # own recursion -- not just that VideoToTextDegrader emits ImageBlocks
    # in isolation.
    raw = _synthetic_video_bytes(n_frames=8, fps=8)
    block = VideoBlock(media_type="video/mp4", data=raw)
    msg = Message(role="user", content=[block])

    result = await degrade_message(
        msg,
        supported={Modality.TEXT},
        degraders={Modality.VIDEO: VideoToTextDegrader(), Modality.IMAGE: ImageToTextDegrader()},
    )

    assert all(isinstance(b, TextBlock) for b in result.content)
    assert len(result.content) >= 2  # the video summary, plus one text block per sampled frame
    assert any("does not support video input" in b.text for b in result.content)
    assert any("32x24" in b.text for b in result.content)  # from ImageToTextDegrader


async def test_video_degrade_falls_back_cleanly_on_a_video_stream_with_no_frames():
    # An mp4 container with zero frames muxed -- decodable as a container,
    # but with nothing to sample. Must still degrade honestly, not crash.
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv420p"
        for packet in stream.encode():
            container.mux(packet)
    block = VideoBlock(media_type="video/mp4", data=buf.getvalue(), duration_s=3.0)

    out = await VideoToTextDegrader().degrade(block)

    assert len(out) == 1
    assert isinstance(out[0], TextBlock)
    assert "3.0s" in out[0].text  # falls back to the declared value


# ---------- DocumentToTextDegrader ----------


async def test_document_degrader_extracts_real_text_from_a_real_pdf():
    raw = _minimal_pdf_bytes("Hello World")
    block = DocumentBlock(media_type="application/pdf", data=raw, title="greeting")

    out = await DocumentToTextDegrader().degrade(block)

    assert len(out) == 1
    assert isinstance(out[0], TextBlock)
    assert "Hello World" in out[0].text  # the real extracted text, not a fabricated summary
    assert "'greeting'" in out[0].text
    assert "application/pdf" in out[0].text


async def test_document_degrader_extracts_real_text_from_plain_text_media_types():
    block = DocumentBlock(media_type="text/markdown", data=b"# A real heading\n\nBody text.")

    out = await DocumentToTextDegrader().degrade(block)

    assert len(out) == 1
    assert "# A real heading" in out[0].text
    assert "Body text." in out[0].text


async def test_document_degrader_truncates_very_long_extracted_text_honestly():
    long_text = "x" * 25_000
    block = DocumentBlock(media_type="text/plain", data=long_text.encode())

    out = await DocumentToTextDegrader().degrade(block)

    assert "truncated to 20,000 of 25,000 characters" in out[0].text
    # The actual extracted body really is capped, not just claimed to be
    # -- check the body itself (after the header's own "...:]\n\n"
    # separator), not a naive substring count over the whole message,
    # which would also match incidental "x"s in words like "text/plain".
    body = out[0].text.split(":]\n\n", 1)[1]
    assert body == "x" * 20_000


async def test_document_degrader_falls_back_cleanly_on_a_corrupt_pdf():
    block = DocumentBlock(media_type="application/pdf", data=b"not a real pdf at all")

    out = await DocumentToTextDegrader().degrade(block)

    assert len(out) == 1
    assert "could not be extracted" in out[0].text
    assert "application/pdf" in out[0].text


async def test_document_degrader_falls_back_cleanly_on_an_unsupported_format():
    # e.g. .docx -- a real, named, deliberately unbuilt format (see the
    # module's own docstring), not something silently mishandled.
    block = DocumentBlock(
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        data=b"PK\x03\x04binary docx bytes, not actually parsed",
    )

    out = await DocumentToTextDegrader().degrade(block)

    assert len(out) == 1
    assert "could not be extracted" in out[0].text


async def test_document_degrades_recursively_to_a_supported_modality_via_degrade_message():
    raw = _minimal_pdf_bytes("Recursion works")
    msg = Message(role="user", content=[DocumentBlock(media_type="application/pdf", data=raw)])

    result = await degrade_message(
        msg, supported={Modality.TEXT}, degraders={Modality.DOCUMENT: DocumentToTextDegrader()}
    )

    assert len(result.content) == 1
    assert isinstance(result.content[0], TextBlock)
    assert "Recursion works" in result.content[0].text


# ---------- default_degraders ----------


async def test_default_degraders_covers_image_audio_video_and_document():
    degraders = default_degraders()
    assert set(degraders) == {Modality.IMAGE, Modality.AUDIO, Modality.VIDEO, Modality.DOCUMENT}
    assert isinstance(degraders[Modality.IMAGE], ImageToTextDegrader)
    assert isinstance(degraders[Modality.AUDIO], AudioToTextDegrader)
    assert isinstance(degraders[Modality.VIDEO], VideoToTextDegrader)
    assert isinstance(degraders[Modality.DOCUMENT], DocumentToTextDegrader)


async def test_default_degraders_returns_fresh_instances_each_call():
    # Stateless degraders, but the dict itself shouldn't be a shared
    # mutable singleton a caller could accidentally corrupt across
    # unrelated AgentLoop instances.
    a = default_degraders()
    b = default_degraders()
    assert a is not b
