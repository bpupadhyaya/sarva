"""sarva.multimodal.degraders.video — the third concrete Degrader, now
with real frame sampling closing the gap this module's own docstring
named and deferred: `Degrader`'s docstring in content.py uses "video ->
[image frames + text transcript]" as its motivating example, and until
now this degrader never did that.

Uses PyAV (`av`), which statically bundles its own decoder libraries
into the wheel it ships on PyPI for macOS/Linux/Windows -- unlike
shelling out to a system `ffmpeg` binary, there's no cross-platform CI
availability gamble (this project already paid that tax once, the hard
way, getting Windows sidecar freezing working -- see BUILD-JOURNAL). The
audio degrader's stdlib-only tradeoff was made when the only realistic
options were "stdlib `wave`, which can't touch compressed audio" or "a
heavy dependency not justified for a metadata-only converter"; a
self-contained, genuinely portable decoding library changes that
calculus for video, where sampling actual frames is the whole point of
the modality (there's no stdlib video decoder at all to fall back to,
unlike audio's WAV case).

Same honesty principle as the other two degraders throughout: sampled
frames are real decoded pixels, never a fabricated description of what
they show -- that's still the router/agent loop's decision (route to a
vision-capable model), not this converter's job. And undecodable bytes
(corrupt data, an unsupported container, a block that's actually audio
mislabeled as video) fall back to the original metadata-only report
rather than raising -- "couldn't decode this particular file" is a real,
expected case for a converter that has to handle whatever bytes a caller
hands it, not a bug.
"""

from __future__ import annotations

import io

import av
from av.error import FFmpegError

from sarva.multimodal.content import ImageBlock, Modality, TextBlock, VideoBlock
from sarva.multimodal.fetch import resolve_media_bytes

_MAX_SAMPLED_FRAMES = 4


def _sample_frames(raw: bytes) -> tuple[list[bytes], float | None] | None:
    """Returns (sampled PNG frame bytes, real decoded duration in seconds)
    on success, or None if `raw` can't be decoded as video at all -- the
    caller falls back to the metadata-only report either way."""
    try:
        with av.open(io.BytesIO(raw)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            duration_s = float(stream.duration * stream.time_base) if stream.duration else None
            frames = list(container.decode(stream))
    except (FFmpegError, ValueError):
        return None

    if not frames:
        return None

    # Evenly spaced indices across the decoded frames, capped at
    # _MAX_SAMPLED_FRAMES -- bounds output size regardless of how long the
    # source video actually is, same spirit as the corpus pipeline's
    # length filters bounding a single document's size.
    count = min(_MAX_SAMPLED_FRAMES, len(frames))
    step = len(frames) / count
    indices = [int(i * step) for i in range(count)]

    png_frames = []
    for i in indices:
        buf = io.BytesIO()
        frames[i].to_image().save(buf, format="PNG")
        png_frames.append(buf.getvalue())
    return png_frames, duration_s


class VideoToTextDegrader:
    source = Modality.VIDEO

    async def degrade(self, block: VideoBlock) -> list[ImageBlock | TextBlock]:
        raw = await resolve_media_bytes(block)
        size_kb = len(raw) / 1024
        sampled = _sample_frames(raw)

        if sampled is None:
            duration_text = (
                f"{block.duration_s:.1f}s" if block.duration_s is not None else "unknown duration"
            )
            text = (
                f"[Video attached: {duration_text}, {block.media_type}, ~{size_kb:.0f}KB. "
                "Its frames could not be decoded, so its content could not be described.]"
            )
            return [TextBlock(text=text)]

        png_frames, duration_s = sampled
        duration_s = duration_s if duration_s is not None else block.duration_s
        duration_text = f"{duration_s:.1f}s" if duration_s is not None else "unknown duration"
        text = (
            f"[Video attached: {duration_text}, {block.media_type}, ~{size_kb:.0f}KB. "
            f"{len(png_frames)} frame(s) sampled below. The current model does not "
            "support video input directly, so only these still frames (and no audio "
            "track) could be examined.]"
        )
        frame_blocks: list[ImageBlock | TextBlock] = [
            ImageBlock(media_type="image/png", data=data) for data in png_frames
        ]
        return [TextBlock(text=text), *frame_blocks]
