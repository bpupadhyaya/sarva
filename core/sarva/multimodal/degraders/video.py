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

**The actual decode runs in an isolated subprocess
(`_video_worker.py`), not in-process.** A real bug found by directly
fuzzing a valid MP4 (random byte flips) and running the exact
`av.open`/`container.decode` call this module used to make directly:
3 of 60 seeded trials killed the process outright with a real SIGBUS
(signal 10) -- a native memory fault inside PyAV/libavcodec, not a
catchable Python exception. No `try`/`except`, however broad, can stop
a native crash from taking down the whole process; the worker module's
own docstring has the full detail. `_sample_frames` now spawns that
worker and treats a crash exactly the same as an ordinary decode
failure -- both mean "couldn't safely decode this, fall back to the
metadata-only report" -- so the process-isolation fix required no
change to this module's own error-handling *shape*, only to where the
actual decoding happens.
"""

from __future__ import annotations

import asyncio
import struct
import sys

from sarva.multimodal.content import ImageBlock, Modality, TextBlock, VideoBlock
from sarva.multimodal.fetch import resolve_media_bytes

_DECODE_TIMEOUT_SECONDS = 30


async def _sample_frames(raw: bytes) -> tuple[list[bytes], float | None] | None:
    """Returns (sampled PNG frame bytes, real decoded duration in seconds)
    on success, or None if `raw` can't be decoded as video at all -- the
    caller falls back to the metadata-only report either way. Runs the
    actual decode in `_video_worker.py`, a separate subprocess, so a
    native crash inside PyAV's decoder only kills that subprocess (see
    this module's own docstring for why that's a real, not
    hypothetical, concern)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "sarva.multimodal.degraders._video_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(raw), timeout=_DECODE_TIMEOUT_SECONDS)
    except TimeoutError:
        # Same timeout-must-actually-kill-the-process discipline as
        # RunShellTool's own fix for the identical class of bug: a
        # hung decode (e.g. a video crafted to loop or exhaust memory)
        # must not leave an orphaned worker process running forever.
        proc.kill()
        await proc.wait()
        return None

    # A nonzero OR negative (killed-by-signal) returncode covers both
    # an ordinary decode failure and a native crash identically -- the
    # whole point of routing through a subprocess is that the caller
    # never needs to tell those two cases apart.
    if proc.returncode != 0 or len(stdout) < 8:
        return None

    (duration_raw,) = struct.unpack(">d", stdout[:8])
    duration_s = None if duration_raw != duration_raw else duration_raw  # NaN sentinel

    frames: list[bytes] = []
    offset = 8
    while offset + 4 <= len(stdout):
        (length,) = struct.unpack(">I", stdout[offset : offset + 4])
        offset += 4
        frames.append(stdout[offset : offset + length])
        offset += length

    if not frames:
        return None
    return frames, duration_s


class VideoToTextDegrader:
    source = Modality.VIDEO

    async def degrade(self, block: VideoBlock) -> list[ImageBlock | TextBlock]:
        raw = await resolve_media_bytes(block)
        size_kb = len(raw) / 1024
        sampled = await _sample_frames(raw)

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
