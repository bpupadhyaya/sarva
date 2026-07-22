"""sarva.multimodal.degraders.video — the third concrete Degrader.

Same honesty principle as `ImageToTextDegrader`/`AudioToTextDegrader`:
report only what's verifiably known, never fabricate content. Unlike
audio (where stdlib `wave` can genuinely decode the one format —
uncompressed WAV — most likely to show up), there is no standard-library
module that can decode *any* real-world video container at all, so this
degrader never attempts byte-level decoding — it always falls back to
whatever the block itself already declares (`media_type`, `duration_s`
if the caller set it, and the always-knowable byte size).

What this degrader deliberately does NOT do, despite `Degrader`'s own
docstring using exactly this as its motivating example ("video ->
[image frames + text transcript]"): sample frames into `ImageBlock`s.
Real frame extraction needs a video-decoding dependency (ffmpeg/opencv)
this project doesn't carry yet — not a heavier lift than justified for
a metadata-only converter. Tracked as real, deferred follow-up work
rather than silently declared "the video degrader" and left at that.
"""

from __future__ import annotations

from sarva.multimodal.content import Modality, TextBlock, VideoBlock
from sarva.multimodal.fetch import resolve_media_bytes


class VideoToTextDegrader:
    source = Modality.VIDEO

    async def degrade(self, block: VideoBlock) -> list[TextBlock]:
        raw = await resolve_media_bytes(block)
        size_kb = len(raw) / 1024
        if block.duration_s is not None:
            duration_text = f"{block.duration_s:.1f}s"
        else:
            duration_text = "unknown duration"
        text = (
            f"[Video attached: {duration_text}, {block.media_type}, ~{size_kb:.0f}KB. "
            "The current model does not support video input, so its content "
            "could not be described.]"
        )
        return [TextBlock(text=text)]
