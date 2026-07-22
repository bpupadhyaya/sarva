"""sarva.multimodal.degraders.audio — the second concrete Degrader.

Same honesty principle as `ImageToTextDegrader`: report only what's
actually known, never fabricate a transcript. But the failure-handling
tradeoff is deliberately different. Pillow reliably decodes nearly every
real-world image format, so `ImageToTextDegrader` treats undecodable
bytes as a genuine error. Real-world audio is overwhelmingly compressed
(MP3/AAC/OGG/M4A) — the stdlib `wave` module only handles uncompressed
WAV, and pulling in a heavier dependency (ffmpeg/pydub) isn't justified
for a metadata-only converter. So this degrader never raises on an
undecodable format: "not WAV" is the *expected* case for most real
audio, not an error. It falls back to whatever the block itself already
declares (`media_type`, `duration_s` if the caller set it, and the
actual byte size, which is always knowable) rather than treating
"couldn't decode" as exceptional.
"""

from __future__ import annotations

import io
import wave

from sarva.multimodal.content import AudioBlock, Modality, TextBlock
from sarva.multimodal.fetch import resolve_media_bytes


def _decode_wav_duration(raw: bytes) -> float | None:
    # Deliberately broad except: any failure here just means "not a WAV
    # stdlib can parse" (the expected case for most real audio), not a
    # bug to surface — the caller falls back to declared metadata either
    # way, so there's nothing more specific to distinguish or re-raise.
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except Exception:
        return None


class AudioToTextDegrader:
    source = Modality.AUDIO

    async def degrade(self, block: AudioBlock) -> list[TextBlock]:
        raw = await resolve_media_bytes(block)
        duration_s = _decode_wav_duration(raw) or block.duration_s

        size_kb = len(raw) / 1024
        duration_text = f"{duration_s:.1f}s" if duration_s is not None else "unknown duration"
        text = (
            f"[Audio attached: {duration_text}, {block.media_type}, ~{size_kb:.0f}KB. "
            "The current model does not support audio input, so its content "
            "could not be transcribed.]"
        )
        return [TextBlock(text=text)]
