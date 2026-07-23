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

**Real transcription, not just metadata, when `sarva[audio]` is
installed.** Until now this degrader never actually transcribed
anything — it always said "could not be transcribed," even though
nothing about the *architecture* prevented real transcription, only a
missing implementation. `sarva.audio.transcribe` (real local
`faster-whisper` STT, see that module's docstring) is now attempted
first; only when the extra isn't installed, or transcription genuinely
fails on this specific audio, does this degrader fall back to the
original honest metadata-only message — never a fabricated transcript
standing in for one that couldn't actually be produced.
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

        from sarva.audio import stt_extra_installed, transcribe

        if stt_extra_installed():
            # Broad except deliberately: a transcription failure on THIS
            # audio (corrupt bytes, an unsupported codec, a model
            # loading error) should degrade to the honest metadata
            # fallback below, not crash the whole agent turn -- the same
            # "never let a best-effort enrichment take down the request"
            # posture the rest of this degrader already has.
            try:
                text = transcribe(raw)
                if text:
                    return [TextBlock(text=f"[Audio transcript: {text}]")]
            except Exception:
                pass

        duration_s = _decode_wav_duration(raw) or block.duration_s

        size_kb = len(raw) / 1024
        duration_text = f"{duration_s:.1f}s" if duration_s is not None else "unknown duration"
        text = (
            f"[Audio attached: {duration_text}, {block.media_type}, ~{size_kb:.0f}KB. "
            "The current model does not support audio input, so its content "
            "could not be transcribed.]"
        )
        return [TextBlock(text=text)]
