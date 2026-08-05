"""sarva._audio_decode_worker — isolates faster-whisper's PyAV-based
audio decoder in its own subprocess.

**A real bug found by directly fuzzing a real WAV file**, the same
methodology (and the same underlying PyAV/libavcodec dependency) as
`sarva.multimodal.degraders._video_worker`'s own SIGBUS fix -- see
that module's docstring for the full story.
`faster_whisper.audio.decode_audio()` uses the exact same
`av.open`/`container.decode` call that crashed the video decoder.
Confirmed directly: several fuzzed variants of a real WAV file (random
byte flips concentrated in the header, generated across multiple seeds)
killed the process outright with a real SIGBUS (signal 10), not a
Python exception, when fed through `sarva.audio.transcribe()`. A
corrupt or malicious audio attachment could crash the entire `sarva
serve`/CLI process the identical way a corrupted video attachment could
before that earlier fix.

Reads raw audio bytes from stdin, decodes via faster-whisper's own
`decode_audio()` at its default 16kHz mono sampling rate -- the fixed
rate `WhisperModel`'s own feature extractor always expects (not
per-model-size configurable), confirmed by reading
`faster_whisper.feature_extractor.FeatureExtractor`'s own default --
and writes the resulting float32 samples raw to stdout on success.
Exits nonzero (writing nothing to stdout) on any decode failure,
ordinary or native crash alike -- the caller can't tell, and doesn't
need to, which one happened; both mean "couldn't safely decode this."

**The duration cap (`sys.argv[1]`, `sarva.audio._MAX_TRANSCRIBE_SECONDS`)
is enforced HERE, before any of the decoded array is ever written to
stdout** -- a real bug found by a fresh-eyes sweep: the cap used to be
checked only in the parent, after `_decode_audio_isolated` had already
read the ENTIRE decoded array back via `capture_output=True`. Confirmed
live: a 7.2MB, 2-hour silent MP3 (an entirely ordinary, non-adversarial
shape -- a long lecture/meeting/voicemail recording with quiet
stretches) decoded to a 460.8MB float32 array that fully materialized
in the long-lived PARENT `sarva serve` process before the "protective"
cap ever fired, driving parent RSS from 27MB to ~1.2GB -- an unbounded,
linear relationship with no ceiling (a similarly-shaped ~24-hour file,
still well under 100MB on disk, would have materialized roughly 5.5GB
in the parent). Checking the cap here instead means a too-long file's
decoded array is discarded the moment this short-lived, disposable
subprocess exits -- the parent process only ever receives the small
stderr rejection message, never the multi-hundred-megabyte payload.
"""

from __future__ import annotations

import io
import sys


def _run(raw: bytes, max_seconds: float | None) -> int:
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(io.BytesIO(raw), sampling_rate=16000)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    duration_s = len(audio) / 16000
    if max_seconds is not None and duration_s > max_seconds:
        # A distinct, parseable marker so the parent (_decode_audio_
        # isolated) can raise its own specific "too long" RuntimeError
        # instead of the generic "could not decode audio" one -- the two
        # failure modes need different messages, the same "ordinary
        # failure vs. a real, actionable limit" distinction this
        # module's own docstring already draws for decode errors vs. a
        # timeout.
        print(f"AUDIO_TOO_LONG:{duration_s}", file=sys.stderr)
        return 2

    sys.stdout.buffer.write(audio.astype("float32").tobytes())
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    _max_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(_run(sys.stdin.buffer.read(), _max_seconds))
