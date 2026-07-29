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
"""

from __future__ import annotations

import io
import sys


def _run(raw: bytes) -> int:
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(io.BytesIO(raw), sampling_rate=16000)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    sys.stdout.buffer.write(audio.astype("float32").tobytes())
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(_run(sys.stdin.buffer.read()))
