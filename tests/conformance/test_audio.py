"""Conformance tests for sarva.audio -- local, offline speech
transcription (STT) and synthesis (TTS). Closes T2's own definition of
done ("audio in/out (local Whisper/TTS)"), confirmed unmet before
starting: `grep -rln "whisper\\|Whisper\\|TTS" core/sarva` returned
nothing, and AudioToTextDegrader always reported "could not be
transcribed" regardless of input.

Real engines, real round trips where the local platform supports them
(skipped, not faked, where it doesn't) -- see test_degraders.py's
end-to-end AudioToTextDegrader test for the fullest version of this
same proof."""

from __future__ import annotations

import io
import math
import random
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from sarva.audio import (
    stt_extra_installed,
    synthesize,
    transcribe,
    tts_engine_available,
)

_needs_tts = pytest.mark.skipif(not tts_engine_available(), reason="no local TTS engine detected")
_has_espeak = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


@_needs_tts
def test_synthesize_produces_real_nonempty_wav_bytes():
    audio_bytes = synthesize("testing one two three")
    assert audio_bytes.startswith(b"RIFF")
    assert b"WAVE" in audio_bytes[:16]
    assert len(audio_bytes) > 1000  # a real utterance, not an empty/near-empty file


@_needs_tts
def test_synthesize_with_default_voice_produces_full_length_audio():
    # Regression pin for a real bug found while building this: macOS
    # `say`'s own DEFAULT voice (no -v) produced near-silent,
    # sub-10-millisecond output for real text in this environment --
    # confirmed with `afinfo`, not assumed. synthesize() must always
    # pass an explicit voice to avoid silently regressing into that.
    # Generic enough to double as the equivalent regression check for
    # Windows SAPI's own default voice, exercised for real on the
    # windows-latest CI runner (see .github/workflows/ci.yml).
    short = synthesize("hello")
    longer = synthesize("this is a substantially longer sentence than the other one")
    assert len(longer) > len(short)
    assert len(short) > 1000


@_needs_tts
@pytest.mark.skipif(not stt_extra_installed(), reason="sarva[audio] (faster-whisper) not installed")
def test_synthesize_then_transcribe_round_trips_real_words():
    # Deliberately not asserting on "Sarva" itself -- a real finding
    # while writing these tests is that the "tiny" Whisper model
    # sometimes mishears it as "Serve a," an uncommon-proper-noun
    # near-miss, not a bug in this round trip. Common words only, to
    # keep this test a reliable signal rather than occasionally flaky.
    audio_bytes = synthesize("the assistant can now hear and speak")
    text = transcribe(audio_bytes)
    lowered = text.lower()
    assert "hear" in lowered
    assert "speak" in lowered


@pytest.mark.skipif(stt_extra_installed(), reason="this test needs the extra NOT installed")
def test_transcribe_raises_a_clear_error_without_the_extra():
    with pytest.raises(ImportError, match="sarva\\[audio\\]"):
        transcribe(b"irrelevant, never reached")


def test_synthesize_raises_a_clear_runtime_error_with_no_engine(monkeypatch):
    import sarva.audio as audio_module

    monkeypatch.setattr(audio_module.platform, "system", lambda: "Nonexistent")
    monkeypatch.setattr(audio_module.shutil, "which", lambda *_: None)

    with pytest.raises(RuntimeError, match="no local text-to-speech engine"):
        synthesize("this should fail")


@pytest.mark.skipif(not _has_espeak, reason="espeak/espeak-ng not installed")
def test_synthesize_falls_back_to_espeak_when_say_is_unavailable(monkeypatch):
    # On real macOS the Darwin branch (say) always wins, so this is the
    # only way to exercise the espeak path for real in this environment:
    # hide `say` specifically (still resolving every other command,
    # including the real installed espeak-ng) rather than faking the
    # whole platform, so the actual espeak subprocess call runs for
    # real -- not mocked, not skipped, genuinely verified against a
    # real installed binary, the same bar the macOS `say` path already
    # cleared.
    import sarva.audio as audio_module

    real_which = shutil.which
    monkeypatch.setattr(
        audio_module.shutil, "which", lambda cmd: None if cmd == "say" else real_which(cmd)
    )

    audio_bytes = synthesize("the quick brown fox")

    assert audio_bytes.startswith(b"RIFF")
    assert b"WAVE" in audio_bytes[:16]
    assert len(audio_bytes) > 1000


@pytest.mark.skipif(not _has_espeak, reason="espeak/espeak-ng not installed")
def test_synthesize_raises_a_clean_runtime_error_when_the_engine_itself_fails(monkeypatch):
    # A real bug found by actually running this against the real
    # installed espeak-ng binary: an unrecognized --voice name makes
    # espeak-ng genuinely exit 1 ("Error: The specified espeak-ng voice
    # does not exist."), and the raw subprocess.CalledProcessError
    # propagated uncaught -- a bare Python traceback instead of the
    # same clean RuntimeError the "no engine at all" case already
    # raises, and which the CLI's `speak` command specifically catches.
    import sarva.audio as audio_module

    real_which = shutil.which
    monkeypatch.setattr(
        audio_module.shutil, "which", lambda cmd: None if cmd == "say" else real_which(cmd)
    )

    with pytest.raises(RuntimeError, match="text-to-speech engine failed") as excinfo:
        synthesize("hello", voice="totally-bogus-voice-name-xyz")

    # The engine's own real diagnostic must survive into the message --
    # the one piece of information that actually explains what broke.
    assert "does not exist" in str(excinfo.value)
    assert not isinstance(excinfo.value, subprocess.CalledProcessError)


def test_windows_branch_never_puts_raw_text_on_the_command_line(monkeypatch):
    # This project has no Windows machine to run the real SAPI branch
    # against locally -- the windows-latest CI job is what verifies it
    # actually speaks (see .github/workflows/ci.yml's windows-audio
    # job). What CAN be verified here, on any OS, hermetically: the
    # structural safety property that makes the branch safe to call
    # with arbitrary (e.g. model-produced) text in the first place --
    # `text` never becomes part of the subprocess argv or the
    # PowerShell script content, only the content of a temp file read
    # back via `Get-Content`, so it can never be interpreted as
    # PowerShell syntax no matter what it contains.
    import sarva.audio as audio_module

    dangerous_text = '"; Remove-Item -Recurse -Force C:\\ ; Write-Host "pwned'
    captured = {}

    monkeypatch.setattr(audio_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        audio_module.shutil, "which", lambda cmd: "powershell.exe" if cmd == "powershell" else None
    )

    def fake_run(args, check, capture_output):
        # Inspect the temp text/script files *inside* the fake call --
        # synthesize()'s own TemporaryDirectory is cleaned up as soon as
        # it returns, so this is the only point they're on disk.
        captured["args"] = args
        text_path = Path(args[args.index("-TextPath") + 1])
        script_path = Path(args[args.index("-File") + 1])
        captured["text_file_content"] = text_path.read_text(encoding="utf-8")
        captured["script_content"] = script_path.read_text(encoding="utf-8")
        out_path = Path(args[args.index("-OutPath") + 1])
        out_path.write_bytes(b"RIFF....WAVEfake")

        class _Result:
            pass

        return _Result()

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    result = synthesize(dangerous_text)

    assert result == b"RIFF....WAVEfake"
    args = captured["args"]
    assert all(dangerous_text not in str(a) for a in args)
    assert captured["text_file_content"] == dangerous_text
    assert dangerous_text not in captured["script_content"]
    assert "Get-Content" in captured["script_content"]


@pytest.mark.skipif(
    not (_has_espeak and stt_extra_installed()),
    reason="needs espeak/espeak-ng and sarva[audio] (faster-whisper)",
)
def test_espeak_synthesis_then_transcribe_round_trips_real_words(monkeypatch):
    import sarva.audio as audio_module

    real_which = shutil.which
    monkeypatch.setattr(
        audio_module.shutil, "which", lambda cmd: None if cmd == "say" else real_which(cmd)
    )

    audio_bytes = synthesize("the quick brown fox jumps over the lazy dog")
    text = transcribe(audio_bytes)
    lowered = text.lower()
    assert "quick brown fox" in lowered
    assert "lazy dog" in lowered


def _synthetic_wav_bytes(duration_s: float = 1.0, freq: int = 440, rate: int = 16000) -> bytes:
    """A real, tiny, stdlib-decodable WAV -- a pure sine wave encoded
    with the stdlib `wave` module itself, not a fixture file checked
    into the repo or dependent on any locally installed TTS engine, so
    this test's own base audio is portable and regenerable on any
    platform CI happens to run on (the same discipline
    test_degraders.py's `_synthetic_video_bytes` already established
    for the video degrader's own SIGBUS regression test)."""
    n = int(duration_s * rate)
    samples = [int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


def _fuzzed_wav_bytes(seed: int, trials: int) -> bytes:
    """Deterministically regenerates one specific fuzzed variant from a
    fixed-seed random-byte-flip fuzzing run over a real, valid WAV --
    the exact methodology (and the same underlying PyAV/libavcodec
    dependency) that found a real SIGBUS crash in this project's own
    video degrader (see `_video_worker.py`'s docstring) and,
    independently, in `faster_whisper.audio.decode_audio()` (see
    `sarva._audio_decode_worker`'s own docstring): this exact seed and
    trial count reliably reproduces one of several confirmed
    process-killing fuzzed variants found by mutating this base WAV's
    header bytes, confirmed directly against the real installed
    `av`/libavcodec build in this environment, not assumed. Regenerating
    the exact byte sequence here (rather than checking in an opaque
    binary fixture) keeps the crash trigger self-documenting and
    reproducible against whatever PyAV build is actually installed when
    this test runs."""
    rng = random.Random(seed)
    raw = bytearray(_synthetic_wav_bytes())
    fuzzed = raw
    for _ in range(trials):
        fuzzed = bytearray(raw)
        n_flips = rng.randint(1, 20)
        for _ in range(n_flips):
            idx = rng.randrange(min(len(fuzzed), 2000))
            fuzzed[idx] = rng.randrange(256)
    return bytes(fuzzed)


@pytest.mark.skipif(not stt_extra_installed(), reason="sarva[audio] (faster-whisper) not installed")
def test_transcribe_survives_a_native_decoder_crash_not_just_a_python_exception():
    # A real bug found by directly fuzzing a valid WAV (random byte
    # flips) and running the exact faster_whisper.audio.decode_audio()
    # call this module used to make in-process: some fuzzed variants
    # don't raise a Python exception at all, they kill the process
    # outright with a real SIGBUS -- the identical native-crash bug
    # class already found and fixed in the video degrader
    # (sarva.multimodal.degraders._video_worker), confirmed here
    # independently rather than merely assumed to apply by analogy,
    # since faster-whisper uses the same PyAV/libavcodec dependency for
    # its own audio decoding. Fixed by running the actual decode in an
    # isolated subprocess (sarva._audio_decode_worker) -- a crash there
    # only kills that subprocess. This test's own success (it completes
    # at all, in the same pytest process that started it) is the actual
    # proof the isolation works, not just an assertion on the result.
    fuzzed = _fuzzed_wav_bytes(seed=55, trials=7)

    with pytest.raises(RuntimeError, match="could not decode audio"):
        transcribe(fuzzed)
