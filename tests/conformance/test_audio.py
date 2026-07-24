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

import shutil
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
