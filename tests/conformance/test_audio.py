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

import pytest
from sarva.audio import (
    stt_extra_installed,
    synthesize,
    transcribe,
    tts_engine_available,
)

_needs_tts = pytest.mark.skipif(not tts_engine_available(), reason="no local TTS engine detected")


@_needs_tts
def test_synthesize_produces_real_nonempty_wav_bytes():
    audio_bytes = synthesize("testing one two three")
    assert audio_bytes.startswith(b"RIFF")
    assert b"WAVE" in audio_bytes[:16]
    assert len(audio_bytes) > 1000  # a real utterance, not an empty/near-empty file


@_needs_tts
def test_synthesize_with_default_macos_voice_produces_full_length_audio():
    # Regression pin for a real bug found while building this: macOS
    # `say`'s own DEFAULT voice (no -v) produced near-silent,
    # sub-10-millisecond output for real text in this environment --
    # confirmed with `afinfo`, not assumed. synthesize() must always
    # pass an explicit voice to avoid silently regressing into that.
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
