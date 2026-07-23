"""sarva.audio — local, offline speech: transcription (STT) and
synthesis (TTS). Closes T2's own definition-of-done ("audio in/out
(local Whisper/TTS)"), the one modality-completeness promise this
project's own roadmap named but never delivered on — confirmed before
starting by `grep -rln "whisper\\|Whisper\\|TTS" core/sarva` returning
nothing, and by `AudioToTextDegrader`'s own body, which never actually
transcribed anything, always reporting "could not be transcribed."

Deliberately different substrate choices for the two directions, each
picked for the same reason: avoid a heavy dependency where a real,
already-installed OS-native tool does the job.

- **TTS shells out to the operating system's own bundled speech
  synthesizer** (macOS `say`, Linux `espeak-ng`/`espeak`) rather than a
  Python TTS library. `pyttsx3`, the most common cross-platform
  wrapper, was tried and rejected: installing it on macOS pulled in the
  ENTIRE `pyobjc` framework suite (100+ separate packages) just to
  reach the same `say` command this module now calls directly, for a
  fraction of the dependency footprint.
- **STT uses `faster-whisper`** (CTranslate2-based) — a real, warranted
  new dependency, since there's no OS-native local speech recognizer to
  shell out to the way TTS has one. Its hard dependencies
  (`ctranslate2`, `huggingface-hub`, `tokenizers`, `onnxruntime`, `av`,
  `tqdm`) pull in no torch — `av` is already a `core` hard dependency
  (the video degrader), so this genuinely adds one new lightweight
  package tree, not a second heavy ML stack alongside `sarva[foundry]`.

**A real bug found empirically while building this, not a hypothetical:**
macOS `say`'s own DEFAULT voice (invoked with no `-v`) produced
near-silent, sub-10-millisecond output for real text in this
environment — confirmed directly with `afinfo`, not assumed — while an
explicitly named, always-bundled voice (`Samantha`) produced correct,
full-length audio for the identical text. `synthesize()` below always
passes an explicit voice for exactly this reason; letting `say` resolve
its own default silently produced near-silent WAV files that would
have looked like a working feature until someone actually listened to
one.
"""

from __future__ import annotations

import io
import platform
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

DEFAULT_MACOS_VOICE = "Samantha"


def tts_engine_available() -> bool:
    """Best-effort probe, same role `ollama_reachable`/
    `_foundry_extra_installed` play elsewhere in this project: one
    source of truth backing both `sarva doctor` and `synthesize()`'s own
    error path, so a check can never claim availability `synthesize()`
    itself would then fail to honor."""
    if platform.system() == "Darwin":
        return shutil.which("say") is not None
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def synthesize(text: str, voice: str | None = None) -> bytes:
    """Real local text-to-speech, returned as WAV bytes. Raises
    RuntimeError with a clear, actionable message if no engine is
    detected on this platform — never fabricates silent or empty audio
    as a fallback, the same no-fabrication discipline the multimodal
    degraders already apply to what they report, not what they
    synthesize."""
    if platform.system() == "Darwin" and shutil.which("say"):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "speech.wav"
            subprocess.run(
                [
                    "say",
                    "-v",
                    voice or DEFAULT_MACOS_VOICE,
                    "--file-format=WAVE",
                    "--data-format=LEI16@22050",
                    "-o",
                    str(out_path),
                    text,
                ],
                check=True,
                capture_output=True,
            )
            return out_path.read_bytes()

    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if engine:
        # espeak/espeak-ng writes a real WAV directly via -w, no format
        # flags needed the way `say` requires them.
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "speech.wav"
            args = [engine, "-w", str(out_path)]
            if voice:
                args += ["-v", voice]
            args.append(text)
            subprocess.run(args, check=True, capture_output=True)
            return out_path.read_bytes()

    raise RuntimeError(
        "no local text-to-speech engine detected -- macOS's `say` or "
        "Linux's `espeak`/`espeak-ng` is required. Windows has no "
        "supported engine yet: a real, open gap, not silently assumed "
        "away (this project has no Windows machine to verify a "
        "PowerShell-based implementation against, the same honest "
        "limitation named for the desktop sidecar's own Windows gap)."
    )


def stt_extra_installed() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=4)
def _whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes, model_size: str = "tiny") -> str:
    """Real local speech-to-text via `faster-whisper`. Raises
    `ImportError` (with a clear `pip install sarva[audio]` message) if
    the extra isn't installed -- callers that want a graceful fallback
    instead (e.g. `AudioToTextDegrader`) catch that explicitly, the same
    optional-dependency pattern `FoundryProvider` already uses for its
    own optional `torch` dependency.

    `model_size` defaults to `"tiny"` (~75MB download, cached by
    `huggingface_hub` after the first call) -- the same "commodity
    substrate, sensible default, no fabricated benchmark claiming a
    bigger model is `sarva[audio]`'s default" discipline this project
    applies elsewhere; a caller who wants better accuracy passes a
    larger `model_size` explicitly."""
    if not stt_extra_installed():
        raise ImportError(
            "faster-whisper is not installed -- pip install sarva[audio] for local speech-to-text"
        )
    model = _whisper_model(model_size)
    segments, _info = model.transcribe(io.BytesIO(audio_bytes))
    return " ".join(segment.text.strip() for segment in segments).strip()
