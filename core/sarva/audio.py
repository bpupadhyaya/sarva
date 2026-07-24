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

Both non-Windows TTS branches are verified against real installed
binaries in this environment, not just written to documented CLI
shapes: the `say` branch runs unconditionally on real macOS; the
`espeak-ng` branch was verified too, by installing it (`brew install
espeak-ng`) and hiding `say` specifically in a dedicated test so the
espeak code path actually runs for real (macOS's own Darwin branch
would otherwise always win).

**Windows now has a real engine too**: `System.Speech.Synthesis`
(SAPI), reached via PowerShell — no third-party install needed, since
it ships as part of every desktop Windows .NET Framework install, the
same "already on the machine" bar `say`/`espeak-ng` were picked for.
The text to speak is written to a temp file and read back inside the
PowerShell script via `Get-Content`, deliberately never interpolated
into the command string itself — the same reason `subprocess.run`'s
non-Windows branches already pass `text` as a separate argv element
rather than building a shell string: arbitrary model-produced text
(this function's whole reason to exist is an agent speaking its own
output) must never be able to break out of a command's syntax.

This project has no Windows machine to develop against directly — the
same honest limitation this module's docstring named before this
change. What actually verifies this branch is a real `windows-latest`
GitHub Actions runner (see `.github/workflows/ci.yml`'s
`windows-audio` job), running this exact code path end to end on
genuine Windows, not a mock or a `platform.system()` monkeypatch
standing in for one.
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

# Deliberately parameterized, never string-formatted with `text` itself --
# `text` reaches this script only via `Get-Content` on a temp file passed as
# a separate argv element, so arbitrary (e.g. model-produced) text can never
# be interpreted as PowerShell syntax.
_WINDOWS_TTS_SCRIPT = """
param(
    [Parameter(Mandatory=$true)][string]$TextPath,
    [Parameter(Mandatory=$true)][string]$OutPath,
    [string]$VoiceName
)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($VoiceName) {
    $synth.SelectVoice($VoiceName)
}
$synth.SetOutputToWaveFile($OutPath)
$text = Get-Content -Path $TextPath -Raw -Encoding UTF8
$synth.Speak($text)
$synth.Dispose()
"""


def tts_engine_available() -> bool:
    """Best-effort probe, same role `ollama_reachable`/
    `_foundry_extra_installed` play elsewhere in this project: one
    source of truth backing both `sarva doctor` and `synthesize()`'s own
    error path, so a check can never claim availability `synthesize()`
    itself would then fail to honor."""
    if platform.system() == "Darwin":
        return shutil.which("say") is not None
    if platform.system() == "Windows":
        return shutil.which("powershell") is not None or shutil.which("pwsh") is not None
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def synthesize(text: str, voice: str | None = None) -> bytes:
    """Real local text-to-speech, returned as WAV bytes. Raises
    RuntimeError with a clear, actionable message if no engine is
    detected on this platform, or if a detected engine actually runs
    but fails (e.g. an unrecognized `voice` -- a real, reproducible
    case: `espeak-ng` genuinely exits 1 for an unknown voice name,
    confirmed directly against the real installed binary in this
    environment) — never fabricates silent or empty audio as a
    fallback, the same no-fabrication discipline the multimodal
    degraders already apply to what they report, not what they
    synthesize."""
    try:
        return _synthesize_with_detected_engine(text, voice)
    except subprocess.CalledProcessError as e:
        # A real bug found by actually running `synthesize(text,
        # voice="bogus")` against the real espeak-ng branch: the raw
        # CalledProcessError propagated uncaught, and the CLI's `speak`
        # command only ever caught RuntimeError -- a raw Python
        # traceback instead of a clean, actionable message. `e.stderr`
        # carries the engine's own real diagnostic (e.g. "Error: The
        # specified espeak-ng voice does not exist."), decoded and
        # included here rather than dropped, since it's the one piece
        # of information that actually explains what went wrong.
        stderr = e.stderr.decode(errors="replace").strip() if e.stderr else ""
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"text-to-speech engine failed{detail}") from e


def _synthesize_with_detected_engine(text: str, voice: str | None) -> bytes:
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

    if platform.system() == "Windows":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell:
            with tempfile.TemporaryDirectory() as tmp:
                text_path = Path(tmp) / "speech_input.txt"
                out_path = Path(tmp) / "speech.wav"
                script_path = Path(tmp) / "synthesize.ps1"
                text_path.write_text(text, encoding="utf-8")
                script_path.write_text(_WINDOWS_TTS_SCRIPT, encoding="utf-8")
                args = [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    "-TextPath",
                    str(text_path),
                    "-OutPath",
                    str(out_path),
                ]
                if voice:
                    args += ["-VoiceName", voice]
                subprocess.run(args, check=True, capture_output=True)
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
        "no local text-to-speech engine detected -- macOS's `say`, "
        "Linux's `espeak`/`espeak-ng`, or Windows's PowerShell "
        "(System.Speech/SAPI) is required."
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
