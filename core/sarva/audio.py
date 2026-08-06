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

import platform
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

DEFAULT_MACOS_VOICE = "Samantha"

# Mirrors RunShellTool's own timeout (core/sarva/agent/tools.py) -- a real
# bug found by actually running synthesize() against a hung TTS binary
# (a fake `say` that never returns): every subprocess.run call below had
# no `timeout=` at all, unlike the sibling STT decode path
# (_DECODE_TIMEOUT_SECONDS), so a wedged or resource-exhausted OS speech
# engine hung the whole call indefinitely. This module's own docstring
# names the real threat model this guards against -- "an agent speaking
# its own output" -- so `text` here can be arbitrary, potentially huge,
# model-generated content, not just a short human-typed phrase.
_TTS_TIMEOUT_SECONDS = 60

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
    except subprocess.TimeoutExpired as e:
        # A real bug found by actually running synthesize() against a
        # hung TTS binary (a fake `say` that never returns): every
        # subprocess.run call in _synthesize_with_detected_engine had no
        # timeout at all, so a wedged or resource-exhausted OS speech
        # engine hung this call indefinitely with no way to recover --
        # confirmed live. subprocess.run's own `timeout=` already kills
        # and reaps the child on expiry (no manual cleanup needed, the
        # same "TimeoutExpired handles it" discipline the sibling STT
        # decode path already established), so this only needs to turn
        # the raw exception into the same clean RuntimeError every other
        # engine failure below already produces.
        raise RuntimeError(f"text-to-speech engine timed out after {e.timeout}s") from e
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
    except OSError as e:
        # A real bug found by a fresh-eyes sweep: the macOS (`say`) and
        # Linux (`espeak`/`espeak-ng`) branches both pass `text` as a
        # literal argv element to `subprocess.run`, with no size cap
        # anywhere -- unlike the Windows branch, which already writes
        # `text` to a temp file and reads it back via PowerShell's
        # `Get-Content` specifically to avoid interpolating arbitrary
        # text into a command at all. If `text` is large enough to
        # exceed the OS's real `execve()` argument-list limit
        # (`ARG_MAX` -- confirmed on this machine via `getconf ARG_MAX`,
        # 1MB; Linux additionally caps any SINGLE argv string at 128KB
        # via `MAX_ARG_STRLEN`, independent of the total), `Popen.
        # __init__` raises `OSError: [Errno 7] Argument list too long`
        # *before the child process is even spawned* -- neither a
        # `TimeoutExpired` nor a `CalledProcessError`, the only two
        # types this function already catches, so it propagated raw
        # clean past `sarva speak`'s own `except RuntimeError` and
        # crashed with a full Python traceback. Confirmed live: `sarva
        # speak "$(cat transcript.txt)"` on an ordinarily long piece of
        # text (a book chapter, an article, a meeting transcript --
        # `speak`'s own advertised use case, no crafted input needed)
        # hits this once the text approaches the real OS limit. This
        # module's own docstring already names "an agent speaking its
        # own output" as the real threat model `text` can be arbitrary,
        # potentially large content for -- the same reasoning that
        # already motivated this function's own timeout fix.
        raise RuntimeError(f"text-to-speech engine could not be started: {e}") from e


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
                timeout=_TTS_TIMEOUT_SECONDS,
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
                subprocess.run(args, check=True, capture_output=True, timeout=_TTS_TIMEOUT_SECONDS)
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
            subprocess.run(args, check=True, capture_output=True, timeout=_TTS_TIMEOUT_SECONDS)
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


# A real bug found by a fresh-eyes sweep, applying the identical lens
# already used twice over on this project's other module-level caches
# (_probe_ollama, round 128; FoundryProvider's load_checkpoint_bundle,
# round 130): functools.lru_cache (used here previously) only holds its
# own internal lock around the cache dict's read/insert bookkeeping --
# it releases that lock BEFORE calling the wrapped function itself, so
# two threads racing a cold cache both see a miss and both construct a
# real WhisperModel (an expensive load: a weight download on first use,
# then real CTranslate2 model initialization). AudioToTextDegrader.
# degrade() calls transcribe() via asyncio.to_thread, so two concurrent
# voice-message transcriptions -- two users, or two tabs of the same
# user, against a freshly-started `sarva serve` process, both defaulting
# to model_size="tiny" -- genuinely race this on real OS threads.
# Confirmed live: 8 threads synchronized to hit a cold cache at the same
# instant produced 8 real WhisperModel constructions, not the 1 this
# cache exists to guarantee -- defeating the exact "reloading it every
# call would be a real performance regression" guarantee this module's
# own docstring names. Fixed with a manual dict + threading.Lock around
# the whole check-construct-store span (the same shape as the round
# 128/130 fixes), preserving lru_cache's original maxsize=4 LRU-eviction
# behavior via an OrderedDict rather than an unbounded dict.
_whisper_model_lock = threading.Lock()
_whisper_model_cache: OrderedDict[str, object] = OrderedDict()
_WHISPER_MODEL_CACHE_MAXSIZE = 4


def _whisper_model(model_size: str):
    with _whisper_model_lock:
        cached = _whisper_model_cache.get(model_size)
        if cached is not None:
            _whisper_model_cache.move_to_end(model_size)
            return cached

        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _whisper_model_cache[model_size] = model
        if len(_whisper_model_cache) > _WHISPER_MODEL_CACHE_MAXSIZE:
            _whisper_model_cache.popitem(last=False)
        return model


_DECODE_TIMEOUT_SECONDS = 30


def _decode_audio_isolated(audio_bytes: bytes):
    """Runs `faster_whisper.audio.decode_audio()` in an isolated
    subprocess (`sarva._audio_decode_worker`, see its own docstring for
    the full story) rather than in-process. A real bug found by
    directly fuzzing a real WAV file: `decode_audio()` uses the same
    PyAV/libavcodec call that crashed the video degrader with a native
    SIGBUS -- several fuzzed variants killed the whole process the
    identical way, confirmed directly, not assumed from the video fix
    alone. A crash here now only kills this subprocess; the timeout
    (mirroring `RunShellTool`'s own fix for a hung/resource-exhausting
    subprocess) is handled by `subprocess.run` itself, which kills and
    reaps the child on expiry -- no manual cleanup needed the way the
    asyncio-based fixes elsewhere in this project require.

    The duration cap is passed to the worker as an argument and enforced
    THERE, before the decoded array is ever written back to this
    process via stdout -- see `_audio_decode_worker`'s own docstring for
    the real, measured parent-process memory-exhaustion bug this closes
    (checking the cap only after `capture_output=True` had already
    brought the whole array back into this long-lived process was too
    late to protect it at all).

    Raises `RuntimeError` (never propagates a raw PyAV/ctranslate2
    exception, and never lets a native crash escape uncaught) on any
    decode failure -- ordinary or a native crash, deliberately
    indistinguishable to the caller, since both mean the same thing:
    "couldn't safely decode this audio." """
    import subprocess
    import sys

    import numpy as np

    try:
        result = subprocess.run(
            [sys.executable, "-m", "sarva._audio_decode_worker", str(_MAX_TRANSCRIBE_SECONDS)],
            input=audio_bytes,
            capture_output=True,
            timeout=_DECODE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("audio decode timed out") from e

    # A real bug found by a fresh-eyes sweep, the identical "empty is a
    # real, known value that truthiness silently treats as absent/
    # failed" shape already found and fixed once in this same audio
    # subsystem, one file over (AudioToTextDegrader's `duration_s or
    # ...`): `not result.stdout` treats a genuinely valid, zero-frame
    # WAV -- e.g. a voice-message client where the user tapped-and-
    # immediately-released record, a plausible real artifact, not
    # contrived -- as a decode FAILURE, even though the worker exits 0
    # and writes 0 bytes to stdout exactly per its own documented
    # contract ("writes the resulting float32 samples raw to stdout on
    # success" -- zero samples is still a success). Confirmed live: a
    # real, valid zero-frame WAV decoded successfully by the worker
    # (returncode=0, empty stderr) still raised "could not decode
    # audio" in the parent, misleadingly reported past a completely
    # successful decode. The worker's own returncode is already the
    # correct, sufficient success/failure signal -- stdout emptiness on
    # a returncode==0 success is meaningful data (zero samples), not a
    # second failure indicator to OR in.
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        if detail.startswith("AUDIO_TOO_LONG:"):
            duration_s = float(detail.removeprefix("AUDIO_TOO_LONG:"))
            raise RuntimeError(
                f"audio too long to transcribe safely: {duration_s:.0f}s exceeds the "
                f"{_MAX_TRANSCRIBE_SECONDS}s cap (whisper's own feature extraction scales "
                "memory linearly with duration -- confirmed live at roughly 100MB per minute)"
            )
        raise RuntimeError(f"could not decode audio{f': {detail}' if detail else ''}")
    return np.frombuffer(result.stdout, dtype=np.float32)


# A real bug found by actually generating an ordinary, non-adversarial
# 30-minute audio file (a plain sine tone, ffmpeg-encoded, 14.4MB mp3 --
# nothing crafted or malicious) and measuring peak memory with real
# numbers, not an estimate: transcribing it drove peak RSS to ~3.0 GB,
# roughly linear at ~100MB per minute of audio. Root cause: faster-
# whisper's own `FeatureExtractor` computes the log-mel spectrogram for
# the ENTIRE decoded audio array in one call -- a full-length STFT,
# materializing overlapping-frame, FFT, magnitude, and mel arrays all
# at once -- before its own 30-second-window inference loop even
# begins. Architecturally the identical shape as the video degrader's
# just-fixed memory bug (materialize everything, then process a piece
# at a time), just one layer further along the same "real user-
# supplied media attachment" pipeline, and inside a vendored dependency
# this project doesn't own the internals of -- so the fix has to be a
# duration cap at this call site, not a change to faster-whisper
# itself. A completely ordinary, real-world audio length (a 2-hour
# recording is an entirely normal meeting/podcast/lecture attachment)
# would extrapolate to ~12 GB, easily OOM-killing a typical 8-16 GB
# host, with nothing upstream bounding it -- locally path/data-sourced
# audio blocks bypass `fetch.py`'s own 20MB URL-fetch cap entirely,
# since that only applies to `url`-sourced blocks.
_MAX_TRANSCRIBE_SECONDS = 600  # 10 minutes -- ~1GB worst case at the measured rate


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
    larger `model_size` explicitly.

    The actual audio decode (not the whisper model inference itself)
    runs in an isolated subprocess -- see `_decode_audio_isolated`'s own
    docstring for the real, confirmed bug this closes. Only the risky
    native-decode step is isolated, not the (expensive to reload)
    whisper model itself, which stays cached in-process via
    `_whisper_model`'s own lock-guarded cache (see its own comment for
    why this is no longer a plain `functools.lru_cache`).

    Raises `RuntimeError` if the decoded audio exceeds
    `_MAX_TRANSCRIBE_SECONDS` -- see this module's own comment just
    above, and `_audio_decode_worker`'s own docstring, for the real,
    measured memory-exhaustion bug this closes. The cap is actually
    *enforced* inside the isolated worker subprocess, before the
    decoded array is ever sent back to this process -- `_decode_
    audio_isolated` already raises with this same message for a real
    over-cap file, so the check just below is unreachable defense-in-
    depth for the ordinary case, kept in case a future caller ever
    reaches `_decode_audio_isolated` some other way. `AudioToTextDegrader`
    already catches every `RuntimeError` this function can raise (decode
    failure, decode timeout, and now this) identically, falling back to
    the honest metadata-only report -- "too long to safely transcribe"
    needs no special handling beyond what "couldn't decode this"
    already gets."""
    if not stt_extra_installed():
        raise ImportError(
            "faster-whisper is not installed -- pip install sarva[audio] for local speech-to-text"
        )
    audio_array = _decode_audio_isolated(audio_bytes)
    duration_s = len(audio_array) / 16000  # _decode_audio_isolated always resamples to 16kHz
    if duration_s > _MAX_TRANSCRIBE_SECONDS:
        raise RuntimeError(
            f"audio too long to transcribe safely: {duration_s:.0f}s exceeds the "
            f"{_MAX_TRANSCRIBE_SECONDS}s cap (whisper's own feature extraction scales "
            "memory linearly with duration -- confirmed live at roughly 100MB per minute)"
        )
    model = _whisper_model(model_size)
    segments, _info = model.transcribe(audio_array)
    return " ".join(segment.text.strip() for segment in segments).strip()
