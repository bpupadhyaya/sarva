"""sarva.runtime — shared provider/router wiring for every skin.

The CLI and the server must agree on what "zero-config" means (route to
the offline mock model with no API key) and how local providers get
detected. This is the one place that decides it, so skins never drift out
of sync on availability logic.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from sarva.config import get_env
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva.providers.google_provider import GoogleProvider
from sarva.providers.mock import MockProvider
from sarva.providers.ollama_provider import OllamaProvider, _strip_local_prefix
from sarva.providers.openai_provider import OpenAIProvider
from sarva.providers.registry import Registry, Router, load_routing

_DATA_DIR = Path(__file__).parent / "providers" / "data"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# A real bug found by actually running build_router() immediately
# followed by build_providers() the way sarva.server.app's /chat and
# /ws/chat handlers do it on EVERY request: ollama_reachable() and
# ollama_pulled_models() each independently hit Ollama's real
# /api/tags endpoint, so one request made 2-3 redundant identical
# network calls (build_router's own reachability check + its
# ollama_pulled_models() call, then build_providers' own separate
# reachability check re-deriving what build_router had just computed a
# moment earlier in the same call stack) -- for EVERY request, even
# ones that never end up routed to Ollama at all, since routing hasn't
# been decided yet when these functions run. Confirmed live: pointing
# OLLAMA_HOST at a black-holed address (a realistic firewalled/slow-
# starting-host scenario, not just "nothing listening on localhost")
# measured 0.61s of pure blocking network wait added to a single
# request. Collapsed to one real network call per _OLLAMA_PROBE_
# CACHE_SECONDS window -- short enough that a server actually starting
# up mid-session is still noticed within a couple of requests, long
# enough to absorb the microseconds between build_router() and
# build_providers() (or run_diagnostics()'s own identical pair) within
# one request.
_OLLAMA_PROBE_CACHE_SECONDS = 2.0
_ollama_probe_cache: dict[str, tuple[float, bool, set[str]]] = {}
# A real bug found by a fresh-eyes sweep, one layer beyond the fix
# right above this: the cache's own read-check (`_ollama_probe_cache.
# get(host)`) and its write-back at the end of this function are two
# separate, unsynchronized steps with a real, blocking `httpx.get()`
# in between -- an unguarded check-then-act race, the identical shape
# `agent/loop.py`'s own `spend_lock` was added to close for concurrent
# subagent budget grants, just never applied here. Confirmed live:
# `build_router()`/`build_providers()` are called from every `/chat`,
# `/ws/chat`, `/models`, and `/doctor` handler via `asyncio.to_thread`,
# so genuine concurrent requests (the desktop app's own two independent
# `useEffect` hooks fire `/doctor` and `/models` simultaneously on
# page load, not a contrived scenario) land in separate real OS
# threads -- 6 concurrent `build_router()` calls against a cold cache
# made 6 real network calls, not the 1 the cache exists to guarantee,
# silently reintroducing the exact redundant-network-call regression
# (up to 0.61s of blocking wait per request against a black-holed
# OLLAMA_HOST) the cache above was built to eliminate. A `threading.
# Lock` around the whole check-probe-write span turns this into a real
# single-flight cache: a thread that loses the race to acquire the
# lock reads back the FRESH entry the winner just wrote, instead of
# racing its own redundant probe. `httpx.get`'s own `timeout=0.3`
# bounds how long any thread can be blocked waiting for the lock.
_ollama_probe_lock = threading.Lock()


def _probe_ollama(host: str) -> tuple[bool, set[str]]:
    with _ollama_probe_lock:
        now = time.monotonic()
        cached = _ollama_probe_cache.get(host)
        if cached is not None and now - cached[0] < _OLLAMA_PROBE_CACHE_SECONDS:
            return cached[1], cached[2]
        reachable = False
        pulled: set[str] = set()
        try:
            response = httpx.get(f"{host}/api/tags", timeout=0.3)
            # Reachability means the connection itself succeeded, independent
            # of the response status -- matching the original ollama_reachable
            # semantics exactly (a 4xx/5xx from Ollama still means something
            # answered, just not usefully; only pulled stays empty in that case).
            reachable = True
            response.raise_for_status()
            pulled = {m["name"] for m in response.json().get("models", [])}
        except httpx.HTTPError:
            pass
        except Exception:
            # A real bug found by a fresh-eyes sweep: `except httpx.HTTPError`
            # alone doesn't cover whatever's actually listening at OLLAMA_HOST
            # answering with a 200 whose body isn't the JSON shape Ollama's
            # own `/api/tags` returns -- a real, ordinary condition (a
            # corporate captive portal, a stale/misconfigured reverse proxy,
            # simple port reuse by an unrelated service), not a contrived
            # attack. `response.json()` raises `json.JSONDecodeError` on a
            # non-JSON body; a differently-shaped-but-valid JSON body (a top-
            # level list/string instead of a dict, or a dict entry missing
            # "name") raises `AttributeError`/`KeyError` instead -- neither a
            # subclass of `httpx.HTTPError`. Confirmed live: this crashed
            # `GET /models` and `GET /doctor` with a raw, plain-text 500 (no
            # matching except clause here, and no generic exception handler
            # registered in sarva.server.app besides ConfigError's own),
            # unlike `POST /chat`, which only degrades cleanly to
            # `state=failed` by the accident of `JSONDecodeError` happening to
            # be a `ValueError` subclass that handler's own broad except
            # already covers for an unrelated reason. This probe's entire
            # contract, per its own docstring, is "best-effort" -- treating
            # every other malformed-response shape the identical way
            # `httpx.HTTPError` already is (reachable stays whatever it was
            # set to above, pulled stays empty) is the correct generalization,
            # not enumerating each new exception type reactively as it's
            # found one at a time.
            pass
        _ollama_probe_cache[host] = (now, reachable, pulled)
        return reachable, pulled


def ollama_reachable(host: str = OLLAMA_HOST) -> bool:
    """Best-effort, fast probe — never blocks startup for more than a beat.

    Cached across calls -- see `_probe_ollama`'s own comment above."""
    reachable, _pulled = _probe_ollama(host)
    return reachable


def ollama_pulled_models(host: str = OLLAMA_HOST) -> set[str]:
    """The real set of locally-pulled model tags (e.g. `{"qwen2.5:0.5b"}`),
    queried from the same `/api/tags` endpoint `ollama_reachable()` already
    hits -- empty if unreachable or nothing is pulled yet.

    A real bug this closes, found by actually running Sarva against a
    real local Ollama server with a small model pulled instead of the
    registry's own default `qwen3:8b`: `build_router()` used to mark
    EVERY registered `ollama/*` model as "available" the instant the
    server was merely reachable, regardless of whether that specific
    model tag was actually present -- a real `sarva run`/`sarva chat`
    request routed straight to an unpulled model and failed outright,
    with the zero-config Mock fallback never getting a chance. This
    powers the fix: availability now means "this exact tag is really
    there," not just "some Ollama server answered."

    Cached across calls -- see `_probe_ollama`'s own comment above."""
    _reachable, pulled = _probe_ollama(host)
    return pulled


def _google_key() -> str | None:
    return get_env("GEMINI_API_KEY") or get_env("GOOGLE_API_KEY")


def foundry_checkpoints_dir() -> Path | None:
    """Opt-in only: `SARVA_FOUNDRY_CHECKPOINTS` pointing at a real
    directory of locally-trained checkpoints (see
    `sarva.providers.foundry_provider`). Unset by default -- a plain
    install never scans for or loads anything foundry-related."""
    raw = os.environ.get("SARVA_FOUNDRY_CHECKPOINTS")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def _foundry_extra_installed() -> bool:
    """Cheap probe, same role `ollama_reachable` plays for the local
    Ollama runtime: gates both availability (`build_router`) and actual
    provider construction (`build_providers`) from one source of truth,
    so a model is never marked available with no provider able to serve
    it, or vice versa."""
    try:
        import sarva_foundry  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def build_router() -> Router:
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    available = {"mock"}
    if get_env("ANTHROPIC_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "anthropic"}
    if get_env("OPENAI_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "openai"}
    if _google_key():
        available |= {m.id for m in registry.all() if m.provider == "google"}
    if ollama_reachable():
        pulled = ollama_pulled_models()
        available |= {
            m.id
            for m in registry.all()
            if m.provider == "ollama" and _strip_local_prefix(m.id) in pulled
        }
    fdir = foundry_checkpoints_dir()
    if fdir is not None and _foundry_extra_installed():
        from sarva.providers.foundry_provider import (
            discover_checkpoint_bundles,
            model_info_for_bundle,
        )

        for bundle_name, bundle_path in discover_checkpoint_bundles(fdir).items():
            try:
                info = model_info_for_bundle(bundle_name, bundle_path)
            except (OSError, ValueError, KeyError):
                # A real bug found by actually corrupting a bundle's
                # config.json two ways (malformed JSON, and valid JSON
                # missing the required "max_seq_len" key):
                # model_info_for_bundle() raised an uncaught
                # json.JSONDecodeError/KeyError with nothing here to
                # catch it, crashing every caller of build_router() --
                # every CLI command, /chat, /ws/chat, server startup --
                # not just a request that tried to use that specific
                # checkpoint. discover_checkpoint_bundles only checks
                # that the three bundle files *exist*, never that
                # config.json is actually valid, so this was reachable
                # any time a bundle got corrupted after being
                # discovered. The same "corrupted on-disk state" bug
                # class already fixed for FoundryProvider.__init__
                # (which covers build_providers(), a separate call
                # site) -- just not closed here until now. Skipped
                # (not fatal to every OTHER valid bundle), the same
                # "one bad case shouldn't take down everything" posture
                # FoundryProvider.__init__ and run_benchmark already
                # apply.
                continue
            registry.register(info)
            available.add(info.id)
    return Router(registry, routing, available)


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    ok: bool
    detail: str


def run_diagnostics() -> list[DiagnosticCheck]:
    """Backs `sarva doctor`. One check per condition `build_router`/
    `build_providers` above actually gate availability on -- deliberately
    kept in this same module (reading the same env vars, calling the same
    `ollama_reachable`/`_has_google_key`/`_foundry_extra_installed`
    helpers) so this report can never silently drift out of sync with
    what "available" really means elsewhere in this file. `ok=False`
    means "not configured," not "broken" -- every one of these is a
    genuinely optional provider; a fresh, zero-config install is expected
    to fail every check here and still work fine via the Mock provider."""
    checks: list[DiagnosticCheck] = []

    has_anthropic = bool(get_env("ANTHROPIC_API_KEY"))
    checks.append(
        DiagnosticCheck(
            "Anthropic API key",
            has_anthropic,
            "ANTHROPIC_API_KEY is set"
            if has_anthropic
            else "ANTHROPIC_API_KEY not set -- Claude models unavailable",
        )
    )

    has_openai = bool(get_env("OPENAI_API_KEY"))
    checks.append(
        DiagnosticCheck(
            "OpenAI API key",
            has_openai,
            "OPENAI_API_KEY is set"
            if has_openai
            else "OPENAI_API_KEY not set -- OpenAI models unavailable",
        )
    )

    has_google = bool(_google_key())
    checks.append(
        DiagnosticCheck(
            "Google API key",
            has_google,
            "GEMINI_API_KEY or GOOGLE_API_KEY is set"
            if has_google
            else "GEMINI_API_KEY/GOOGLE_API_KEY not set -- Gemini models unavailable",
        )
    )

    ollama_ok = ollama_reachable()
    if ollama_ok:
        pulled = ollama_pulled_models()
        ollama_detail = (
            f"reachable at {OLLAMA_HOST} -- pulled: {', '.join(sorted(pulled))}"
            if pulled
            else f"reachable at {OLLAMA_HOST}, but no models pulled yet -- "
            "run `ollama pull <model>` before routing to it"
        )
    else:
        ollama_detail = f"not reachable at {OLLAMA_HOST} -- local Ollama models unavailable"
    checks.append(DiagnosticCheck("Ollama (local models)", ollama_ok, ollama_detail))

    foundry_extra = _foundry_extra_installed()
    fdir = foundry_checkpoints_dir()
    if not foundry_extra:
        foundry_ok = False
        foundry_detail = (
            "sarva[foundry] extra not installed -- pip install sarva[foundry] "
            "to serve a locally-trained checkpoint"
        )
    elif fdir is None:
        foundry_ok = False
        foundry_detail = (
            "sarva[foundry] installed, but SARVA_FOUNDRY_CHECKPOINTS is unset "
            "(or isn't a real directory) -- no checkpoints will be discovered"
        )
    else:
        from sarva.providers.foundry_provider import discover_checkpoint_bundles

        bundles = discover_checkpoint_bundles(fdir)
        foundry_ok = bool(bundles)
        foundry_detail = (
            f"{len(bundles)} checkpoint bundle(s) found under {fdir}: {', '.join(sorted(bundles))}"
            if bundles
            else f"SARVA_FOUNDRY_CHECKPOINTS={fdir} has no valid checkpoint bundles"
        )
    checks.append(
        DiagnosticCheck("Foundry (local from-scratch models)", foundry_ok, foundry_detail)
    )

    from sarva.audio import stt_extra_installed, tts_engine_available

    stt_ok = stt_extra_installed()
    checks.append(
        DiagnosticCheck(
            "Speech-to-text (local Whisper)",
            stt_ok,
            "sarva[audio] installed"
            if stt_ok
            else "sarva[audio] extra not installed -- pip install sarva[audio] for local "
            "audio transcription",
        )
    )

    tts_ok = tts_engine_available()
    checks.append(
        DiagnosticCheck(
            "Text-to-speech (local)",
            tts_ok,
            "a local engine (say/espeak) is available"
            if tts_ok
            else "no local text-to-speech engine detected -- `sarva speak` will fail",
        )
    )

    # A real gap found by a fresh-eyes sweep of the two tools just
    # shipped alongside WebSearchTool (RunCodeTool, ImageGenerationTool):
    # every other optional capability this function checks (Ollama,
    # foundry, STT, TTS) gets a `sarva doctor` line reporting whether
    # it's actually usable -- these two didn't, so a user with neither
    # Docker/Podman nor sarva[image] installed had no way to learn
    # `run_code`/`generate_image` would fail before actually trying
    # them mid-conversation. `shutil.which` only, not the stricter
    # `<binary> info` liveness probe `RunCodeTool`'s own
    # `_find_container_runtime` uses before actually running code --
    # "installed" is an honest, reasonable signal for a status display;
    # the tool's own real-daemon check still runs fresh at call time
    # and gives the precise "isn't currently running" error if the
    # daemon happens to be down.
    docker_or_podman = shutil.which("docker") or shutil.which("podman")
    checks.append(
        DiagnosticCheck(
            "Code execution sandbox (Docker/Podman)",
            docker_or_podman is not None,
            f"found at {docker_or_podman}"
            if docker_or_podman
            else "neither docker nor podman found on PATH -- the run_code tool will "
            "return a clear error rather than run code unsandboxed",
        )
    )

    image_extra_installed = importlib.util.find_spec("diffusers") is not None
    has_openai_key = bool(get_env("OPENAI_API_KEY"))
    checks.append(
        DiagnosticCheck(
            "Image generation (local FLUX or OpenAI)",
            image_extra_installed or has_openai_key,
            "sarva[image] installed -- generate_image uses the free local model"
            if image_extra_installed
            else "OPENAI_API_KEY is set -- generate_image will use the paid OpenAI "
            "Images API (sarva[image] not installed)"
            if has_openai_key
            else "sarva[image] extra not installed and OPENAI_API_KEY not set -- "
            "pip install sarva[image] for free local generation",
        )
    )

    return checks


def build_providers() -> dict[str, Any]:
    # Every SDK client is constructed with an EXPLICIT api_key rather than
    # left to the SDK's own os.environ auto-detection -- a key that only
    # exists in sarva.config's saved file (not a real process env var)
    # would otherwise pass every availability check above and then fail
    # to authenticate the moment a real request went out, since the raw
    # SDK constructors never look anywhere but os.environ themselves.
    providers: dict[str, Any] = {"mock": MockProvider()}
    anthropic_key = get_env("ANTHROPIC_API_KEY")
    if anthropic_key:
        import anthropic

        providers["anthropic"] = AnthropicProvider(
            client=anthropic.AsyncAnthropic(api_key=anthropic_key)
        )
    openai_key = get_env("OPENAI_API_KEY")
    if openai_key:
        import openai

        providers["openai"] = OpenAIProvider(client=openai.AsyncOpenAI(api_key=openai_key))
    google_key = _google_key()
    if google_key:
        from google import genai

        providers["google"] = GoogleProvider(client=genai.Client(api_key=google_key))
    if ollama_reachable():
        providers["ollama"] = OllamaProvider(host=OLLAMA_HOST)
    fdir = foundry_checkpoints_dir()
    if fdir is not None and _foundry_extra_installed():
        from sarva.providers.foundry_provider import FoundryProvider, discover_checkpoint_bundles

        if discover_checkpoint_bundles(fdir):
            providers["foundry"] = FoundryProvider(fdir)
    return providers
