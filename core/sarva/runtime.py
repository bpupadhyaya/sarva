"""sarva.runtime — shared provider/router wiring for every skin.

The CLI and the server must agree on what "zero-config" means (route to
the offline mock model with no API key) and how local providers get
detected. This is the one place that decides it, so skins never drift out
of sync on availability logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from sarva.config import get_env
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva.providers.google_provider import GoogleProvider
from sarva.providers.mock import MockProvider
from sarva.providers.ollama_provider import OllamaProvider
from sarva.providers.openai_provider import OpenAIProvider
from sarva.providers.registry import Registry, Router, load_routing

_DATA_DIR = Path(__file__).parent / "providers" / "data"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_reachable(host: str = OLLAMA_HOST) -> bool:
    """Best-effort, fast probe — never blocks startup for more than a beat."""
    try:
        httpx.get(f"{host}/api/tags", timeout=0.3)
        return True
    except httpx.HTTPError:
        return False


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
        available |= {m.id for m in registry.all() if m.provider == "ollama"}
    fdir = foundry_checkpoints_dir()
    if fdir is not None and _foundry_extra_installed():
        from sarva.providers.foundry_provider import (
            discover_checkpoint_bundles,
            model_info_for_bundle,
        )

        for bundle_name, bundle_path in discover_checkpoint_bundles(fdir).items():
            info = model_info_for_bundle(bundle_name, bundle_path)
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
    checks.append(
        DiagnosticCheck(
            "Ollama (local models)",
            ollama_ok,
            f"reachable at {OLLAMA_HOST}"
            if ollama_ok
            else f"not reachable at {OLLAMA_HOST} -- local Ollama models unavailable",
        )
    )

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
