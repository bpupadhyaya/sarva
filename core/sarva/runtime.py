"""sarva.runtime — shared provider/router wiring for every skin.

The CLI and the server must agree on what "zero-config" means (route to
the offline mock model with no API key) and how local providers get
detected. This is the one place that decides it, so skins never drift out
of sync on availability logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

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


def _has_google_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


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
    if os.environ.get("ANTHROPIC_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "anthropic"}
    if os.environ.get("OPENAI_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "openai"}
    if _has_google_key():
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


def build_providers() -> dict[str, Any]:
    providers: dict[str, Any] = {"mock": MockProvider()}
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["anthropic"] = AnthropicProvider()
    if os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = OpenAIProvider()
    if _has_google_key():
        providers["google"] = GoogleProvider()
    if ollama_reachable():
        providers["ollama"] = OllamaProvider(host=OLLAMA_HOST)
    fdir = foundry_checkpoints_dir()
    if fdir is not None and _foundry_extra_installed():
        from sarva.providers.foundry_provider import FoundryProvider, discover_checkpoint_bundles

        if discover_checkpoint_bundles(fdir):
            providers["foundry"] = FoundryProvider(fdir)
    return providers
