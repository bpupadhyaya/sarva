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


def build_router() -> Router:
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    available = {"mock"}
    if os.environ.get("ANTHROPIC_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "anthropic"}
    if os.environ.get("OPENAI_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "openai"}
    if ollama_reachable():
        available |= {m.id for m in registry.all() if m.provider == "ollama"}
    return Router(registry, routing, available)


def build_providers() -> dict[str, Any]:
    providers: dict[str, Any] = {"mock": MockProvider()}
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["anthropic"] = AnthropicProvider()
    if os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = OpenAIProvider()
    if ollama_reachable():
        providers["ollama"] = OllamaProvider(host=OLLAMA_HOST)
    return providers
