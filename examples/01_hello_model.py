"""Example 01 — Hello, model.

The smallest possible use of Sarva: route a request through the registry
and get a response back, with zero configuration (falls back to the offline
mock model if ANTHROPIC_API_KEY isn't set).

Run: uv run python examples/01_hello_model.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import always_allow
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva.providers.base import TextDeltaEvent
from sarva.providers.mock import MockProvider
from sarva.providers.registry import Registry, Router, load_routing

DATA_DIR = Path(__file__).parent.parent / "core" / "sarva" / "providers" / "data"


async def main() -> None:
    registry = Registry.load(DATA_DIR / "models.yaml")
    routing = load_routing(DATA_DIR / "routing.yaml")

    available = {"mock"}
    providers = {"mock": MockProvider()}
    if os.environ.get("ANTHROPIC_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "anthropic"}
        providers["anthropic"] = AnthropicProvider()

    router = Router(registry, routing, available)
    loop = AgentLoop(router=router, providers=providers, tools=[], confirm=always_allow)

    async for event in loop.run("In one sentence, what is Sarva?"):
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            print(event.event.text, end="", flush=True)
        if event.type == "run_done":
            print()
            print(f"\n(state={event.state}, model_calls={event.spend.model_calls})")


if __name__ == "__main__":
    asyncio.run(main())
