"""Example 05 — Using a real built-in tool (web_fetch) with a real model.

Unlike examples 02-04 (which script the mock provider to demonstrate loop
mechanics), this one needs a real model to decide when to fetch — so it
requires ANTHROPIC_API_KEY. Run examples 01-04 first if you don't have one
yet; this is the one that shows the whole stack working together.

Run: ANTHROPIC_API_KEY=sk-... uv run python examples/05_web_fetch.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import WebFetchTool, always_allow
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva.providers.base import TextDeltaEvent
from sarva.providers.registry import Registry, Router, load_routing

DATA_DIR = Path(__file__).parent.parent / "core" / "sarva" / "providers" / "data"


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example (see examples/01-04 for offline demos).")
        sys.exit(1)

    registry = Registry.load(DATA_DIR / "models.yaml")
    routing = load_routing(DATA_DIR / "routing.yaml")
    available = {m.id for m in registry.all() if m.provider == "anthropic"}
    router = Router(registry, routing, available)

    loop = AgentLoop(
        router=router,
        providers={"anthropic": AnthropicProvider()},
        tools=[WebFetchTool()],
        confirm=always_allow,  # web_fetch is non-destructive, never gated anyway
    )

    task = "Fetch https://example.com and tell me the page's title in one sentence."
    async for event in loop.run(task):
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            print(event.event.text, end="", flush=True)
        elif event.type == "tool_started":
            print(f"\n[fetching {event.call.arguments.get('url')}]")
        elif event.type == "run_done":
            print()


if __name__ == "__main__":
    asyncio.run(main())
