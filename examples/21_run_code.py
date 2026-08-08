"""Example 21 — Using a real built-in tool (run_code) with a real model.

Unlike examples 02-04 (which script the mock provider to demonstrate loop
mechanics), this one needs a real model to decide what code to run — so it
requires ANTHROPIC_API_KEY. Run examples 01-04 first if you don't have one
yet.

Also requires Docker or Podman installed and running (both free, open
source) — run_code executes code in a locked-down, network-disabled
container, never on the host directly. If neither is reachable, the tool
itself returns a clear error rather than running code unsandboxed; this
example will print that error rather than crash. See docs/agent-loop.md's
RunCodeTool section for the full isolation details.

run_code is destructive=True (matching run_shell), so this example uses
always_allow to auto-approve — a real interactive session (`sarva chat`/
`sarva run`) prompts for confirmation before every call instead.

Run: ANTHROPIC_API_KEY=sk-... uv run python examples/21_run_code.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import RunCodeTool, always_allow
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
        tools=[RunCodeTool()],
        confirm=always_allow,  # a real session prompts for approval instead
    )

    task = "Use Python to compute the 20th Fibonacci number and tell me the answer."
    async for event in loop.run(task):
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            print(event.event.text, end="", flush=True)
        elif event.type == "tool_started":
            print(f"\n[running {event.call.arguments.get('language')} code in a sandbox]")
        elif event.type == "run_done":
            print()


if __name__ == "__main__":
    asyncio.run(main())
