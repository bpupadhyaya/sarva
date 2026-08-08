"""Example 22 — Using a real built-in tool (generate_image) with a real model.

Unlike examples 02-04 (which script the mock provider to demonstrate loop
mechanics), this one needs a real model to decide what to generate — so it
requires ANTHROPIC_API_KEY. Run examples 01-04 first if you don't have one
yet.

Free by default: generate_image uses a local, open-weight model
(black-forest-labs/FLUX.1-schnell, genuinely Apache-2.0) via the optional
`sarva[image]` extra (`pip install sarva[image]` or `uv sync --extra image`)
-- a real download and real local compute, slower without a GPU. If that
extra isn't installed but OPENAI_API_KEY is also set, the tool falls back to
the paid OpenAI Images API instead. If neither is available, the tool
returns a clear error rather than silently failing; this example will print
that error rather than crash. See docs/agent-loop.md's ImageGenerationTool
section for the full design.

generate_image is destructive=True (it writes a file, and the paid fallback
has a real cost), so this example uses always_allow to auto-approve -- a
real interactive session (`sarva chat`/`sarva run`) prompts for confirmation
before every call instead.

Run: ANTHROPIC_API_KEY=sk-... uv run python examples/22_generate_image.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import ImageGenerationTool, always_allow
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
        tools=[ImageGenerationTool()],
        confirm=always_allow,  # a real session prompts for approval instead
        workdir=str(Path(__file__).parent),
    )

    task = "Generate an image of a red circle on a white background and save it as circle.png."
    async for event in loop.run(task):
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            print(event.event.text, end="", flush=True)
        elif event.type == "tool_started":
            print(f"\n[generating image: {event.call.arguments.get('prompt')!r}]")
        elif event.type == "run_done":
            print()


if __name__ == "__main__":
    asyncio.run(main())
