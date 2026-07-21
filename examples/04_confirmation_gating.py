"""Example 04 — Confirmation gating for destructive tools.

Tools declare `destructive=True`; the loop — not the tool — decides whether
to pause for approval. This example uses a scripted "always deny" policy to
show a destructive call being blocked, then the loop continuing cleanly
with a "declined" result the model can react to.

Run: uv run python examples/04_confirmation_gating.py
"""

from __future__ import annotations

import asyncio

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import ToolContext
from sarva.multimodal.content import Modality, TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.base import ModelCapabilities, ModelCost, ModelInfo, ToolSpec
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass


class DeleteEverythingTool:
    spec = ToolSpec(
        name="delete_everything",
        description="Pretend to delete something irreversible.",
        input_schema={"type": "object", "properties": {}},
        destructive=True,  # <- this is what triggers the confirmation gate
    )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResultBlock:
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text="deleted!")])


async def deny_everything(call: ToolCallBlock) -> bool:
    print(f"[confirmation] denying {call.name}({call.arguments})")
    return False


async def main() -> None:
    call = ToolCallBlock(id="d1", name="delete_everything", arguments={})
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(text="Understood, I won't delete anything."),
        ]
    )

    model = ModelInfo(
        id="mock",
        provider="mock",
        display_name="Mock",
        capabilities=ModelCapabilities(
            modalities_in={Modality.TEXT},
            modalities_out={Modality.TEXT},
            tool_use=True,
            thinking=False,
            context_window=100_000,
            max_output=8_000,
        ),
        cost=ModelCost(),
    )
    registry = Registry(models={"mock": model})
    router = Router(registry, routing={TaskClass.MAIN: ["mock"]}, available={"mock"})

    loop = AgentLoop(
        router=router,
        providers={"mock": provider},
        tools=[DeleteEverythingTool()],
        confirm=deny_everything,
    )

    async for event in loop.run("delete everything"):
        if event.type == "needs_confirmation":
            print(f"[loop] pausing for confirmation on {event.call.name}")
        elif event.type == "run_done":
            print(f"\nFinal answer: {event.final_message.text()}")


if __name__ == "__main__":
    asyncio.run(main())
