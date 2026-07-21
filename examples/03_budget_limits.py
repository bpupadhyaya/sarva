"""Example 03 — Budgets stop a runaway loop cleanly.

A model that always wants to call a tool would loop forever without a
budget. Sarva's agent loop treats "budget exceeded" as a normal terminal
state — no exception, no hang, just a clean stop with a spend summary.

Run: uv run python examples/03_budget_limits.py
"""

from __future__ import annotations

import asyncio

from sarva.agent.budget import Budget
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import ToolContext, always_allow
from sarva.multimodal.content import Modality, TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.base import ModelCapabilities, ModelCost, ModelInfo, ToolSpec
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass


class LoopyTool:
    spec = ToolSpec(
        name="loopy",
        description="A tool that never satisfies the caller — for demoing budgets.",
        input_schema={"type": "object", "properties": {}},
        destructive=False,
    )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResultBlock:
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text="not done yet")])


async def main() -> None:
    always_wants_a_tool_call = ScriptedTurn(
        tool_calls=[ToolCallBlock(id="x", name="loopy", arguments={})]
    )
    provider = MockProvider(script=[always_wants_a_tool_call])  # repeats forever

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
        tools=[LoopyTool()],
        confirm=always_allow,
        budget=Budget(max_model_calls=5),
    )

    async for event in loop.run("keep trying forever"):
        if event.type == "run_done":
            print(f"stopped: {event.state} after {event.spend.model_calls} model calls")


if __name__ == "__main__":
    asyncio.run(main())
