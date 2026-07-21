"""Example 02 — Tool use.

Shows the agent loop dispatching a tool call and completing the turn once
the tool result comes back. Uses MockProvider's scripting so this runs
fully offline, deterministically, with no model required — the point is to
see the *loop's* tool-use mechanics, not a real model's tool-choice.

Run: uv run python examples/02_tool_use.py
"""

from __future__ import annotations

import asyncio

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import Tool, ToolContext, always_allow
from sarva.multimodal.content import TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.base import ToolSpec
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass


class GetWeatherTool:
    """A pretend tool — no network, just demonstrates the round trip."""

    spec = ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        destructive=False,
    )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResultBlock:
        return ToolResultBlock(
            tool_call_id="", content=[TextBlock(text=f"{args['city']}: 22C, sunny")]
        )


async def main() -> None:
    call = ToolCallBlock(id="c1", name="get_weather", arguments={"city": "Kathmandu"})
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(text="It's a sunny 22C in Kathmandu right now."),
        ]
    )

    registry = Registry(models={m.id: m for m in [_mock_model()]})
    router = Router(registry, routing={TaskClass.MAIN: ["mock"]}, available={"mock"})

    tools: list[Tool] = [GetWeatherTool()]
    loop = AgentLoop(router=router, providers={"mock": provider}, tools=tools, confirm=always_allow)

    async for event in loop.run("What's the weather in Kathmandu?"):
        if event.type == "tool_started":
            print(f"-> calling {event.call.name}({event.call.arguments})")
        elif event.type == "tool_finished":
            print(f"<- {event.result.content[0].text}")
        elif event.type == "run_done":
            print(f"\nFinal answer: {event.final_message.text()}")


def _mock_model():
    from sarva.multimodal.content import Modality
    from sarva.providers.base import ModelCapabilities, ModelCost, ModelInfo

    return ModelInfo(
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


if __name__ == "__main__":
    asyncio.run(main())
