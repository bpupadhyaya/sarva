"""Conformance tests for the provider contract — see spec-01 invariants.

Runs against the MockProvider (always) — a real adapter under test would be
parametrized alongside it and marked `@pytest.mark.live`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sarva.multimodal.content import Message, Modality, TextBlock, ToolCallBlock
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    StopReason,
    StreamErrorEvent,
    ToolSpec,
    complete,
)
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass, UnknownModelError, load_routing

_DATA_DIR = Path(__file__).parent.parent.parent / "core" / "sarva" / "providers" / "data"


def _req(text: str, tools: list[ToolSpec] | None = None) -> GenerateRequest:
    return GenerateRequest(
        model="mock",
        messages=[Message(role="user", content=[TextBlock(text=text)])],
        tools=tools or [],
    )


@pytest.mark.asyncio
async def test_terminal_event_law():
    provider = MockProvider()
    events = [e async for e in provider.generate(_req("hi"))]
    assert isinstance(events[-1], DoneEvent)
    assert sum(isinstance(e, (DoneEvent, StreamErrorEvent)) for e in events) == 1


@pytest.mark.asyncio
async def test_delta_message_equivalence():
    provider = MockProvider(script=[ScriptedTurn(text="hello world")])
    events = [e async for e in provider.generate(_req("hi"))]
    done = next(e for e in events if isinstance(e, DoneEvent))
    from sarva.providers.base import TextDeltaEvent

    deltas = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert deltas.strip() == done.message.text().strip()


@pytest.mark.asyncio
async def test_tool_round_trip():
    call = ToolCallBlock(id="tc1", name="get_weather", arguments={"city": "Paris"})
    provider = MockProvider(
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="it is sunny")]
    )
    tool = ToolSpec(name="get_weather", description="d", input_schema={"type": "object"})
    events = [e async for e in provider.generate(_req("weather?", tools=[tool]))]
    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.stop_reason == StopReason.TOOL_USE
    assert any(b.type == "tool_call" for b in done.message.content)

    from sarva.multimodal.content import ToolResultBlock

    followup_messages = [
        Message(role="user", content=[TextBlock(text="weather?")]),
        done.message,
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="tc1", content=[TextBlock(text="sunny, 20C")])],
        ),
    ]
    req2 = GenerateRequest(model="mock", messages=followup_messages, tools=[tool])
    final = await complete(provider, req2)
    assert final.stop_reason == StopReason.END_TURN


@pytest.mark.asyncio
async def test_mid_stream_error_yields_not_raises():
    provider = MockProvider(script=[ScriptedTurn(error="simulated failure", error_retryable=False)])
    events = [e async for e in provider.generate(_req("hi"))]
    assert isinstance(events[-1], StreamErrorEvent)
    assert events[-1].retryable is False


@pytest.mark.asyncio
async def test_complete_raises_on_error():
    from sarva.providers.base import ProviderError

    provider = MockProvider(script=[ScriptedTurn(error="boom")])
    with pytest.raises(ProviderError):
        await complete(provider, _req("hi"))


@pytest.mark.asyncio
async def test_usage_present():
    provider = MockProvider(script=[ScriptedTurn(text="a reasonably long response")])
    done = await complete(provider, _req("hi"))
    assert done.usage.output_tokens > 0
    assert done.usage.cost_usd == 0.0  # mock is free


def test_registry_loads_and_validates():
    registry = Registry.load(_DATA_DIR / "models.yaml")
    assert registry.get("mock").provider == "mock"
    assert len(registry.all()) >= 4


def test_router_respects_modality_and_availability():
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    router = Router(registry, routing, available={"mock"})
    picked = router.pick(TaskClass.MAIN)
    assert picked.id == "mock"  # only mock is "available"

    with pytest.raises(LookupError):
        Router(registry, routing, available=set()).pick(TaskClass.MAIN)


def test_router_never_returns_unsupported_modality():
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    router = Router(registry, routing, available={"mock"})
    picked = router.pick(TaskClass.VISION, needs={Modality.IMAGE})
    assert Modality.IMAGE in picked.capabilities.modalities_in


def test_router_pick_with_a_real_override_bypasses_availability_and_modality():
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    router = Router(registry, routing, available=set())  # nothing "available"
    picked = router.pick(TaskClass.MAIN, override="mock")
    assert picked.id == "mock"


def test_router_pick_with_an_unknown_override_raises_a_distinct_error():
    # Deliberately NOT a plain LookupError -- AgentLoop.run() depends on
    # this being a distinct type so an explicit-but-wrong model override
    # can never be silently caught by the modality-degradation fallback
    # and substituted with a different model. See UnknownModelError's
    # own docstring.
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    router = Router(registry, routing, available={"mock"})
    with pytest.raises(UnknownModelError, match="bogus-model-id"):
        router.pick(TaskClass.MAIN, override="bogus-model-id")
    assert not issubclass(UnknownModelError, LookupError)


@pytest.mark.asyncio
async def test_cancellation_does_not_hang():
    provider = MockProvider(script=[ScriptedTurn(text="a very long response " * 50)])
    gen = provider.generate(_req("hi"))
    await gen.__anext__()  # take one event
    await gen.aclose()  # must not hang or raise
