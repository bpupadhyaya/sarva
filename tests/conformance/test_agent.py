"""Conformance tests for the agent loop — see spec-03 invariants."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sarva.agent.budget import Budget
from sarva.agent.events import LEGAL, AgentState
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import ToolContext, always_allow
from sarva.multimodal.content import TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.base import ToolSpec
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, load_routing

_DATA_DIR = Path(__file__).parent.parent.parent / "core" / "sarva" / "providers" / "data"


def _router() -> Router:
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    return Router(registry, routing, available={"mock"})


class _EchoTool:
    spec = ToolSpec(
        name="echo",
        description="echo the input back",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        destructive=False,
    )

    async def run(self, args, ctx: ToolContext) -> ToolResultBlock:
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=args["text"])])


class _DestructiveTool:
    spec = ToolSpec(
        name="delete_thing",
        description="pretend to delete something",
        input_schema={"type": "object", "properties": {}},
        destructive=True,
    )

    async def run(self, args, ctx: ToolContext) -> ToolResultBlock:
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text="deleted")])


class _RaisingTool:
    spec = ToolSpec(
        name="explode",
        description="always raises",
        input_schema={"type": "object", "properties": {}},
        destructive=False,
    )

    async def run(self, args, ctx: ToolContext):
        raise RuntimeError("kaboom")


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "runs"
    yield str(root)
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
async def test_state_legality_and_single_run_done(run_root):
    provider = MockProvider(script=[ScriptedTurn(text="done")])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)
    events = [e async for e in loop.run("say hi")]

    state_events = [e for e in events if e.type == "state_changed"]
    for a, b in zip(state_events, state_events[1:], strict=False):
        assert b.state in LEGAL[a.state] or a.state == b.state

    run_done = [e for e in events if e.type == "run_done"]
    assert len(run_done) == 1
    assert events[-1].type == "run_done"
    assert run_done[0].state == AgentState.DONE


@pytest.mark.asyncio
async def test_tool_result_completeness_and_order(run_root):
    calls = [
        ToolCallBlock(id="a", name="echo", arguments={"text": "first"}),
        ToolCallBlock(id="b", name="echo", arguments={"text": "second"}),
    ]
    provider = MockProvider(script=[ScriptedTurn(tool_calls=calls), ScriptedTurn(text="ok")])
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_EchoTool()],
        run_root=run_root,
    )
    events = [e async for e in loop.run("do two things")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert [f.result.tool_call_id for f in finished] == ["a", "b"]
    assert not any(f.result.is_error for f in finished)


@pytest.mark.asyncio
async def test_tool_errors_do_not_kill_the_loop(run_root):
    call = ToolCallBlock(id="x", name="explode", arguments={})
    provider = MockProvider(
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="recovered")]
    )
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_RaisingTool()],
        run_root=run_root,
    )
    events = [e async for e in loop.run("break something")]
    finished = [e for e in events if e.type == "tool_finished"]
    assert finished[0].result.is_error is True
    assert events[-1].state == AgentState.DONE


@pytest.mark.asyncio
async def test_unknown_tool_name_does_not_crash(run_root):
    call = ToolCallBlock(id="x", name="does_not_exist", arguments={})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="ok")])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)
    events = [e async for e in loop.run("call a fake tool")]
    finished = [e for e in events if e.type == "tool_finished"]
    assert finished[0].result.is_error is True
    assert events[-1].state == AgentState.DONE


@pytest.mark.asyncio
async def test_budget_enforcement(run_root):
    call = ToolCallBlock(id="a", name="echo", arguments={"text": "again"})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call])])  # always wants tools
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_EchoTool()],
        budget=Budget(max_model_calls=2),
        run_root=run_root,
    )
    events = [e async for e in loop.run("loop forever")]
    run_done = events[-1]
    assert run_done.state == AgentState.BUDGET_EXCEEDED
    assert run_done.spend.model_calls == 2


@pytest.mark.asyncio
async def test_confirmation_gating_deny(run_root):
    call = ToolCallBlock(id="d", name="delete_thing", arguments={})
    provider = MockProvider(
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="ok, skipped")]
    )

    async def deny(_call) -> bool:
        return False

    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_DestructiveTool()],
        confirm=deny,
        run_root=run_root,
    )
    events = [e async for e in loop.run("delete it")]
    assert any(e.type == "needs_confirmation" for e in events)
    finished = [e for e in events if e.type == "tool_finished"]
    assert finished[0].result.is_error is True
    assert "declined" in finished[0].result.content[0].text


@pytest.mark.asyncio
async def test_non_destructive_tool_never_asks_confirmation(run_root):
    call = ToolCallBlock(id="e", name="echo", arguments={"text": "hi"})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="ok")])
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_EchoTool()],
        confirm=always_allow,
        run_root=run_root,
    )
    events = [e async for e in loop.run("echo hi")]
    assert not any(e.type == "needs_confirmation" for e in events)


@pytest.mark.asyncio
async def test_transcript_is_replayable(run_root):
    provider = MockProvider(script=[ScriptedTurn(text="hi there")])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)
    events = [e async for e in loop.run("hello")]

    run_dirs = list(Path(run_root).iterdir())
    assert len(run_dirs) == 1
    lines = (run_dirs[0] / "transcript.jsonl").read_text().splitlines()
    assert len(lines) == len(events)
