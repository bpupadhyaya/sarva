"""Conformance tests for the agent loop — see spec-03 invariants."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sarva.agent.budget import Budget
from sarva.agent.events import LEGAL, AgentState
from sarva.agent.loop import AgentLoop, _required_modalities
from sarva.agent.tools import ToolContext, always_allow
from sarva.multimodal.content import (
    ImageBlock,
    Message,
    Modality,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sarva.providers.base import ModelCapabilities, ModelCost, ModelInfo, ToolSpec
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass, load_routing

_DATA_DIR = Path(__file__).parent.parent.parent / "core" / "sarva" / "providers" / "data"


def _router() -> Router:
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    return Router(registry, routing, available={"mock"})


def _text_only_model() -> ModelInfo:
    return ModelInfo(
        id="text-only",
        provider="mock",
        display_name="Text Only Mock",
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


def _text_only_router() -> Router:
    model = _text_only_model()
    registry = Registry(models={model.id: model})
    return Router(registry, routing={TaskClass.MAIN: ["text-only"]}, available={"text-only"})


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


def test_required_modalities_text_only():
    messages = [Message(role="user", content=[TextBlock(text="hi")])]
    assert _required_modalities(messages) == {Modality.TEXT}


def test_required_modalities_includes_image_when_present():
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="what's this?"),
                ImageBlock(media_type="image/png", data=b"\x89PNG\r\n"),
            ],
        )
    ]
    assert _required_modalities(messages) == {Modality.TEXT, Modality.IMAGE}


@pytest.mark.asyncio
async def test_image_content_with_no_vision_capable_model_fails_cleanly(run_root):
    """The loop asks the router for a model supporting every modality present
    in the conversation. When none is available, this must be a clean
    terminal FAILED state — never an unhandled exception out of the
    generator."""
    provider = MockProvider(script=[ScriptedTurn(text="should never be reached")])
    loop = AgentLoop(router=_text_only_router(), providers={"mock": provider}, run_root=run_root)
    image = ImageBlock(media_type="image/png", data=b"\x89PNG\r\n")

    events = [e async for e in loop.run("what's in this image?", extra_content=[image])]

    assert [e.type for e in events] == ["state_changed", "run_done"]
    assert events[0].state == AgentState.FAILED
    assert events[-1].state == AgentState.FAILED
    assert events[-1].final_message is None


@pytest.mark.asyncio
async def test_text_only_task_still_works_against_text_only_model(run_root):
    """Regression guard: modality-aware routing must not break the plain
    text-only path that every other test in this file relies on."""
    provider = MockProvider(script=[ScriptedTurn(text="all good")])
    loop = AgentLoop(router=_text_only_router(), providers={"mock": provider}, run_root=run_root)

    events = [e async for e in loop.run("hello")]

    assert events[-1].type == "run_done"
    assert events[-1].state == AgentState.DONE


@pytest.mark.asyncio
async def test_transcript_out_includes_final_turn_on_plain_success(run_root):
    """Regression test for a real bug: `messages` (and therefore
    transcript_out) used to only gain the final assistant turn on the
    TOOL_USE path — a plain END_TURN success silently dropped it."""
    provider = MockProvider(script=[ScriptedTurn(text="the answer is 42")])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)
    transcript: list[Message] = []

    events = [e async for e in loop.run("what's the answer?", transcript_out=transcript)]

    assert events[-1].state == AgentState.DONE
    assert [m.role for m in transcript] == ["user", "assistant"]
    assert transcript[0].text() == "what's the answer?"
    assert transcript[1].text() == "the answer is 42"


@pytest.mark.asyncio
async def test_transcript_out_includes_full_tool_use_round(run_root):
    """The whole reason transcript_out exists: recover history across a
    tool-use round for session persistence, since RunDoneEvent.final_message
    alone only ever carries the *last* turn."""
    call = ToolCallBlock(id="c1", name="echo", arguments={"text": "ping"})
    provider = MockProvider(
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done: ping")]
    )
    loop = AgentLoop(
        router=_router(), providers={"mock": provider}, tools=[_EchoTool()], run_root=run_root
    )
    transcript: list[Message] = []

    events = [e async for e in loop.run("echo ping please", transcript_out=transcript)]

    assert events[-1].state == AgentState.DONE
    assert [m.role for m in transcript] == ["user", "assistant", "user", "assistant"]
    assert any(b.type == "tool_call" for b in transcript[1].content)  # assistant requests the tool
    assert any(b.type == "tool_result" for b in transcript[2].content)  # user carries the result
    assert transcript[3].text() == "done: ping"


@pytest.mark.asyncio
async def test_transcript_out_populated_even_on_failure(run_root):
    """The contract says 'any terminal state', not just success — a caller
    debugging a failed run should still see what led up to it."""
    provider = MockProvider(script=[ScriptedTurn(error="boom", error_retryable=False)])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)
    transcript: list[Message] = []

    events = [e async for e in loop.run("this will fail", transcript_out=transcript)]

    assert events[-1].state == AgentState.FAILED
    assert len(transcript) == 1
    assert transcript[0].role == "user"


@pytest.mark.asyncio
async def test_transcript_out_defaults_to_none_and_is_optional(run_root):
    """Purely additive: every existing call site that doesn't pass
    transcript_out must be completely unaffected."""
    provider = MockProvider(script=[ScriptedTurn(text="fine")])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)

    events = [e async for e in loop.run("no transcript wanted here")]

    assert events[-1].state == AgentState.DONE
