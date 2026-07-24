"""Conformance tests for the agent loop — see spec-03 invariants."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from PIL import Image
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
from sarva.multimodal.degraders.image import ImageToTextDegrader
from sarva.providers.base import ModelCapabilities, ModelCost, ModelInfo, ToolSpec
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass, load_routing


def _real_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 8), color=(0, 128, 255)).save(buf, format="PNG")
    return buf.getvalue()


class _NullAudioDegrader:
    """A degrader for a modality that's never actually present in these
    tests — exists only to prove degraders={} being non-empty isn't by
    itself what makes the fallback succeed; it must cover the *specific*
    modality that's actually missing (IMAGE)."""

    source = Modality.AUDIO

    async def degrade(self, block):
        return [TextBlock(text="[audio omitted]")]


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


class _SessionIdCaptureTool:
    """Echoes back ctx.session_id so a test can assert on it directly --
    the real proof that AgentLoop.run(session_id=...) actually reaches a
    tool's ToolContext, not just that the parameter exists."""

    spec = ToolSpec(
        name="capture_session_id",
        description="echo back the session id from ToolContext",
        input_schema={"type": "object", "properties": {}},
        destructive=False,
    )

    async def run(self, args, ctx: ToolContext) -> ToolResultBlock:
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=str(ctx.session_id))])


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


@pytest.mark.asyncio
async def test_degradation_fallback_succeeds_and_sends_degraded_content(run_root):
    """The recoverable case `test_image_content_with_no_vision_capable_model_fails_cleanly`
    documents as *not yet wired*: with a degrader configured for the
    missing modality, the same scenario now falls back to the
    text-capable model instead of failing. Echo-mode MockProvider (no
    script) echoes the last user message's TextBlocks back, so the
    echoed response proves the *degraded* text — not just the original
    task text — actually reached the provider, not merely that the run
    happened to end in DONE for an unrelated reason."""
    provider = MockProvider()  # echo mode
    loop = AgentLoop(
        router=_text_only_router(),
        providers={"mock": provider},
        run_root=run_root,
        degraders={Modality.IMAGE: ImageToTextDegrader()},
    )
    image = ImageBlock(media_type="image/png", data=_real_png_bytes())

    events = [e async for e in loop.run("what's in this image?", extra_content=[image])]

    assert events[-1].state == AgentState.DONE
    echoed = events[-1].final_message.text()
    assert "what's in this image?" in echoed
    assert "could not be described" in echoed  # the degrader's own disclaimer text
    assert "12x8" in echoed  # the degrader's real decoded metadata, not a stub


@pytest.mark.asyncio
async def test_degradation_fallback_does_not_help_when_no_degrader_covers_the_modality(run_root):
    """A non-empty `degraders` dict must not make every unsupported-modality
    run succeed regardless of content — it must cover the *specific*
    modality actually present. A degrader registered only for AUDIO must
    leave an IMAGE-only conversation failing exactly as it did with no
    degraders configured at all."""
    provider = MockProvider(script=[ScriptedTurn(text="should never be reached")])
    loop = AgentLoop(
        router=_text_only_router(),
        providers={"mock": provider},
        run_root=run_root,
        degraders={Modality.AUDIO: _NullAudioDegrader()},
    )
    image = ImageBlock(media_type="image/png", data=_real_png_bytes())

    events = [e async for e in loop.run("what's in this image?", extra_content=[image])]

    assert events[-1].state == AgentState.FAILED
    assert events[-1].final_message is None


@pytest.mark.asyncio
async def test_degradation_fallback_not_triggered_when_a_supporting_model_exists(run_root):
    """Regression guard: with a vision-capable model actually available
    (the registry's `mock` entry supports image input directly — see
    models.yaml), the degradation path must never trigger — the original
    ImageBlock should reach the model unmodified, not a degraded
    placeholder, exactly as before this feature existed."""
    provider = MockProvider()  # echo mode
    loop = AgentLoop(
        router=_router(),  # available={"mock"}; mock's own capabilities include image
        providers={"mock": provider},
        run_root=run_root,
        degraders={Modality.IMAGE: ImageToTextDegrader()},
    )
    image = ImageBlock(media_type="image/png", data=_real_png_bytes())

    events = [e async for e in loop.run("what's in this image?", extra_content=[image])]

    assert events[-1].state == AgentState.DONE
    echoed = events[-1].final_message.text()
    assert "could not be described" not in echoed


@pytest.mark.asyncio
async def test_degradation_fallback_double_failure_still_fails_cleanly(run_root):
    """If even the TEXT-only fallback model can't be found (degenerate
    config: zero available models at all), the loop must still terminate
    cleanly in FAILED, not raise out of the generator."""
    registry = Registry(models={})
    router = Router(registry, routing={}, available=set())
    provider = MockProvider(script=[ScriptedTurn(text="unreachable")])
    loop = AgentLoop(
        router=router,
        providers={"mock": provider},
        run_root=run_root,
        degraders={Modality.IMAGE: ImageToTextDegrader()},
    )

    events = [e async for e in loop.run("hello")]

    assert events[-1].type == "run_done"
    assert events[-1].state == AgentState.FAILED


@pytest.mark.asyncio
async def test_model_override_reaches_the_provider_request(run_root):
    """model_override isn't just accepted -- the real registered model's
    id must be the one that actually reaches GenerateRequest.model, not
    whatever the router's default candidate list would have picked."""
    provider = MockProvider(script=[ScriptedTurn(text="ok")])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)

    events = [e async for e in loop.run("hi", model_override="mock")]

    assert events[-1].state == AgentState.DONE


@pytest.mark.asyncio
async def test_unknown_model_override_fails_cleanly_without_silent_substitution(run_root):
    """The real safety property this exists for: an explicit but wrong
    model_override must never be silently caught by the modality-
    degradation fallback and swapped for a different model -- even with
    degraders configured (the exact condition that would otherwise
    trigger the fallback path for a genuinely unsupported modality)."""
    provider = MockProvider(script=[ScriptedTurn(text="should never be reached")])
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        run_root=run_root,
        degraders={Modality.IMAGE: ImageToTextDegrader()},
    )

    events = [e async for e in loop.run("hi", model_override="totally-not-a-real-model")]

    assert events[-1].type == "run_done"
    assert events[-1].state == AgentState.FAILED
    state_changed = next(e for e in events if e.type == "state_changed")
    assert "totally-not-a-real-model" in state_changed.detail


@pytest.mark.asyncio
async def test_run_session_id_reaches_the_tool_context(run_root):
    """The actual proof session_id threading works end to end: a tool
    that echoes ctx.session_id back must see the exact value passed to
    run(session_id=...), not None and not some other placeholder."""
    call = ToolCallBlock(id="a", name="capture_session_id", arguments={})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done")])
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_SessionIdCaptureTool()],
        run_root=run_root,
    )

    events = [e async for e in loop.run("what's my session?", session_id="my-real-session")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert finished[0].result.content[0].text == "my-real-session"


@pytest.mark.asyncio
async def test_run_without_session_id_leaves_ctx_session_id_none(run_root):
    """Regression guard: every existing call site that doesn't pass
    session_id (the vast majority of this test file) must be completely
    unaffected -- ToolContext.session_id stays None, not some accidental
    default."""
    call = ToolCallBlock(id="a", name="capture_session_id", arguments={})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done")])
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_SessionIdCaptureTool()],
        run_root=run_root,
    )

    events = [e async for e in loop.run("what's my session?")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert finished[0].result.content[0].text == "None"
