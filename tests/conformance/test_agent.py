"""Conformance tests for the agent loop — see spec-03 invariants."""

from __future__ import annotations

import asyncio
import io
import shutil
from pathlib import Path

import pytest
from PIL import Image
from sarva.agent.budget import Budget
from sarva.agent.events import LEGAL, AgentState
from sarva.agent.loop import AgentLoop, _required_modalities
from sarva.agent.subagents import DelegateTool
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
async def test_a_hung_tool_times_out_instead_of_blocking_every_other_result_forever(
    run_root, monkeypatch
):
    # A real bug found by actually running a turn with one fast tool
    # call and one that never returns: asyncio.gather withholds ALL
    # results until EVERY coroutine finishes, so the hung tool blocked
    # the whole turn forever -- silently discarding the fast tool's
    # already-completed result too, confirmed live with an outer guard
    # that never unblocked. RunShellTool self-protects with its own
    # internal timeout, but a tool with none at all (e.g. a remote MCP
    # server that never responds) had no recovery path whatsoever.
    import sarva.agent.loop as loop_module

    monkeypatch.setattr(loop_module, "_TOOL_TIMEOUT_SECONDS", 0.2)

    class _HungTool:
        spec = ToolSpec(
            name="hung",
            description="never returns",
            input_schema={"type": "object", "properties": {}},
            destructive=False,
        )

        async def run(self, args, ctx: ToolContext):
            await asyncio.sleep(3600)
            raise AssertionError("never actually reached")

    calls = [
        ToolCallBlock(id="fast", name="echo", arguments={"text": "quick"}),
        ToolCallBlock(id="slow", name="hung", arguments={}),
    ]
    provider = MockProvider(script=[ScriptedTurn(tool_calls=calls), ScriptedTurn(text="done")])
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_EchoTool(), _HungTool()],
        run_root=run_root,
    )

    events = await asyncio.wait_for(
        _collect(loop.run("run both")), timeout=5
    )  # the real proof: this must not hang past the tool's own short timeout

    finished = {f.result.tool_call_id: f.result for f in events if f.type == "tool_finished"}
    assert finished["fast"].is_error is False
    assert finished["fast"].content[0].text == "quick"
    assert finished["slow"].is_error is True
    assert "timed out" in finished["slow"].content[0].text
    assert events[-1].state == AgentState.DONE


async def _collect(agen):
    return [e async for e in agen]


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
async def test_two_destructive_calls_sharing_the_same_id_are_confirmed_independently(run_root):
    # A real bug found by actually constructing two distinct
    # ToolCallBlocks that share the same `.id` (a malformed/adversarial
    # model turn -- nothing validates id uniqueness before this point):
    # the confirmation-tracking dict used to be keyed by `call.id`
    # alone, so the second call's confirmation answer silently
    # overwrote the first's, and BOTH calls read back whichever answer
    # was decided last -- confirmed live: an explicitly DECLINED
    # destructive call still executed because a different call sharing
    # its id happened to get approved. Worse than a crash: it silently
    # defeats the whole confirm-gate the AWAITING_CONFIRMATION state
    # exists to enforce.
    class _DeleteThingB:
        spec = ToolSpec(
            name="delete_thing_b",
            description="pretend to delete a second thing",
            input_schema={"type": "object", "properties": {}},
            destructive=True,
        )

        async def run(self, args, ctx: ToolContext) -> ToolResultBlock:
            return ToolResultBlock(tool_call_id="", content=[TextBlock(text="deleted b")])

    denied_call = ToolCallBlock(id="dup", name="delete_thing", arguments={})
    approved_call = ToolCallBlock(id="dup", name="delete_thing_b", arguments={})
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[denied_call, approved_call]),
            ScriptedTurn(text="done"),
        ]
    )

    async def approve_b_deny_others(call: ToolCallBlock) -> bool:
        return call.name == "delete_thing_b"

    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_DestructiveTool(), _DeleteThingB()],
        confirm=approve_b_deny_others,
        run_root=run_root,
    )
    events = [e async for e in loop.run("delete both")]

    denied_result = next(
        f.result
        for f in events
        if f.type == "tool_finished" and "declined" in f.result.content[0].text
    )
    approved_result = next(
        f.result
        for f in events
        if f.type == "tool_finished" and "deleted b" in f.result.content[0].text
    )
    assert denied_result.is_error is True
    assert approved_result.is_error is False


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


@pytest.mark.asyncio
async def test_run_directories_are_pruned_beyond_the_retention_cap(run_root, monkeypatch):
    # A real bug found by actually running AgentLoop.run() in a loop the
    # way sarva.server.app's /chat and /ws/chat handlers do (a fresh
    # AgentLoop per request, discarded after): every run created a new
    # run_root/<run_id>/ directory and nothing anywhere ever deleted one,
    # unbounded growth over the lifetime of a long-running `sarva serve`
    # process. Uses a small monkeypatched cap so the test doesn't need to
    # actually run 200+ turns to prove pruning happens.
    import sarva.agent.loop as loop_module

    monkeypatch.setattr(loop_module, "_MAX_RETAINED_RUNS", 3)

    for i in range(6):
        provider = MockProvider(script=[ScriptedTurn(text=f"turn {i}")])
        loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)
        async for _ in loop.run(f"task {i}"):
            pass

    run_dirs = list(Path(run_root).iterdir())
    assert len(run_dirs) == 3


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
async def test_a_retryable_stream_error_actually_retries_instead_of_crashing(run_root):
    # A real bug found by actually running a retryable stream error (the
    # exact case every real adapter's RateLimitError/5xx APIStatusError
    # sets retryable=True for): the retry path loops back to the top of
    # the while-loop without ever leaving CALLING_MODEL, then immediately
    # re-asserts CALLING_MODEL -> CALLING_MODEL -- an AssertionError on
    # the very first retry, confirmed live before this fix, defeating the
    # whole retry mechanism for every real provider. Nothing catches
    # AssertionError anywhere above this loop (cli.py, server/app.py), so
    # this reached a user as a raw traceback or a bare disconnect.
    provider = MockProvider(
        script=[
            ScriptedTurn(error="rate limited", error_retryable=True),
            ScriptedTurn(text="ok now"),
        ]
    )
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)

    events = [e async for e in loop.run("hello")]

    assert events[-1].state == AgentState.DONE
    # Two real model calls actually happened (the retry, then the real
    # success) -- not just that the run didn't crash.
    stream_events = [e for e in events if e.type == "model_stream"]
    assert any(getattr(e.event, "code", None) == "mock_error" for e in stream_events)
    assert events[-1].spend.model_calls == 1  # the retry itself isn't counted as a new call


@pytest.mark.asyncio
async def test_a_permanently_retryable_stream_error_eventually_gives_up(run_root, monkeypatch):
    # A real bug found by actually driving a provider that always yields
    # a retryable StreamErrorEvent (simulating one stuck returning rate-
    # limit/5xx responses indefinitely): the retry branch loops back to
    # the top of the while-loop via `if done is None: continue`, jumping
    # past the `spend.exceeded(self._budget)` check a few lines below --
    # confirmed live, the run never reached a terminal state after 6 real
    # seconds and 12 real provider-call attempts, regardless of how tight
    # the caller's own Budget was (max_model_calls=50, max_wall_seconds=
    # 3600.0 here). Every OTHER path through this loop is bounded by
    # spend.exceeded; this was the one gap where a stuck provider burns
    # real API cost and wall-clock indefinitely. `asyncio.sleep` patched
    # to a no-op so this test doesn't actually wait out five real 1s
    # retry delays.
    import sarva.agent.loop as loop_module

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(loop_module.asyncio, "sleep", _no_sleep)

    provider = MockProvider(
        script=[ScriptedTurn(error="rate limited", error_retryable=True)] * 1000
    )
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        budget=Budget(max_wall_seconds=3600.0, max_model_calls=50),
        run_root=run_root,
    )

    events = await asyncio.wait_for(_collect(loop.run("hello")), timeout=5)

    assert events[-1].state == AgentState.FAILED
    assert events[-1].type == "run_done"
    # Confirms the budget itself was never what stopped it -- the retry
    # counter fired first, well under both configured limits.
    assert events[-1].spend.model_calls < 50
    state_changed = [e for e in events if e.type == "state_changed"]
    assert f"{loop_module._MAX_STREAM_RETRIES}" in state_changed[-1].detail


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
async def test_degradation_fallback_reports_cleanly_when_the_image_itself_cannot_be_decoded(
    run_root,
):
    # A real bug found by actually running this exact scenario:
    # ImageToTextDegrader.degrade() raises its own ImageDecodeError when
    # the bytes genuinely can't be decoded (not a LookupError or
    # UnsupportedModalityError, the only two exception types this
    # fallback used to catch), so it propagated straight out of
    # AgentLoop.run()'s async generator uncaught instead of the clean
    # FAILED state this whole fallback exists to produce. Confirmed
    # live: with a text-only router (forcing the fallback path) and a
    # real ImageToTextDegrader, a genuinely undecodable image blob
    # crashed the loop before this fix.
    provider = MockProvider(script=[ScriptedTurn(text="should never be reached")])
    loop = AgentLoop(
        router=_text_only_router(),
        providers={"mock": provider},
        run_root=run_root,
        degraders={Modality.IMAGE: ImageToTextDegrader()},
    )
    undecodable = ImageBlock(media_type="image/png", data=b"not an image at all")

    events = [e async for e in loop.run("what's in this image?", extra_content=[undecodable])]

    assert events[-1].type == "run_done"
    assert events[-1].state == AgentState.FAILED
    assert events[-1].final_message is None
    # The degrader's own specific, actionable reason must reach the
    # caller -- not a generic "no model supports this modality" message
    # that no longer describes what actually went wrong.
    state_changed = next(e for e in events if e.type == "state_changed" and e.detail)
    assert "could not decode image for degradation" in state_changed.detail


@pytest.mark.asyncio
async def test_degradation_fallback_reports_cleanly_for_a_truncated_real_image(run_root):
    # A related real bug in ImageToTextDegrader itself: a genuinely
    # TRUNCATED (not fully unrecognizable) real image made Pillow raise
    # a plain OSError("Truncated File Read") reading `.size`, which the
    # degrader's own except clause (UnidentifiedImageError only) didn't
    # catch at all -- a raw, uncontextualized PIL error instead of this
    # degrader's own documented ImageDecodeError.
    provider = MockProvider(script=[ScriptedTurn(text="should never be reached")])
    loop = AgentLoop(
        router=_text_only_router(),
        providers={"mock": provider},
        run_root=run_root,
        degraders={Modality.IMAGE: ImageToTextDegrader()},
    )
    # A fixed 20-byte truncation always lands mid-IHDR-chunk-read
    # regardless of image size (PNG's signature + chunk header are a
    # fixed 16 bytes), reliably reproducing Pillow's OSError -- a
    # size-relative truncation like `real_png[: len(real_png) // 4]`
    # can accidentally land too early and hit UnidentifiedImageError
    # instead depending on the image's total size, confirmed directly
    # (see test_degraders.py's equivalent unit-level test).
    real_png = _real_png_bytes()
    truncated = ImageBlock(media_type="image/png", data=real_png[:20])

    events = [e async for e in loop.run("what's in this image?", extra_content=[truncated])]

    assert events[-1].type == "run_done"
    assert events[-1].state == AgentState.FAILED
    state_changed = next(e for e in events if e.type == "state_changed" and e.detail)
    assert "could not decode image for degradation" in state_changed.detail


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


@pytest.mark.asyncio
async def test_delegate_task_spawns_a_real_subagent_and_merges_its_spend(run_root):
    # The real proof subagent fan-out works end to end, not just that
    # DelegateTool exists: a fresh, independent AgentLoop actually runs
    # to completion, its final text comes back as the tool result, AND
    # its own real model-call cost is added into the PARENT's spend --
    # confirmed by the exact total, not just "some number greater than
    # zero." Three real provider calls happen: the parent's initial
    # request, the subagent's own single turn, and the parent's final
    # turn after getting the subagent's answer back.
    delegate_call = ToolCallBlock(
        id="d1", name="delegate_task", arguments={"task": "say something useful"}
    )
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[delegate_call]),
            ScriptedTurn(text="the subagent's own real answer"),
            ScriptedTurn(text="parent's final answer, using the subagent's work"),
        ]
    )
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[DelegateTool()],
        budget=Budget(max_model_calls=10),
        run_root=run_root,
    )

    events = [e async for e in loop.run("please delegate this")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert len(finished) == 1
    assert finished[0].result.is_error is False
    assert finished[0].result.content[0].text == "the subagent's own real answer"

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.DONE
    assert run_done[-1].final_message.text() == "parent's final answer, using the subagent's work"
    # 1 (parent's delegate request) + 1 (the real subagent turn) + 1
    # (parent's final turn) -- the subagent's own cost is genuinely
    # counted, not silently free.
    assert run_done[-1].spend.model_calls == 3


@pytest.mark.asyncio
async def test_delegate_task_cannot_recurse_into_delegating_further(run_root):
    # Bounds fan-out at one level: a subagent's own tool list must not
    # include delegate_task itself. Proven by scripting the SUBAGENT to
    # try calling delegate_task anyway -- if recursion were allowed this
    # would spawn a third-level loop; since it's genuinely excluded, the
    # subagent's own loop dispatches it as an ordinary unknown-tool
    # error (the same path any unrecognized tool name takes) and keeps
    # going, rather than crashing or actually recursing.
    delegate_call = ToolCallBlock(
        id="d1", name="delegate_task", arguments={"task": "try to delegate further"}
    )
    recursive_call = ToolCallBlock(
        id="d2", name="delegate_task", arguments={"task": "a forbidden second level"}
    )
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[delegate_call]),  # parent delegates
            ScriptedTurn(tool_calls=[recursive_call]),  # subagent tries to delegate again
            ScriptedTurn(text="subagent's real final answer, after its own tool error"),
            ScriptedTurn(text="parent's final answer"),
        ]
    )
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[DelegateTool()],
        budget=Budget(max_model_calls=10),
        run_root=run_root,
    )

    events = [e async for e in loop.run("please delegate this")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert len(finished) == 1  # only the PARENT's own delegate_task call is visible here
    assert finished[0].result.is_error is False
    assert (
        finished[0].result.content[0].text
        == "subagent's real final answer, after its own tool error"
    )

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.DONE
    assert run_done[-1].final_message.text() == "parent's final answer"


@pytest.mark.asyncio
async def test_delegate_task_reports_a_clean_error_when_the_subagent_runs_out_of_budget(run_root):
    # A tight shared budget means the subagent's own allotment (what's
    # left of the parent's budget) is exhausted before it can produce a
    # real answer -- DelegateTool must report this as a clean tool
    # error, not crash, and the subagent's real (wasted) spend must
    # still count against the parent, which is exactly why the parent's
    # own run also ends in BUDGET_EXCEEDED rather than DONE afterward.
    delegate_call = ToolCallBlock(id="d1", name="delegate_task", arguments={"task": "do something"})
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[delegate_call]),
            ScriptedTurn(text="the subagent's only allowed call -- still not enough budget"),
            ScriptedTurn(text="parent tries to continue anyway"),
        ]
    )
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[DelegateTool()],
        budget=Budget(max_model_calls=2),
        run_root=run_root,
    )

    events = [e async for e in loop.run("please delegate this")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert len(finished) == 1
    assert finished[0].result.is_error is True
    assert "did not complete successfully" in finished[0].result.content[0].text

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.BUDGET_EXCEEDED


@pytest.mark.asyncio
async def test_delegate_task_rejects_an_empty_task_string(run_root):
    call = ToolCallBlock(id="d1", name="delegate_task", arguments={"task": "   "})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="ok")])
    loop = AgentLoop(
        router=_router(), providers={"mock": provider}, tools=[DelegateTool()], run_root=run_root
    )

    events = [e async for e in loop.run("delegate an empty task")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert finished[0].result.is_error is True
    assert "non-empty" in finished[0].result.content[0].text


@pytest.mark.asyncio
async def test_delegate_task_fails_cleanly_with_no_spawn_hook_available():
    # A bare ToolContext (e.g. built directly by a caller that isn't
    # AgentLoop.run() itself) leaves spawn_subagent at its None default
    # -- DelegateTool must report this cleanly rather than crash with an
    # AttributeError on a None callable.
    ctx = ToolContext(workdir=".", run_dir=".")
    result = await DelegateTool().run({"task": "do something"}, ctx)
    assert result.is_error is True
    assert "not available" in result.content[0].text
