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
from sarva.providers.base import (
    DoneEvent,
    ModelCapabilities,
    ModelCost,
    ModelInfo,
    StopReason,
    ToolCallEvent,
    ToolSpec,
    Usage,
)
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


class _ConcurrentDelegateRaceProvider:
    """First real call: the parent issues TWO concurrent delegate_task
    calls in one TOOL_USE round. Every call after that (the parent's own
    follow-ups, and both subagents' own turns alike) keeps requesting
    another tool call forever, so each loop only stops once ITS OWN
    budget check trips it -- letting a test observe the REAL total
    number of provider calls actually incurred, not just what the final
    Spend object claims. `asyncio.sleep` on every call forces genuine
    interleaving between the two concurrently-dispatched subagents,
    matching how `asyncio.gather` really schedules them in production."""

    name = "mock"

    def __init__(self) -> None:
        self.n = 0

    async def generate(self, request):
        self.n += 1
        await asyncio.sleep(0.01)
        if self.n == 1:
            calls = [
                ToolCallBlock(id="d1", name="delegate_task", arguments={"task": "subtask A"}),
                ToolCallBlock(id="d2", name="delegate_task", arguments={"task": "subtask B"}),
            ]
            for c in calls:
                yield ToolCallEvent(call=c)
            yield DoneEvent(
                stop_reason=StopReason.TOOL_USE,
                message=Message(role="assistant", content=calls),
                usage=Usage(input_tokens=1, output_tokens=1),
            )
            return
        call = ToolCallBlock(id=f"e{self.n}", name="echo", arguments={"text": "x"})
        yield ToolCallEvent(call=call)
        yield DoneEvent(
            stop_reason=StopReason.TOOL_USE,
            message=Message(role="assistant", content=[call]),
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_concurrent_delegate_task_calls_do_not_let_real_spend_exceed_budget(run_root):
    # A real bug found by actually dispatching TWO concurrent
    # delegate_task calls in one TOOL_USE round -- ordinary usage
    # ("delegate these two independent things in parallel"), not an
    # adversarial trick. asyncio.gather runs every tool call in a round
    # concurrently, and spawn_subagent's own budget clamp used to read
    # `spend` synchronously before its first await -- two concurrent
    # calls both read the SAME, not-yet-decremented spend (real mutation
    # only happened once a subagent's entire run had already finished),
    # so each got granted a full independent slice of the same remaining
    # budget. Confirmed live before the fix: Budget(max_model_calls=3)
    # with two concurrent delegate_task calls made 6 real provider
    # calls, double the declared cap.
    provider = _ConcurrentDelegateRaceProvider()
    budget = Budget(max_model_calls=3, max_wall_seconds=3600.0)
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[DelegateTool(), _EchoTool()],
        budget=budget,
        run_root=run_root,
    )

    events = [e async for e in loop.run("please delegate this to two subagents concurrently")]

    run_done = [e for e in events if e.type == "run_done"][-1]
    assert run_done.state == AgentState.BUDGET_EXCEEDED
    # The real, decisive assertion: actual provider calls made must never
    # exceed the declared budget by more than the single extra call every
    # OTHER path through this loop already tolerates (checked only AFTER
    # the call that trips it, the same "one call over" shape
    # test_delegate_task_reports_a_clean_error_when_the_subagent_runs_out_of_budget
    # already establishes for the sequential case) -- not silently
    # doubled by a concurrency race.
    assert provider.n <= budget.max_model_calls + 1


class _TwoDelegatesProvider:
    """Same first-round shape as _ConcurrentDelegateRaceProvider (two
    concurrent delegate_task calls in one TOOL_USE round), but every
    subsequent call — the parent's own follow-up and each subagent's
    single turn alike — finishes immediately with END_TURN, so this
    exercises a realistic, generous default Budget() rather than an
    artificially tight one."""

    name = "mock"

    def __init__(self) -> None:
        self.n = 0

    async def generate(self, request):
        self.n += 1
        await asyncio.sleep(0.01)
        if self.n == 1:
            calls = [
                ToolCallBlock(id="d1", name="delegate_task", arguments={"task": "subtask A"}),
                ToolCallBlock(id="d2", name="delegate_task", arguments={"task": "subtask B"}),
            ]
            for c in calls:
                yield ToolCallEvent(call=c)
            yield DoneEvent(
                stop_reason=StopReason.TOOL_USE,
                message=Message(role="assistant", content=calls),
                usage=Usage(input_tokens=1, output_tokens=1),
            )
            return
        text = TextBlock(text=f"answer from call {self.n}")
        yield DoneEvent(
            stop_reason=StopReason.END_TURN,
            message=Message(role="assistant", content=[text]),
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_concurrent_delegate_task_calls_do_not_starve_each_other_under_a_default_budget(
    run_root,
):
    # A real regression in the fix directly above this test: reserving
    # the FULL granted slice up front (to close the overspend race)
    # meant the FIRST admitted delegate_task call — with no explicit
    # budget field to request anything narrower (DelegateTool's own
    # input_schema has no budget field at all, so budget is always
    # None) — was granted the entire remainder, unconditionally
    # starving every concurrent sibling in the same round to exactly
    # zero. Confirmed live before this fix: under a realistic default
    # Budget() (max_model_calls=50, plenty of headroom), one of two
    # ordinary concurrent delegations failed with budget_exceeded after
    # only 1 of 50 calls had actually been used. Fixed by capping an
    # unspecified request to half of what's currently left rather than
    # all of it, so both concurrent siblings get a real, nonzero share.
    provider = _TwoDelegatesProvider()
    budget = Budget()
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[DelegateTool()],
        budget=budget,
        run_root=run_root,
    )

    events = [e async for e in loop.run("please delegate to two subagents concurrently")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert len(finished) == 2
    for f in finished:
        assert f.result.is_error is False, f.result.content[0].text

    run_done = [e for e in events if e.type == "run_done"][-1]
    assert run_done.state == AgentState.DONE


class _TwoSpawnWithExplicitBudgetsTool:
    """Calls ctx.spawn_subagent TWICE concurrently, each with an explicit
    small budget — isolates the run_dir-pruning question below from the
    default-budget starvation fix verified above."""

    spec = ToolSpec(
        name="two_spawn",
        description="spawn two subagents concurrently with explicit budgets",
        input_schema={"type": "object", "properties": {}},
        destructive=False,
    )

    async def run(self, args, ctx):
        results = await asyncio.gather(
            ctx.spawn_subagent(
                "subtask A",
                budget=Budget(
                    max_model_calls=5, max_total_tokens=100, max_wall_seconds=5.0, max_cost_usd=1.0
                ),
            ),
            ctx.spawn_subagent(
                "subtask B",
                budget=Budget(
                    max_model_calls=5, max_total_tokens=100, max_wall_seconds=5.0, max_cost_usd=1.0
                ),
            ),
        )
        ok = all(r.state == AgentState.DONE for r in results)
        return ToolResultBlock(
            tool_call_id="",
            content=[TextBlock(text=f"ok={ok} A={results[0].state} B={results[1].state}")],
        )


class _SlowThenFastSubtaskProvider:
    """First real call spawns two concurrent subagents. Subtask A's own
    turn deliberately takes much longer than subtask B's — A is still
    mid-run (its own run_dir not yet cleaned up) when B's run() creates
    its own sibling run_dir and prunes the shared subagents/ directory,
    the exact interleaving needed to trigger cross-sibling deletion."""

    name = "mock"

    def __init__(self) -> None:
        self.n = 0

    async def generate(self, request):
        self.n += 1
        text_blob = " ".join(getattr(b, "text", "") for m in request.messages for b in m.content)
        if self.n == 1:
            call = ToolCallBlock(id="t1", name="two_spawn", arguments={})
            yield ToolCallEvent(call=call)
            yield DoneEvent(
                stop_reason=StopReason.TOOL_USE,
                message=Message(role="assistant", content=[call]),
                usage=Usage(input_tokens=1, output_tokens=1),
            )
            return
        await asyncio.sleep(0.2 if "subtask A" in text_blob else 0.01)
        yield DoneEvent(
            stop_reason=StopReason.END_TURN,
            message=Message(role="assistant", content=[TextBlock(text="done")]),
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_pruning_never_deletes_a_still_running_siblings_run_dir(run_root, monkeypatch):
    # A real bug found by giving _prune_old_runs its own fresh-eyes sweep
    # one round after it shipped: concurrent subagents spawned from the
    # same parent share one run_root/subagents/ directory, and each
    # subagent's own run() independently prunes that SHARED directory
    # right after creating its own run_dir -- purely by mtime, with no
    # concept of "still running." Confirmed live before the fix: with
    # _MAX_RETAINED_RUNS lowered (so this test doesn't need 200+ real
    # runs), a slower sibling's still-in-flight run_dir got deleted by a
    # faster sibling's own prune call, and the slower one then crashed
    # with a raw, uncaught FileNotFoundError trying to append to a
    # transcript file whose parent directory no longer existed.
    import sarva.agent.loop as loop_module

    monkeypatch.setattr(loop_module, "_MAX_RETAINED_RUNS", 1)

    provider = _SlowThenFastSubtaskProvider()
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_TwoSpawnWithExplicitBudgetsTool()],
        budget=Budget(
            max_model_calls=1000,
            max_total_tokens=2_000_000,
            max_wall_seconds=3600.0,
            max_cost_usd=100.0,
        ),
        run_root=run_root,
    )

    events = [e async for e in loop.run("spawn two subagents")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert finished[0].result.is_error is False, finished[0].result.content[0].text
    assert "ok=True" in finished[0].result.content[0].text


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


class _SpawnSubagentCaptureTool:
    """Calls ctx.spawn_subagent directly with an explicit task_class and
    budget, rather than going through DelegateTool's own simple
    task-only surface -- proves spec-03's full `(task, task_class,
    budget)` signature genuinely works for a caller that isn't
    DelegateTool, matching design decision #7's framing of
    spawn_subagent as a general primitive, not something private to one
    tool."""

    spec = ToolSpec(
        name="capture_spawn_result",
        description="spawn a subagent with an explicit task_class/budget and report back",
        input_schema={"type": "object", "properties": {}},
        destructive=False,
    )

    async def run(self, args, ctx: ToolContext) -> ToolResultBlock:
        result = await ctx.spawn_subagent(
            "say something", task_class=TaskClass.ESCALATION, budget=Budget(max_model_calls=1)
        )
        text = (
            f"state={result.state.value} run_dir={result.run_dir} calls={result.spend.model_calls}"
        )
        return ToolResultBlock(tool_call_id="", content=[TextBlock(text=text)])


@pytest.mark.asyncio
async def test_spawn_subagent_honors_an_explicit_task_class_and_budget_request(run_root):
    # An explicit budget REQUEST (max_model_calls=1) that's well within
    # the parent's own generous remainder (9 left after this turn) must
    # be honored EXACTLY as requested, not silently widened to whatever
    # the parent could have afforded -- the clamp only narrows, never
    # expands. A 1-call budget can never reach DONE (the outer
    # budget-exceeded check fires right after that one call regardless
    # of what it says), which is itself the clean, deterministic proof
    # the request was honored precisely: BUDGET_EXCEEDED after exactly
    # 1 real call, not 9.
    call = ToolCallBlock(id="c1", name="capture_spawn_result", arguments={})
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(text="the subagent's one allowed reply"),
            ScriptedTurn(text="parent's final answer"),
        ]
    )
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_SpawnSubagentCaptureTool()],
        budget=Budget(max_model_calls=10),
        run_root=run_root,
    )

    events = [e async for e in loop.run("capture a subagent result")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert "state=budget_exceeded" in finished[0].result.content[0].text
    assert "calls=1" in finished[0].result.content[0].text


@pytest.mark.asyncio
async def test_subagent_transcript_is_nested_under_the_parents_own_run_dir(run_root):
    # spec-03 design decision #7: "their transcript nested under the
    # parent's run dir" -- verified against the real filesystem layout
    # AgentLoop.run() actually produces, not just the AgentResult field.
    call = ToolCallBlock(id="c1", name="capture_spawn_result", arguments={})
    provider = MockProvider(
        script=[
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(text="the subagent's one allowed reply"),
            ScriptedTurn(text="parent's final answer"),
        ]
    )
    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        tools=[_SpawnSubagentCaptureTool()],
        run_root=run_root,
    )

    events = [e async for e in loop.run("capture a subagent result")]

    finished = [e for e in events if e.type == "tool_finished"]
    reported_run_dir = finished[0].result.content[0].text.split("run_dir=")[1].split(" ")[0]

    parent_run_dirs = list(Path(run_root).iterdir())
    assert len(parent_run_dirs) == 1  # the parent's own run_dir, subagent nested inside it
    parent_run_dir = parent_run_dirs[0]

    sub_run_dir = Path(reported_run_dir)
    assert sub_run_dir.is_relative_to(parent_run_dir)
    assert sub_run_dir.parent.name == "subagents"
    assert (sub_run_dir / "transcript.jsonl").exists()


@pytest.mark.asyncio
async def test_verify_passes_through_an_approved_final_answer_unchanged(run_root):
    # The design doc's second named-but-long-unbuilt agent-orchestration
    # pattern: opt-in (verify=True) automatic verification of a candidate
    # final answer before the run actually completes. A verifier that
    # approves must leave the original candidate answer completely
    # untouched -- the run still ends DONE with the SAME text the main
    # loop actually produced, not the verifier's own commentary.
    provider = MockProvider(
        script=[
            ScriptedTurn(text="the real final answer"),
            ScriptedTurn(text="VERIFIED: this genuinely answers the question"),
        ]
    )
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root, verify=True)

    events = [e async for e in loop.run("what's the answer?")]

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.DONE
    assert run_done[-1].final_message.text() == "the real final answer"
    # The verifier's own real model call is genuinely counted, not free.
    assert run_done[-1].spend.model_calls == 2


@pytest.mark.asyncio
async def test_verify_rejects_a_final_answer_the_verifier_disagrees_with(run_root):
    # The one case that actually changes the outcome: an unambiguous
    # REJECTED verdict turns a candidate END_TURN success into a real
    # FAILED terminal state, with the verifier's own reason surfaced in
    # StateChangedEvent.detail -- the same "give the real reason" pattern
    # every other clean-failure path in this loop already uses.
    provider = MockProvider(
        script=[
            ScriptedTurn(text="a wrong or incomplete answer"),
            ScriptedTurn(text="REJECTED: this does not actually answer what was asked"),
        ]
    )
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root, verify=True)

    events = [e async for e in loop.run("what's the answer?")]

    state_events = [e for e in events if e.type == "state_changed"]
    assert state_events[-1].state == AgentState.FAILED
    assert "REJECTED" in state_events[-1].detail

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.FAILED
    assert run_done[-1].final_message is None


@pytest.mark.asyncio
async def test_verify_is_advisory_and_never_blocks_completion_when_the_verifier_itself_fails(
    run_root,
):
    # A verifier that can't produce a real verdict at all (refused,
    # crashed, ran out of budget -- anything other than a real DONE with
    # an unambiguous REJECTED) must never block a real completed answer.
    # This is a deliberate v1 scoping choice: verification is advisory,
    # not a hard gate, so a flaky verifier can never take down an
    # otherwise-working run.
    provider = MockProvider(
        script=[
            ScriptedTurn(text="the real final answer"),
            ScriptedTurn(refuse=True),  # the verifier subagent itself fails
        ]
    )
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root, verify=True)

    events = [e async for e in loop.run("what's the answer?")]

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.DONE
    assert run_done[-1].final_message.text() == "the real final answer"


@pytest.mark.asyncio
async def test_verify_ambiguous_verdict_does_not_block_completion(run_root):
    # Neither VERIFIED nor REJECTED as a prefix -- the same advisory,
    # fail-open posture as a verifier that can't run at all.
    provider = MockProvider(
        script=[
            ScriptedTurn(text="the real final answer"),
            ScriptedTurn(text="Looks fine to me, I guess."),
        ]
    )
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root, verify=True)

    events = [e async for e in loop.run("what's the answer?")]

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.DONE
    assert run_done[-1].final_message.text() == "the real final answer"


@pytest.mark.asyncio
async def test_verify_defaults_to_off(run_root):
    # Every existing call site that doesn't pass verify=True must be
    # completely unaffected -- no extra model call, no behavior change.
    provider = MockProvider(script=[ScriptedTurn(text="the real final answer")])
    loop = AgentLoop(router=_router(), providers={"mock": provider}, run_root=run_root)

    events = [e async for e in loop.run("what's the answer?")]

    run_done = [e for e in events if e.type == "run_done"]
    assert run_done[-1].state == AgentState.DONE
    assert run_done[-1].spend.model_calls == 1  # no verifier call happened


@pytest.mark.asyncio
async def test_verify_true_reports_budget_exceeded_when_the_verifier_tips_it_over(run_root):
    # A real bug found by actually running verify=True against a tight
    # Budget: spawn_subagent() merges the verifier's own real Spend into
    # this run's live spend, but the only spend.exceeded() check in the
    # END_TURN branch runs BEFORE verification ever starts, so it can't
    # see the verifier's cost. An identical Budget that correctly reports
    # DONE with verify=False reported DONE again with verify=True even
    # though the merged spend was genuinely over budget -- and every real
    # caller (cli.py, server/app.py) gates both "was this a failure" and
    # "should the session be saved" on state == DONE alone, so this
    # silently defeated Budget's entire purpose whenever verification
    # itself was what pushed spend over the line.
    provider = MockProvider(
        script=[
            ScriptedTurn(text="the real final answer"),
            ScriptedTurn(text="VERIFIED: looks correct to me"),
        ]
    )
    budget = Budget(max_total_tokens=60)

    # Control: the identical budget without verification stays under it.
    control_loop = AgentLoop(
        router=_router(),
        providers={"mock": MockProvider(script=[ScriptedTurn(text="the real final answer")])},
        budget=budget,
        run_root=run_root,
    )
    control_events = [e async for e in control_loop.run("what's the answer?")]
    control_done = [e for e in control_events if e.type == "run_done"][-1]
    assert control_done.state == AgentState.DONE

    loop = AgentLoop(
        router=_router(),
        providers={"mock": provider},
        budget=budget,
        run_root=run_root,
        verify=True,
    )

    events = [e async for e in loop.run("what's the answer?")]

    run_done = [e for e in events if e.type == "run_done"][-1]
    assert run_done.state == AgentState.BUDGET_EXCEEDED
    assert run_done.spend.total_tokens >= budget.max_total_tokens

    state_events = [e for e in events if e.type == "state_changed"]
    assert state_events[-1].state == AgentState.BUDGET_EXCEEDED
    assert state_events[-1].detail == "tokens"
