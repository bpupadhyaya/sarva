"""sarva.agent.loop — the agent loop: plan/act/verify via the provider layer.

One loop drives every skin. It calls the routed model, dispatches tool
calls (concurrently, gated by confirm policy for destructive tools),
enforces budgets, and yields a single typed AgentEvent stream. The
transcript is append-only JSONL so a run is inspectable and resumable.

T2: the loop is now multimodal-aware at model selection — it scans the
initiating message(s) for the modalities actually present (text, image,
...) and routes to a model that supports all of them, instead of always
assuming text-only. Full content-level degradation (e.g. auto-downgrading
video to sampled frames for a model that can't see video) still lands with
the multimodal I/O pipeline; T2 wires *routing*, not yet *degradation*.

T1 simplifications still true (documented, not hidden):
  * Tool-start/finish events for a batch of concurrent tool calls are
    yielded in two grouped passes (all-started, then all-finished) rather
    than truly interleaved in wall-clock order. Tools still execute
    concurrently via asyncio.gather; only the *event* ordering is batched.
    Tightened when the live-progress UI needs it (T3).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from sarva.agent.budget import Budget, Spend
from sarva.agent.events import (
    LEGAL,
    AgentEvent,
    AgentState,
    ModelStreamEvent,
    NeedsConfirmationEvent,
    RunDoneEvent,
    StateChangedEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)
from sarva.agent.tools import ConfirmPolicy, Tool, ToolContext, always_allow
from sarva.multimodal.content import (
    ContentBlock,
    Degrader,
    Message,
    Modality,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    UnsupportedModalityError,
    degrade_message,
    modality_of,
)
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    Provider,
    StopReason,
    StreamErrorEvent,
    ToolSpec,
)
from sarva.providers.registry import Router, TaskClass


def _required_modalities(messages: list[Message]) -> set[Modality]:
    """What the routed model must support, computed from what's actually in
    the conversation so far. Always includes TEXT — every current model
    handles it, and it keeps the set non-empty for the router's subset check."""
    needed = {Modality.TEXT}
    for m in messages:
        for block in m.content:
            needed.add(modality_of(block))
    return needed


class AgentLoop:
    def __init__(
        self,
        router: Router,
        providers: dict[str, object],
        tools: list[Tool] | None = None,
        confirm: ConfirmPolicy = always_allow,
        budget: Budget | None = None,
        task_class: TaskClass = TaskClass.MAIN,
        workdir: str = ".",
        run_root: str = ".sarva/runs",
        degraders: dict[Modality, Degrader] | None = None,
    ):
        self._router = router
        self._providers: dict[str, Provider] = providers  # type: ignore[assignment]
        self._tools: dict[str, Tool] = {t.spec.name: t for t in (tools or [])}
        self._confirm = confirm
        self._budget = budget or Budget()
        self._task_class = task_class
        self._workdir = workdir
        self._run_root = run_root
        # Opt-in: without a degrader for a modality, a conversation needing
        # it still fails exactly as before if no model supports it. With
        # one supplied, a model that can't see the modality at all is a
        # *recoverable* condition (route to the best available model,
        # degrade the unsupported content into something it *can* see)
        # instead of an automatic hard failure — deliberately opt-in
        # rather than a silent default, so nobody gets a lower-fidelity
        # response than they asked for without having asked for exactly
        # that tradeoff. See BUILD-JOURNAL.md for why this is separate
        # from T2's modality-aware *routing*.
        self._degraders = degraders or {}

    def _tool_specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def run(
        self,
        task: str,
        history: list[Message] | None = None,
        model_override: str | None = None,
        extra_content: list[ContentBlock] | None = None,
        transcript_out: list[Message] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """`extra_content` attaches non-text blocks (e.g. an ImageBlock) to
        the initiating user turn alongside `task`'s text — purely additive,
        every existing text-only call site is unaffected.

        `transcript_out`, if given, is extended in place with the complete
        final message list (history + every turn appended this run,
        including intermediate tool-use/tool-result messages) once the run
        reaches any terminal state. This is the only way to recover a
        tool-using run's full history for session persistence without
        changing the frozen RunDoneEvent shape — `RunDoneEvent.final_message`
        alone only ever carries the *last* assistant turn."""
        run_id = uuid.uuid4().hex[:12]
        run_dir = Path(self._run_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = run_dir / "transcript.jsonl"

        async def emit(event: AgentEvent) -> AgentEvent:
            with transcript_path.open("a") as f:
                f.write(event.model_dump_json() + "\n")
            return event

        messages: list[Message] = list(history or []) + [
            Message(role="user", content=[TextBlock(text=task), *(extra_content or [])])
        ]
        spend = Spend()
        started = time.monotonic()
        state = AgentState.INIT
        final_message: Message | None = None

        try:
            model = self._router.pick(
                self._task_class,
                needs=_required_modalities(messages),
                override=model_override,
            )
        except LookupError as e:
            # No available model supports what this conversation needs (e.g.
            # an image with no vision-capable model configured). With
            # degraders configured, this is recoverable: fall back to the
            # best available text-capable model and degrade the messages
            # down to what it actually supports, rather than failing
            # outright. (router.pick's `override`, when set, always
            # short-circuits with no modality check at all — reaching this
            # except block at all means model_override was None, so there
            # is no explicit model choice this fallback could contradict.)
            model = None
            if self._degraders:
                try:
                    fallback_model = self._router.pick(self._task_class, needs={Modality.TEXT})
                    messages = [
                        await degrade_message(
                            m,
                            supported=fallback_model.capabilities.modalities_in,
                            degraders=self._degraders,
                        )
                        for m in messages
                    ]
                    model = fallback_model
                except (LookupError, UnsupportedModalityError):
                    model = None  # degradation itself couldn't help either; fall through to FAILED

            if model is None:
                # This is an INIT-time failure the frozen LEGAL table doesn't
                # model as a transition (it has no predecessor state to
                # violate) — handled directly rather than via transition(),
                # which doesn't exist yet at this point in the function.
                state = AgentState.FAILED
                yield await emit(StateChangedEvent(state=state, detail=str(e)))
                yield await emit(RunDoneEvent(state=state, final_message=None, spend=spend))
                if transcript_out is not None:
                    transcript_out.extend(messages)
                return
        provider = self._providers[model.provider]
        ctx = ToolContext(workdir=self._workdir, run_dir=str(run_dir), emit=emit)

        def transition(to: AgentState) -> None:
            nonlocal state
            assert to in LEGAL.get(state, set()), f"illegal transition {state} -> {to}"
            state = to

        while True:
            transition(AgentState.CALLING_MODEL)
            yield await emit(StateChangedEvent(state=state))

            request = GenerateRequest(model=model.id, messages=messages, tools=self._tool_specs())

            spend.model_calls += 1
            done: DoneEvent | None = None
            try:
                async for pevent in provider.generate(request):
                    yield await emit(ModelStreamEvent(event=pevent))
                    if isinstance(pevent, DoneEvent):
                        done = pevent
                    elif isinstance(pevent, StreamErrorEvent):
                        if pevent.retryable:
                            spend.model_calls -= 1  # a retry isn't a new call
                            await asyncio.sleep(1.0)
                            break
                        transition(AgentState.FAILED)
                        yield await emit(StateChangedEvent(state=state, detail=pevent.detail))
                        spend.wall_seconds = time.monotonic() - started
                        yield await emit(RunDoneEvent(state=state, final_message=None, spend=spend))
                        if transcript_out is not None:
                            transcript_out.extend(messages)
                        return
            except Exception as e:  # a provider crash never propagates to the skin
                transition(AgentState.FAILED)
                yield await emit(StateChangedEvent(state=state, detail=str(e)))
                spend.wall_seconds = time.monotonic() - started
                yield await emit(RunDoneEvent(state=state, final_message=None, spend=spend))
                if transcript_out is not None:
                    transcript_out.extend(messages)
                return

            if done is None:
                continue  # transient stream error was retried; loop back to CALLING_MODEL

            spend.total_tokens += done.usage.input_tokens + done.usage.output_tokens
            spend.cost_usd += done.usage.cost_usd
            spend.wall_seconds = time.monotonic() - started

            reason = spend.exceeded(self._budget)
            if reason:
                transition(AgentState.BUDGET_EXCEEDED)
                yield await emit(StateChangedEvent(state=state, detail=reason))
                break

            # Appended once here, unconditionally, regardless of stop_reason —
            # `messages` (and therefore `transcript_out`) must reflect every
            # assistant turn that actually happened, not just the ones on the
            # tool-use path. A prior version only appended inside the
            # TOOL_USE branch below, silently dropping the final turn from
            # the recorded history on every successful (END_TURN) run.
            messages.append(done.message)

            if done.stop_reason == StopReason.END_TURN:
                final_message = done.message
                transition(AgentState.DONE)
                yield await emit(StateChangedEvent(state=state))
                break

            if done.stop_reason == StopReason.MAX_TOKENS:
                transition(AgentState.FAILED)
                yield await emit(StateChangedEvent(state=state, detail="truncated: max_tokens"))
                break

            if done.stop_reason == StopReason.REFUSAL:
                transition(AgentState.FAILED)
                yield await emit(StateChangedEvent(state=state, detail="refusal"))
                break

            # StopReason.TOOL_USE
            calls = [b for b in done.message.content if isinstance(b, ToolCallBlock)]

            transition(AgentState.RUNNING_TOOLS)
            yield await emit(StateChangedEvent(state=state))

            destructive_calls = [
                c
                for c in calls
                if self._tools.get(c.name) is not None and self._tools[c.name].spec.destructive
            ]
            approvals: dict[str, bool] = {}
            if destructive_calls:
                transition(AgentState.AWAITING_CONFIRMATION)
                yield await emit(StateChangedEvent(state=state))
                for call in destructive_calls:
                    yield await emit(NeedsConfirmationEvent(call=call))
                    approvals[call.id] = await self._confirm(call)
                transition(AgentState.RUNNING_TOOLS)
                yield await emit(StateChangedEvent(state=state))

            async def run_one(
                call: ToolCallBlock, approvals: dict[str, bool] = approvals
            ) -> tuple[ToolResultBlock, float]:
                tool = self._tools.get(call.name)
                t0 = time.monotonic()
                if tool is None:
                    result = ToolResultBlock(
                        tool_call_id=call.id,
                        content=[TextBlock(text=f"unknown tool: {call.name}")],
                        is_error=True,
                    )
                elif tool.spec.destructive and not approvals.get(call.id, False):
                    result = ToolResultBlock(
                        tool_call_id=call.id,
                        content=[TextBlock(text="declined by user")],
                        is_error=True,
                    )
                else:
                    try:
                        raw = await tool.run(call.arguments, ctx)
                        result = raw.model_copy(update={"tool_call_id": call.id})
                    except Exception as e:  # a tool error never crashes the loop
                        result = ToolResultBlock(
                            tool_call_id=call.id,
                            content=[TextBlock(text=f"{type(e).__name__}: {e}")],
                            is_error=True,
                        )
                return result, time.monotonic() - t0

            for call in calls:
                yield await emit(ToolStartedEvent(call=call))
            outcomes = list(await asyncio.gather(*(run_one(c) for c in calls)))
            results = [r for r, _ in outcomes]
            for result, seconds in outcomes:
                yield await emit(ToolFinishedEvent(result=result, seconds=seconds))

            messages.append(Message(role="user", content=list(results)))

        spend.wall_seconds = time.monotonic() - started
        yield await emit(RunDoneEvent(state=state, final_message=final_message, spend=spend))
        if transcript_out is not None:
            transcript_out.extend(messages)
