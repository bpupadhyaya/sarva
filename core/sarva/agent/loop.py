"""sarva.agent.loop — the agent loop: plan/act/verify via the provider layer.

One loop drives every skin. It calls the routed model, dispatches tool
calls (concurrently, gated by confirm policy for destructive tools),
enforces budgets, and yields a single typed AgentEvent stream. The
transcript is append-only JSONL so a run is inspectable and resumable.

T1 simplifications (documented, not hidden):
  * Multimodal degradation is not yet wired here — T1 tools are text-only,
    so messages already satisfy every routed model's `modalities_in`. Real
    wiring lands with the multimodal I/O pipeline (T2).
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
from sarva.multimodal.content import Message, TextBlock, ToolCallBlock, ToolResultBlock
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    Provider,
    StopReason,
    StreamErrorEvent,
    ToolSpec,
)
from sarva.providers.registry import Router, TaskClass


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
    ):
        self._router = router
        self._providers: dict[str, Provider] = providers  # type: ignore[assignment]
        self._tools: dict[str, Tool] = {t.spec.name: t for t in (tools or [])}
        self._confirm = confirm
        self._budget = budget or Budget()
        self._task_class = task_class
        self._workdir = workdir
        self._run_root = run_root

    def _tool_specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def run(
        self,
        task: str,
        history: list[Message] | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        run_id = uuid.uuid4().hex[:12]
        run_dir = Path(self._run_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = run_dir / "transcript.jsonl"

        async def emit(event: AgentEvent) -> AgentEvent:
            with transcript_path.open("a") as f:
                f.write(event.model_dump_json() + "\n")
            return event

        messages: list[Message] = list(history or []) + [
            Message(role="user", content=[TextBlock(text=task)])
        ]
        spend = Spend()
        started = time.monotonic()
        state = AgentState.INIT
        final_message: Message | None = None

        model = self._router.pick(self._task_class, override=model_override)
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
                        return
            except Exception as e:  # a provider crash never propagates to the skin
                transition(AgentState.FAILED)
                yield await emit(StateChangedEvent(state=state, detail=str(e)))
                spend.wall_seconds = time.monotonic() - started
                yield await emit(RunDoneEvent(state=state, final_message=None, spend=spend))
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
            messages.append(done.message)
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
