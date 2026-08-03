"""sarva.agent.loop — the agent loop: plan/act/verify via the provider layer.

One loop drives every skin. It calls the routed model, dispatches tool
calls (concurrently, gated by confirm policy for destructive tools),
enforces budgets, and yields a single typed AgentEvent stream. The
transcript is append-only JSONL so a run is inspectable and resumable.

T2: the loop is multimodal-aware at model selection — it scans the
initiating message(s) for the modalities actually present (text, image,
...) and routes to a model that supports all of them, instead of always
assuming text-only. Content-level degradation (e.g. auto-downgrading
video to sampled frames for a model that can't see video) is wired in
too, opt-in via the `degraders` constructor parameter: with none
supplied, a conversation needing an unsupported modality still fails
exactly as before; with degraders supplied, that failure becomes
recoverable (fall back to the best available model, degrade the
unsupported content into what it can actually see) instead of
automatic. See BUILD-JOURNAL.md for the entry that closed this gap.

T1 simplifications still true (documented, not hidden):
  * Tool-start/finish events for a batch of concurrent tool calls are
    yielded in two grouped passes (all-started, then all-finished) rather
    than truly interleaved in wall-clock order. Tools still execute
    concurrently via asyncio.gather; only the *event* ordering is batched.
    Tightened when the live-progress UI needs it (T3).
"""

from __future__ import annotations

import asyncio
import shutil
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
from sarva.agent.subagents import DelegateTool
from sarva.agent.tools import ConfirmPolicy, Tool, ToolContext, always_allow
from sarva.multimodal.content import (
    ContentBlock,
    Degrader,
    Message,
    Modality,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
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
from sarva.providers.registry import Router, TaskClass, UnknownModelError

# A real bug found by actually running a turn with one fast tool call
# and one that never returns: asyncio.gather withholds ALL results
# until EVERY coroutine finishes, so the hung tool blocked the entire
# turn forever -- silently discarding the fast tool's already-completed
# result too, confirmed live with a 5-second outer guard that never
# unblocked. RunShellTool self-protects with its own internal 60s
# timeout (core/sarva/agent/tools.py), but that's the only built-in
# tool that does; McpToolAdapter.run() (a remote or stdio MCP server's
# call_tool) has none at all, so any hung/unresponsive MCP server had no
# recovery path whatsoever. 90s (longer than RunShellTool's own 60s) so
# a tool with its own internal timeout gets to fire first and produce
# its own specific message -- this is purely a backstop for tools that
# don't self-protect, not a replacement for RunShellTool's own timeout.
_TOOL_TIMEOUT_SECONDS = 90

# A real bug found by actually driving a MockProvider that always yields
# a retryable StreamErrorEvent (simulating a provider stuck returning
# rate-limit/5xx responses indefinitely): AgentLoop.run()'s retryable-
# error branch (`if pevent.retryable: ... await asyncio.sleep(1.0);
# break`) falls through to `if done is None: continue`, jumping straight
# back to the top of the main `while True` loop -- BEFORE the
# `spend.wall_seconds = ...` / `spend.exceeded(self._budget)` block a
# few lines below ever runs. Confirmed live: with a real Budget
# (max_model_calls=50, max_wall_seconds=3600.0) supplied, the run never
# reached a terminal state after 6 real seconds and 12 real API-call
# attempts -- it spins forever, one real provider call per second, with
# no crash and no distinguishable log signal, regardless of how tight
# the caller's own budget is. Every OTHER path through this loop is
# bounded by `spend.exceeded`; this is the one gap where a stuck
# provider (not a stuck tool -- that's `_TOOL_TIMEOUT_SECONDS`'s own,
# separate concern) burns real API cost and wall-clock indefinitely.
# Deliberately a small, separate counter rather than folding into
# `spend.model_calls` -- a retry isn't a new model call (the existing
# `spend.model_calls -= 1` comment already establishes that), so
# capping it needs its own, independent bound rather than repurposing
# a counter that means something else.
_MAX_STREAM_RETRIES = 5

# A real bug found by actually running `AgentLoop.run()` in a loop the way
# `sarva.server.app`'s /chat and /ws/chat handlers do (a fresh AgentLoop
# per request): every single call creates a new `run_root/<run_id>/`
# directory and never deletes it, no matter how the run ends. Confirmed
# live: 25 ordinary, non-adversarial turns against a real router left 25
# directories on disk with zero cleanup anywhere in this codebase --
# unbounded growth over the lifetime of a long-running `sarva serve`
# process, plus indefinite retention of full transcript content (message
# text, tool arguments, tool results) with no user-visible retention
# policy, the same class of unintended local persistence this project's
# own config.json permissions fix (see BUILD-JOURNAL.md) already treated
# as worth hardening. Pruned to the most recent `_MAX_RETAINED_RUNS`
# directories (by mtime) on every new run rather than removed outright --
# `AgentLoop`'s own docstring calls a run "inspectable," so some retention
# window is the intended behavior, just not an unbounded one.
_MAX_RETAINED_RUNS = 200


def _prune_old_runs(run_root: Path, keep: int) -> None:
    try:
        run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    except OSError:
        return
    if len(run_dirs) <= keep:
        return
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    for old_dir in run_dirs[: len(run_dirs) - keep]:
        shutil.rmtree(old_dir, ignore_errors=True)


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
        session_id: str | None = None,
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
        alone only ever carries the *last* assistant turn.

        `session_id`, if given, flows straight into `ToolContext` so
        session-aware tools (memory recall/remember) can scope themselves
        to the actual conversation this run belongs to — the CLI's
        `--session` value / the server's `session` request field, not a
        run-scoped identifier like `run_id` below (a fresh UUID every
        call, unrelated to which saved conversation this is)."""
        run_id = uuid.uuid4().hex[:12]
        run_root = Path(self._run_root)
        run_dir = run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _prune_old_runs(run_root, keep=_MAX_RETAINED_RUNS)
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
        stream_retries = 0
        state = AgentState.INIT
        final_message: Message | None = None

        try:
            model = self._router.pick(
                self._task_class,
                needs=_required_modalities(messages),
                override=model_override,
            )
        except UnknownModelError as e:
            # An explicit model_override that doesn't name a real
            # registered model -- a hard user error, handled completely
            # separately from the LookupError branch below (no degrader
            # fallback attempted at all): silently routing to some OTHER
            # model in place of the one the caller explicitly asked for
            # by id would be a materially misleading response to a
            # request that was never ambiguous in the first place.
            state = AgentState.FAILED
            yield await emit(StateChangedEvent(state=state, detail=str(e)))
            yield await emit(RunDoneEvent(state=state, final_message=None, spend=spend))
            if transcript_out is not None:
                transcript_out.extend(messages)
            return
        except LookupError as e:
            # No available model supports what this conversation needs (e.g.
            # an image with no vision-capable model configured). With
            # degraders configured, this is recoverable: fall back to the
            # best available text-capable model and degrade the messages
            # down to what it actually supports, rather than failing
            # outright. (router.pick's `override`, when set, either
            # returns a real model or raises UnknownModelError above --
            # never LookupError -- so reaching this branch at all means
            # model_override was None, and there's no explicit model
            # choice this fallback could contradict.)
            model = None
            degrade_error: Exception | None = None
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
                except Exception as degrade_exc:
                    # A real bug found by actually degrading a corrupt/
                    # truncated image with no vision-capable model
                    # configured: a concrete Degrader (e.g.
                    # ImageToTextDegrader) can raise its own decode error
                    # when it genuinely can't make sense of the content --
                    # LookupError/UnsupportedModalityError alone (the only
                    # two this except clause used to catch) don't cover
                    # that, so it propagated straight out of this async
                    # generator uncaught instead of the clean FAILED state
                    # this whole fallback exists to produce. Caught broadly
                    # and deliberately: this block's own purpose is
                    # "attempt a best-effort recovery, and if it doesn't
                    # work for ANY reason, fail cleanly" -- not to
                    # enumerate every exception type a current or future
                    # Degrader implementation might raise.
                    model = None  # degradation itself couldn't help either; fall through to FAILED
                    degrade_error = degrade_exc

            if model is None:
                # This is an INIT-time failure the frozen LEGAL table doesn't
                # model as a transition (it has no predecessor state to
                # violate) — handled directly rather than via transition(),
                # which doesn't exist yet at this point in the function.
                state = AgentState.FAILED
                # The degradation attempt's own failure (e.g. "could not
                # decode image for degradation: ...") is a more specific,
                # actionable reason than the original "no model supports
                # this modality" LookupError -- surface it when present,
                # the same "give the real reason" discipline every other
                # clean-failure path in this project already applies.
                detail = str(degrade_error) if degrade_error is not None else str(e)
                yield await emit(StateChangedEvent(state=state, detail=detail))
                yield await emit(RunDoneEvent(state=state, final_message=None, spend=spend))
                if transcript_out is not None:
                    transcript_out.extend(messages)
                return
        provider = self._providers[model.provider]

        async def spawn_subagent(subtask: str) -> Message | None:
            # Subagent fan-out (docs/agent-loop.md's own long-named,
            # previously-unbuilt gap). Bounded by what's actually left of
            # THIS run's own budget, not a fresh independent one -- computed
            # here (not once at __init__ time) because `spend` keeps
            # changing across the parent's own model calls and tool
            # dispatches, and a subagent spawned late in a long run must
            # see what's genuinely left, not what was available at the
            # very start.
            remaining_calls = self._budget.max_model_calls - spend.model_calls
            remaining_wall = self._budget.max_wall_seconds - spend.wall_seconds
            if remaining_calls <= 0 or remaining_wall <= 0:
                return None
            sub_tools = [t for t in self._tools.values() if not isinstance(t, DelegateTool)]
            sub_loop = AgentLoop(
                router=self._router,
                providers=self._providers,
                tools=sub_tools,
                confirm=self._confirm,
                budget=Budget(
                    max_model_calls=remaining_calls,
                    max_total_tokens=max(0, self._budget.max_total_tokens - spend.total_tokens),
                    max_wall_seconds=remaining_wall,
                    max_cost_usd=max(0.0, self._budget.max_cost_usd - spend.cost_usd),
                ),
                task_class=TaskClass.SUBTASK,
                workdir=self._workdir,
                run_root=self._run_root,
                degraders=self._degraders,
            )
            sub_final: Message | None = None
            sub_state = AgentState.FAILED
            async for sub_event in sub_loop.run(subtask, session_id=session_id):
                if sub_event.type == "run_done":
                    sub_final = sub_event.final_message
                    sub_state = sub_event.state
                    # Merged back into the PARENT's own live spend so the
                    # very next spend.exceeded(self._budget) check below
                    # reflects the subagent's real cost too -- without this
                    # a model could spawn subagents to bypass its own
                    # budget entirely; confirmed live before fixing it this
                    # way (see BUILD-JOURNAL.md).
                    spend.model_calls += sub_event.spend.model_calls
                    spend.total_tokens += sub_event.spend.total_tokens
                    spend.wall_seconds += sub_event.spend.wall_seconds
                    spend.cost_usd += sub_event.spend.cost_usd
            return sub_final if sub_state == AgentState.DONE else None

        ctx = ToolContext(
            workdir=self._workdir,
            run_dir=str(run_dir),
            emit=emit,
            session_id=session_id,
            spawn_subagent=spawn_subagent,
        )

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
                            stream_retries += 1
                            if stream_retries > _MAX_STREAM_RETRIES:
                                transition(AgentState.FAILED)
                                yield await emit(
                                    StateChangedEvent(
                                        state=state,
                                        detail=(
                                            f"gave up after {_MAX_STREAM_RETRIES} retryable "
                                            f"stream errors: {pevent.detail}"
                                        ),
                                    )
                                )
                                spend.wall_seconds = time.monotonic() - started
                                yield await emit(
                                    RunDoneEvent(state=state, final_message=None, spend=spend)
                                )
                                if transcript_out is not None:
                                    transcript_out.extend(messages)
                                return
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

            stream_retries = 0  # a real response arrived; past retries no longer count
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
            # Keyed by the ToolCallBlock's own Python object identity, not
            # its `.id` string. A real bug found by actually constructing
            # two distinct ToolCallBlocks that share the same `.id` (a
            # malformed/adversarial model turn -- nothing validates id
            # uniqueness before this point): keying by `call.id` meant
            # the second confirmation silently overwrote the first
            # entry, so `run_one`'s `approvals.get(call.id, False)` read
            # back the SAME (last-decided) answer for both calls --
            # confirmed live, an explicitly-declined destructive call ran
            # anyway because a different call sharing its id happened to
            # be approved. Every `ToolCallBlock` in `calls`/
            # `destructive_calls` is the same Python object referenced in
            # both this loop and `run_one` below (list comprehensions
            # preserve identity), so `id(call)` is a real per-call key
            # that can never collide the way the model-supplied string
            # can.
            approvals: dict[int, bool] = {}
            if destructive_calls:
                transition(AgentState.AWAITING_CONFIRMATION)
                yield await emit(StateChangedEvent(state=state))
                for call in destructive_calls:
                    yield await emit(NeedsConfirmationEvent(call=call))
                    approvals[id(call)] = await self._confirm(call)
                transition(AgentState.RUNNING_TOOLS)
                yield await emit(StateChangedEvent(state=state))

            async def run_one(
                call: ToolCallBlock, approvals: dict[int, bool] = approvals
            ) -> tuple[ToolResultBlock, float]:
                tool = self._tools.get(call.name)
                t0 = time.monotonic()
                if tool is None:
                    result = ToolResultBlock(
                        tool_call_id=call.id,
                        content=[TextBlock(text=f"unknown tool: {call.name}")],
                        is_error=True,
                    )
                elif tool.spec.destructive and not approvals.get(id(call), False):
                    result = ToolResultBlock(
                        tool_call_id=call.id,
                        content=[TextBlock(text="declined by user")],
                        is_error=True,
                    )
                else:
                    try:
                        raw = await asyncio.wait_for(
                            tool.run(call.arguments, ctx), timeout=_TOOL_TIMEOUT_SECONDS
                        )
                        result = raw.model_copy(update={"tool_call_id": call.id})
                    except TimeoutError:
                        # A hung tool call is scored the same way a raised
                        # exception already is -- a real, visible
                        # is_error=True result the model sees and can react
                        # to -- rather than blocking every OTHER concurrent
                        # tool call's already-completed result forever, the
                        # real bug this closes.
                        result = ToolResultBlock(
                            tool_call_id=call.id,
                            content=[
                                TextBlock(
                                    text=f"tool call timed out after {_TOOL_TIMEOUT_SECONDS}s"
                                )
                            ],
                            is_error=True,
                        )
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
