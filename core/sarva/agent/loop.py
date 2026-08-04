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
    AgentResult,
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
    # Never prune a directory whose run is still genuinely in flight --
    # see `AgentLoop.run`'s own `.active` marker above for the real bug
    # this guards against. `keep` still bounds the TOTAL directory count
    # (active + finished) the same as before -- a first attempt at this
    # fix compared `keep` against only the non-active count, which made
    # the currently-being-created directory (always active at this exact
    # point, since the wrapper marks it before calling in here) invisible
    # to its own retention math, silently growing steady-state disk usage
    # to keep+1 instead of keep (caught immediately by this file's own
    # pre-existing retention-cap test going from 3 to 4). Only the actual
    # REMOVAL set is restricted to finished (non-active) directories --
    # if too few of those exist to reach `keep`, the active ones are left
    # alone and total retention temporarily exceeds `keep` rather than
    # ever deleting something still in use; it self-corrects once they
    # finish.
    prunable = [p for p in run_dirs if not (p / ".active").exists()]
    prunable.sort(key=lambda p: p.stat().st_mtime)
    for old_dir in prunable[: len(run_dirs) - keep]:
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
        verify: bool = False,
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
        # The design doc's second named-but-long-unbuilt agent-orchestration
        # pattern (subagent fan-out was the first, see docs/agent-loop.md):
        # a "verifier subagent" that checks a candidate final answer against
        # the original task before the run actually completes. Opt-in, same
        # posture as `degraders` -- every existing call site that doesn't
        # pass this is completely unaffected. See the verification block
        # inside `run()`'s END_TURN handling for the full scoping notes
        # (advisory-only for this first slice: a verifier that can't run,
        # times out, or gives an ambiguous verdict never blocks a real
        # completed answer -- only an explicit REJECTED does).
        self._verify = verify

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
        run_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Thin wrapper around `_run_impl` that marks this run's directory
        "active" for the wrapper's entire lifetime -- a real bug found by
        giving `_prune_old_runs` (above) its own fresh-eyes sweep one round
        later: concurrent subagents spawned from the same parent share one
        `run_root/subagents/` directory (see `spawn_subagent` below), and
        each subagent's own `run()` independently prunes THAT shared
        directory right after creating its own run_dir -- purely by mtime,
        with no concept of "still running." Confirmed live with
        `_MAX_RETAINED_RUNS` lowered the same way this project's own retention
        test already does: a second, still-in-flight sibling subagent's run_dir
        (older by mtime, since it started first but was doing more real work)
        got deleted out from under it while its own transcript was still being
        written. A `.active` marker file, created here before `_run_impl` ever
        starts and removed in `finally` however the run ends (success,
        exception, or the caller closing the generator early), lets pruning
        skip any directory genuinely still in use -- regardless of which
        level (a top-level run or a nested subagent) created it, and
        regardless of how many concurrent siblings share the directory.
        `_prune_old_runs` still bounds unbounded growth of directories no
        one is using; it just no longer treats "hasn't been touched
        recently" as synonymous with "safe to delete." """
        run_id = run_id or uuid.uuid4().hex[:12]
        run_dir = Path(self._run_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        active_marker = run_dir / ".active"
        active_marker.touch()
        try:
            async for event in self._run_impl(
                task,
                history=history,
                model_override=model_override,
                extra_content=extra_content,
                transcript_out=transcript_out,
                session_id=session_id,
                run_id=run_id,
            ):
                yield event
        finally:
            active_marker.unlink(missing_ok=True)

    async def _run_impl(
        self,
        task: str,
        history: list[Message] | None = None,
        model_override: str | None = None,
        extra_content: list[ContentBlock] | None = None,
        transcript_out: list[Message] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
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
        call, unrelated to which saved conversation this is).

        `run_id`, if given, is used as-is instead of generating a fresh
        UUID -- the one thing `spawn_subagent` (below) needs to compute a
        subagent's real `run_dir` path *before* awaiting its run, so
        `AgentResult.run_dir` reports where the transcript genuinely
        landed rather than a placeholder. Every other caller leaves this
        unset and gets the same fresh-UUID behavior as before."""
        run_id = run_id or uuid.uuid4().hex[:12]
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

        # A real bug found by actually dispatching TWO concurrent
        # delegate_task calls in one TOOL_USE round (ordinary usage --
        # "delegate these two independent things in parallel" -- not an
        # adversarial trick): `asyncio.gather` (below) runs every tool
        # call in a round concurrently, and `spawn_subagent`'s own budget
        # clamp reads `spend` synchronously before its first `await`. Two
        # concurrent calls both read the SAME, not-yet-decremented
        # `spend` (real mutation only happened once a subagent's ENTIRE
        # run had already finished), so each got granted a full
        # independent slice of the same remaining budget -- confirmed
        # live: `Budget(max_model_calls=3)` with two concurrent
        # delegate_task calls made 6 real provider calls, double the
        # declared cap, with the parent still reporting a coherent (if
        # too-late) `spend.model_calls == 6`. This lock serializes just
        # the grant decision (read remaining budget, decide the slice,
        # immediately reserve it against `spend`) so a second concurrent
        # caller sees the correctly-reduced remainder -- the subagents
        # themselves still run fully concurrently afterward, only the
        # admission decision is serialized. The reservation is
        # reconciled down to the subagent's REAL usage once it finishes,
        # so the final reported spend still reflects true cost, not the
        # conservative up-front grant.
        spend_lock = asyncio.Lock()

        async def spawn_subagent(
            subtask: str,
            task_class: TaskClass = TaskClass.SUBTASK,
            budget: Budget | None = None,
        ) -> AgentResult:
            # Subagent fan-out -- spec-03's own frozen signature
            # (`spawn_subagent: Callable[..., Awaitable[AgentResult]]`,
            # documented there as `(task, task_class, budget)`) and design
            # decision #7 ("subagents are just recursive loops with their
            # own budget slice ... and their transcript nested under the
            # parent's run dir"), scaffolded since Spec 03 was frozen via
            # the unused `AgentResult` type in events.py but never actually
            # implemented until docs/agent-loop.md's own long-named gap was
            # finally picked up. `budget`, if given, is a REQUEST, not a
            # grant: every field is clamped to what's actually left of
            # THIS run's own budget -- computed fresh here (not once at
            # __init__ time) because `spend` keeps changing across the
            # parent's own model calls and tool dispatches, and a subagent
            # spawned late in a long run must see what's genuinely left,
            # not what was available at the very start. Matches spec-03's
            # own conformance invariant #8 verbatim: "its budget slice
            # cannot exceed the parent's remainder."
            #
            # A real bug found by actually dispatching two ordinary
            # concurrent `delegate_task` calls under a realistic, generous
            # default `Budget()` (50 calls, plenty of headroom): the FIRST
            # admitted call was granted (and immediately reserved) the
            # ENTIRE remainder -- correct in isolation, but `DelegateTool`
            # never passes an explicit `budget`, so this happened on
            # *every* call, unconditionally starving every sibling in the
            # same round to exactly zero regardless of how much headroom
            # actually existed, and reporting a factually misleading
            # "budget_exceeded" for the starved one. Fixed by capping an
            # UNSPECIFIED request (no explicit `budget` argument at all)
            # to half of whatever is currently left, rather than all of
            # it: a lone delegation still gets a generous share (half the
            # remainder), and each additional concurrent one gets half of
            # what's left AFTER its predecessors' reservations -- a
            # geometric split that approaches, but mathematically never
            # reaches, zero, so no ordinary number of concurrent
            # delegations can ever starve one to nothing outright.
            # Deliberately scoped to the `budget is None` case only: an
            # EXPLICIT `Budget(...)` request (no real caller makes one
            # today -- `DelegateTool` has no budget field in its own
            # schema, and `verify=True`'s own call never runs concurrently
            # with anything else) is still honored exactly, unhalved,
            # matching this file's own pre-existing, already-tested
            # "explicit tight budget honored precisely" contract.
            def _clamp(requested: float | None, remaining: float) -> float:
                return max(0.0, min(requested, remaining) if requested is not None else remaining)

            _UNSPECIFIED_SHARE = 0.5

            async with spend_lock:
                remaining_calls = self._budget.max_model_calls - spend.model_calls
                remaining_tokens = self._budget.max_total_tokens - spend.total_tokens
                remaining_wall = self._budget.max_wall_seconds - spend.wall_seconds
                remaining_cost = self._budget.max_cost_usd - spend.cost_usd
                sub_budget = Budget(
                    max_model_calls=int(
                        _clamp(
                            budget.max_model_calls
                            if budget
                            else remaining_calls * _UNSPECIFIED_SHARE,
                            remaining_calls,
                        )
                    ),
                    max_total_tokens=int(
                        _clamp(
                            budget.max_total_tokens
                            if budget
                            else remaining_tokens * _UNSPECIFIED_SHARE,
                            remaining_tokens,
                        )
                    ),
                    max_wall_seconds=_clamp(
                        budget.max_wall_seconds if budget else remaining_wall * _UNSPECIFIED_SHARE,
                        remaining_wall,
                    ),
                    max_cost_usd=_clamp(
                        budget.max_cost_usd if budget else remaining_cost * _UNSPECIFIED_SHARE,
                        remaining_cost,
                    ),
                )
                # A real bug found by a fresh-eyes sweep: this gate only
                # ever checked 2 of `Budget`'s 4 dimensions --
                # `max_total_tokens`/`max_cost_usd` are clamped exactly
                # the same way as the two checked here (and can round
                # down to 0 via `int(remaining * _UNSPECIFIED_SHARE)`
                # the identical way `max_model_calls` already does), but
                # were never included in this upfront rejection.
                # Confirmed live: a parent with generous remaining
                # `max_model_calls`/`max_wall_seconds` headroom but a
                # nearly-exhausted token budget (an entirely ordinary,
                # foreseeable case -- a long-running conversation nearing
                # its `Budget.max_total_tokens`) let a subagent with a
                # granted `max_total_tokens == 0` slip straight past this
                # gate. A full subagent `AgentLoop` was then actually
                # constructed and run, making one real, costly
                # `provider.generate()` call before its own post-call
                # `spend.exceeded()` check caught it -- exactly the
                # "wasted real call" failure mode the `max_model_calls`/
                # `max_wall_seconds` half of this same gate already
                # exists to prevent, just for the other two `Budget`
                # dimensions. Fixed by checking all four, matching
                # `Spend.exceeded()`'s own complete four-dimension check.
                if (
                    sub_budget.max_model_calls <= 0
                    or sub_budget.max_wall_seconds <= 0
                    or sub_budget.max_total_tokens <= 0
                    or sub_budget.max_cost_usd <= 0
                ):
                    return AgentResult(
                        state=AgentState.BUDGET_EXCEEDED,
                        final_message=None,
                        spend=Spend(),
                        run_dir="",
                    )
                # Reserve the FULL granted slice against the parent's
                # spend right now, while still holding the lock -- this
                # is a correct (not overly conservative) upper bound,
                # not a guess: the subagent's own budget check enforces
                # that it can never itself exceed `sub_budget`, so
                # reserving exactly that much is the real worst case.
                spend.model_calls += sub_budget.max_model_calls
                spend.total_tokens += sub_budget.max_total_tokens
                spend.wall_seconds += sub_budget.max_wall_seconds
                spend.cost_usd += sub_budget.max_cost_usd

            sub_tools = [t for t in self._tools.values() if not isinstance(t, DelegateTool)]
            sub_run_id = uuid.uuid4().hex[:12]
            sub_run_root = run_dir / "subagents"
            sub_run_dir = sub_run_root / sub_run_id
            sub_loop = AgentLoop(
                router=self._router,
                providers=self._providers,
                tools=sub_tools,
                confirm=self._confirm,
                budget=sub_budget,
                task_class=task_class,
                workdir=self._workdir,
                run_root=str(sub_run_root),
                degraders=self._degraders,
            )
            sub_final: Message | None = None
            sub_state = AgentState.FAILED
            sub_spend = Spend()
            reconciled = False
            try:
                async for sub_event in sub_loop.run(
                    subtask, session_id=session_id, run_id=sub_run_id
                ):
                    if sub_event.type == "run_done":
                        sub_final = sub_event.final_message
                        sub_state = sub_event.state
                        sub_spend = sub_event.spend
                        # Reconcile the earlier reservation down to what
                        # the subagent ACTUALLY used (never more, since
                        # its own budget bounds it) -- without this the
                        # parent's reported spend would stay pinned at
                        # the conservative up-front grant forever,
                        # overstating real cost for every subagent that
                        # finished under its own slice.
                        async with spend_lock:
                            spend.model_calls += (
                                sub_event.spend.model_calls - sub_budget.max_model_calls
                            )
                            spend.total_tokens += (
                                sub_event.spend.total_tokens - sub_budget.max_total_tokens
                            )
                            spend.wall_seconds += (
                                sub_event.spend.wall_seconds - sub_budget.max_wall_seconds
                            )
                            spend.cost_usd += sub_event.spend.cost_usd - sub_budget.max_cost_usd
                        reconciled = True
            finally:
                # A real bug found by giving this reservation/reconcile
                # pair its own fresh-eyes sweep: the reconciliation above
                # only ever runs on a NORMAL run_done. `run_one`'s own
                # `_TOOL_TIMEOUT_SECONDS` wraps every tool call (including
                # whatever delegate_task-like tool called us) in
                # `asyncio.wait_for` -- if a subagent's own provider call
                # is still in flight when that fires, `CancelledError` is
                # thrown into this `async for` and `run_done` never
                # arrives, leaving the FULL up-front reservation stuck on
                # `spend` forever: confirmed live, a cancelled subagent
                # left the parent's reported `spend.model_calls` at 12
                # against only 3 real provider calls actually made,
                # permanently understating headroom for every later
                # sibling delegation in the same run (the exact
                # starvation shape the two rounds directly above this one
                # already fixed for the *admission* side, now closed on
                # the *release* side too). Released here rather than left
                # reconciled to the subagent's true partial usage,
                # because there IS no true partial usage available -- its
                # own `Spend` object lives inside its own now-abandoned
                # `run()` call and was never surfaced before cancellation
                # cut it off. Releasing the whole reservation can only
                # ever UNDER-count that one subagent's own now-unrecoverable
                # real cost, never leave a phantom amount stuck forever --
                # a one-time, honest undercount on a rare cancellation
                # path is strictly better than a permanent, silent
                # overcount that starves every later delegation.
                if not reconciled:
                    async with spend_lock:
                        spend.model_calls -= sub_budget.max_model_calls
                        spend.total_tokens -= sub_budget.max_total_tokens
                        spend.wall_seconds -= sub_budget.max_wall_seconds
                        spend.cost_usd -= sub_budget.max_cost_usd
            return AgentResult(
                state=sub_state, final_message=sub_final, spend=sub_spend, run_dir=str(sub_run_dir)
            )

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

            # Appended once here, unconditionally, before the budget check
            # below and regardless of stop_reason — `messages` (and
            # therefore `transcript_out`) must reflect every assistant turn
            # that actually happened, not just the ones on the tool-use
            # path or the ones that stayed under budget. A prior version
            # appended after the budget check, so a response whose own
            # usage is what tipped `spend` over `Budget` was silently
            # dropped from history entirely — the model was really called
            # and really billed for it (`spend.total_tokens` etc. already
            # reflect it, right above), but the turn that cost that spend
            # left no trace in `messages`/`transcript_out`. Confirmed live:
            # a run with `Budget(max_total_tokens=1)` produced a real,
            # billed END_TURN response that `transcript_out` never
            # contained at all. `final_message` deliberately still stays
            # unset on this path (see the matching budget re-check after
            # verification below) — only the append needed to move.
            messages.append(done.message)

            reason = spend.exceeded(self._budget)
            if reason:
                transition(AgentState.BUDGET_EXCEEDED)
                yield await emit(StateChangedEvent(state=state, detail=reason))
                break

            if done.stop_reason == StopReason.END_TURN:
                # Verifier subagent (opt-in via `verify=True`), scoped
                # deliberately narrow for a first real slice, matching this
                # project's pattern elsewhere: ADVISORY, not a hard gate.
                # A verifier that can't run at all (no budget left, its own
                # FAILED/INTERRUPTED terminal state) or gives an ambiguous
                # response (doesn't start with REJECTED) never blocks a
                # real completed answer -- only an unambiguous REJECTED
                # verdict does. This must run and decide BEFORE
                # transition(DONE) below, not after: DONE has no legal
                # outgoing transition in `LEGAL` (it's terminal), so
                # rejecting after already transitioning there would trip
                # `transition()`'s own legality assertion -- the decision
                # has to happen while `state` is still CALLING_MODEL, which
                # legally transitions to either DONE or FAILED.
                reject_detail: str | None = None
                if self._verify:
                    verify_prompt = (
                        "You are verifying whether an AI assistant's answer "
                        "actually satisfies the user's original task. Respond "
                        "with exactly one word first -- VERIFIED or REJECTED "
                        "-- followed by a one-sentence reason.\n\n"
                        f"Original task: {task}\n\n"
                        f"Candidate answer: {done.message.text()}"
                    )
                    verify_result = await spawn_subagent(
                        verify_prompt, task_class=TaskClass.SUBTASK
                    )
                    verify_text = (
                        verify_result.final_message.text()
                        if verify_result.final_message is not None
                        else ""
                    )
                    if (
                        verify_result.state == AgentState.DONE
                        and verify_text.strip().upper().startswith("REJECTED")
                    ):
                        reject_detail = f"verifier rejected the final answer: {verify_text}"

                # A real bug found by actually running verify=True against a
                # tight Budget: spawn_subagent() (above) merges the
                # verifier's own real Spend into this run's live `spend`
                # object, but the ONLY spend.exceeded() check in this
                # function runs once, at line ~502 -- BEFORE this verify
                # block ever executes. Confirmed live: an identical Budget
                # that correctly reported DONE with verify=False reported
                # DONE again with verify=True even though the verifier's
                # own cost pushed the merged spend genuinely over budget
                # (total_tokens=478 against max_total_tokens=40). Every real
                # caller (cli.py, server/app.py) gates both "was this run a
                # failure" and "should the session be saved" on state ==
                # DONE alone, so an over-budget verify=True run was
                # reported as ordinary success everywhere -- silently
                # defeating the one purpose Budget/spend.exceeded exists to
                # serve. Re-checked here, after the only other thing in
                # this branch that can change `spend`, rather than folding
                # into the earlier check (which runs before verification
                # even starts and so can't see its cost yet). Given
                # priority over a REJECTED verdict, matching the budget
                # check's own priority everywhere else in this loop.
                budget_reason = spend.exceeded(self._budget)
                if budget_reason:
                    transition(AgentState.BUDGET_EXCEEDED)
                    yield await emit(StateChangedEvent(state=state, detail=budget_reason))
                elif reject_detail is not None:
                    transition(AgentState.FAILED)
                    yield await emit(StateChangedEvent(state=state, detail=reject_detail))
                else:
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
