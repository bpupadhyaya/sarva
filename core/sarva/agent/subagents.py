"""sarva.agent.subagents — subagent fan-out: a `delegate_task` tool that
lets the main agent loop spawn a fresh, independent sub-`AgentLoop` for a
self-contained subtask and get back its final answer as a tool result.

Named in the design doc's own architecture section since this project's
very first agent-loop chapter (`docs/agent-loop.md`'s "What's honestly
not built yet") and left deliberately unbuilt until now, tracked
directly rather than silently forgotten.

The actual spawning logic lives in `AgentLoop.run()` itself
(`core/sarva/agent/loop.py`), not here — that's the one place with a
router, providers, budget, and live spend to build a subagent from.
This module only defines the `Tool` a model calls to trigger it, kept
deliberately free of any import of `AgentLoop`/`loop.py` to avoid a
circular import (`loop.py` already imports this module, to exclude
`DelegateTool` from a subagent's own tool list; `tools.py` imports this
module too, to register `DelegateTool` in `BUILTIN_TOOLS`).

Scoped deliberately narrow for a first real slice, not the full
"planner"/"verifier subagent" pattern the design doc also names (a
separate, bigger design decision, still left honestly unbuilt -- see
`docs/agent-loop.md`):

- **One level of fan-out only.** A subagent's own tool list never
  includes `DelegateTool` itself (`AgentLoop.run()`'s own spawn closure
  filters it out), so a delegated task cannot itself delegate further --
  bounds recursion depth at 1 rather than risking a model that decides
  to keep delegating indefinitely.
- **Subagent spend counts against the PARENT's own remaining budget,
  not a fresh independent one.** `AgentLoop.run()` builds the subagent
  with `Budget(max_model_calls=<parent's remaining>, ...)` computed from
  the parent's own `Budget` minus its `Spend` so far, and adds the
  subagent's own final `Spend` back into the parent's live `Spend`
  object once it completes -- so the very next `spend.exceeded(...)`
  check in the parent loop reflects the subagent's real cost too.
  Without this, a model could spawn subagents to bypass its own budget
  entirely; confirmed live before fixing it this way (see
  BUILD-JOURNAL.md).
- **Routed via `TaskClass.SUBTASK`** -- cheap delegated work, distinct
  from the parent's own `TaskClass.MAIN`, matching that enum entry's own
  documented purpose (confirmed genuinely unused anywhere in this
  codebase before this).
- **Confirmation-gated tool calls made BY the subagent still go through
  the exact same `ConfirmPolicy` the parent loop was constructed with**
  -- delegation doesn't bypass destructive-tool confirmation.
- **The subagent gets the same tool list as the parent** (minus
  `DelegateTool` itself), not a separate curated "safe" subset -- the
  same confirm-gating discipline already governs destructive tools
  either way, so there's no meaningful safety reason to give it fewer
  tools, only the bounded-recursion reason to give it one fewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sarva.multimodal.content import TextBlock, ToolResultBlock
from sarva.providers.base import ToolSpec

if TYPE_CHECKING:
    from sarva.agent.tools import ToolContext


class DelegateTool:
    spec = ToolSpec(
        name="delegate_task",
        description=(
            "Delegate a self-contained subtask to a fresh, independent subagent "
            "and get back its final answer as plain text. Use this for a "
            "well-defined piece of work that doesn't need this conversation's "
            "own history -- e.g. 'summarize this file', 'find every usage of X "
            "in this directory', 'draft a commit message for these changes'. "
            "Write the task so a fresh agent with no other context could "
            "complete it on its own. The subagent has its own tools and its "
            "own model-call budget (bounded by what's left of this run's own "
            "budget) but cannot delegate further itself."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The self-contained subtask to delegate, written so a "
                        "fresh agent with no other context could complete it."
                    ),
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        destructive=False,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResultBlock:
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text="delegate_task requires a non-empty 'task' string")],
                is_error=True,
            )
        if ctx.spawn_subagent is None:
            return ToolResultBlock(
                tool_call_id="",
                content=[TextBlock(text="subagent delegation is not available in this context")],
                is_error=True,
            )
        final_message = await ctx.spawn_subagent(task)
        if final_message is None:
            return ToolResultBlock(
                tool_call_id="",
                content=[
                    TextBlock(
                        text="the subagent did not complete successfully "
                        "(ran out of budget, failed, or was interrupted)"
                    )
                ],
                is_error=True,
            )
        return ToolResultBlock(
            tool_call_id="", content=[TextBlock(text=final_message.text())], is_error=False
        )
