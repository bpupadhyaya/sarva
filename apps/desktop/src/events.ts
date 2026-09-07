/**
 * Typed mirror of sarva.agent.events.AgentEvent (core/sarva/agent/events.py).
 *
 * Kept minimal and local to this app rather than importing `sdks/typescript/`
 * directly — the SDK package now exists (it didn't when this comment
 * originally said "once a second consumer needs it"), but this app isn't
 * wired to depend on it yet; a real, still-open follow-up, not silently
 * assumed done. Field names match the Python Pydantic models' JSON output
 * exactly (snake_case), since that's the wire format over /ws/chat.
 *
 * A real drift bug found by a fresh-eyes sweep while investigating that
 * stale comment: this local copy's `run_done` variant was missing the
 * `spend` field entirely, even though `sarva.agent.events.RunDoneEvent`
 * (the real Pydantic model actually serialized over the wire) has
 * required it since `Spend` tracking shipped -- confirmed by reading
 * `RunDoneEvent` directly, not assumed from this file's own claim to be
 * an accurate mirror. `sdks/typescript/src/types.ts`'s own `AgentEvent`
 * already had it correctly; only this app's independent, never-reconciled
 * copy had silently drifted.
 *
 * A second instance of the identical drift class, found by comparing
 * this file against that same sibling SDK mirror directly rather than
 * just its own claim to be accurate: `ToolCall` here was a minimal stub
 * (`id`/`name`/`arguments`, no `type`) and `tool_finished`'s own
 * `result` field was `{ is_error: boolean }` only -- but the real
 * `sarva.multimodal.content.ToolCallBlock`/`ToolResultBlock` Pydantic
 * models (what actually crosses the wire) also carry `type`, and
 * `ToolResultBlock` additionally carries `tool_call_id` and `content`
 * (the tool's actual output) -- fields `sdks/typescript/src/types.ts`'s
 * own `ToolCallBlock`/`ToolResultBlock` already modeled completely.
 * `App.tsx` only ever reads `.name`/`.arguments`/`.is_error` today, so
 * this wasn't visibly broken -- but a future feature reading `event.
 * result.content` (e.g. actually showing what a tool returned, not just
 * ok/error) would have hit a field this "typed mirror" claimed didn't
 * exist, even though the server has always sent it.
 */
export interface Spend {
  model_calls: number;
  total_tokens: number;
  wall_seconds: number;
  cost_usd: number;
}

export type AgentState =
  | "init"
  | "calling_model"
  | "running_tools"
  | "awaiting_confirmation"
  | "done"
  | "failed"
  | "interrupted"
  | "budget_exceeded";

interface ToolCall {
  type: "tool_call";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

/** Any single content block in a tool result (usually TextBlock, but
 * image/audio blocks are also possible) -- deliberately not fully
 * modeled here, matching `sdks/typescript/src/types.ts`'s own honest
 * "partial, not guessed" scoping for the same union. */
type ContentBlock = { type: string; [key: string]: unknown };

interface ToolResult {
  type: "tool_result";
  tool_call_id: string;
  content: ContentBlock[];
  is_error: boolean;
}

interface ProviderEvent {
  type: string;
  text?: string;
}

export type AgentEvent =
  | { type: "state_changed"; state: AgentState; detail?: string | null }
  | { type: "model_stream"; event: ProviderEvent }
  | { type: "tool_started"; call: ToolCall }
  | { type: "tool_finished"; result: ToolResult; seconds: number }
  | { type: "needs_confirmation"; call: ToolCall }
  | { type: "run_done"; state: AgentState; final_message: unknown; spend: Spend };
