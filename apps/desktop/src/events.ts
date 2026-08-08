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
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

interface ProviderEvent {
  type: string;
  text?: string;
}

export type AgentEvent =
  | { type: "state_changed"; state: AgentState; detail?: string | null }
  | { type: "model_stream"; event: ProviderEvent }
  | { type: "tool_started"; call: ToolCall }
  | { type: "tool_finished"; result: { is_error: boolean }; seconds: number }
  | { type: "needs_confirmation"; call: ToolCall }
  | { type: "run_done"; state: AgentState; final_message: unknown; spend: Spend };
