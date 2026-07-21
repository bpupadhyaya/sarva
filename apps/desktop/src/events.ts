/**
 * Typed mirror of sarva.agent.events.AgentEvent (core/sarva/agent/events.py).
 *
 * Kept minimal and local to this app for now — a proper `sdks/typescript/`
 * package (per the design doc) can factor this out once a second consumer
 * needs it. Field names match the Python Pydantic models' JSON output
 * exactly (snake_case), since that's the wire format over /ws/chat.
 */

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
  | { type: "run_done"; state: AgentState; final_message: unknown };
