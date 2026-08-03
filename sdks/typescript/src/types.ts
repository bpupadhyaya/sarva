/**
 * Wire types for the Sarva REST + WebSocket API.
 *
 * Field names match the Python Pydantic models' JSON output exactly
 * (snake_case) -- this is a genuinely thin client, not a translation
 * layer, so every shape here is verified against the real source it
 * mirrors rather than guessed:
 *   - REST request/response bodies: core/sarva/server/schemas.py
 *   - WebSocket event frames: core/sarva/agent/events.py
 *   - Provider stream events: core/sarva/providers/base.py
 *   - Content blocks: core/sarva/multimodal/content.py
 *
 * `ContentBlock` is deliberately NOT fully modeled (image/audio blocks
 * carry a Python `bytes | None` field whose exact JSON encoding isn't
 * pinned down here) -- kept as `unknown` rather than guessed, the same
 * "honestly scoped, not silently assumed" discipline the rest of this
 * project applies. `textFromContent()` below covers the common case
 * (extracting the text of a message) without needing the full union.
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

export interface Spend {
  model_calls: number;
  total_tokens: number;
  wall_seconds: number;
  cost_usd: number;
}

// ---------- REST: /chat ----------

export interface ChatRequest {
  message: string;
  session?: string | null;
  image_base64?: string | null;
  image_media_type?: string | null;
  model?: string | null;
  verify?: boolean;
}

export interface ChatResponse {
  state: string;
  message: string | null;
  spend: Spend;
  detail: string | null;
}

// ---------- REST: /models, /doctor, /config ----------

export interface ModelInfo {
  id: string;
  display_name: string;
  available: boolean;
}

export interface DoctorCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface SaveConfigRequest {
  anthropic_api_key?: string | null;
  openai_api_key?: string | null;
  gemini_api_key?: string | null;
}

// ---------- Content blocks (partial -- see module docstring) ----------

export interface TextBlock {
  type: "text";
  text: string;
}

/** Any other content block type (image, audio, thinking, tool_call,
 * tool_result, ...) -- present in the JSON but not individually typed. */
export type ContentBlock = TextBlock | { type: string; [key: string]: unknown };

export interface MessageOut {
  role: "system" | "user" | "assistant";
  content: ContentBlock[];
}

/** Mirrors `sarva.multimodal.content.Message.text()` exactly: the
 * concatenated text of every TextBlock in a message, ignoring every
 * other block type. */
export function textFromContent(content: ContentBlock[]): string {
  return content
    .filter((b): b is TextBlock => b.type === "text" && typeof (b as TextBlock).text === "string")
    .map((b) => b.text)
    .join("");
}

// ---------- WebSocket: /ws/chat ----------

export interface WsChatRequest {
  message: string;
  session?: string | null;
  image_base64?: string | null;
  image_media_type?: string | null;
  model?: string | null;
  auto?: boolean;
  verify?: boolean;
}

export interface ToolCallBlock {
  type: "tool_call";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultBlock {
  type: "tool_result";
  tool_call_id: string;
  content: ContentBlock[];
  is_error: boolean;
}

export interface TextDeltaEvent {
  type: "text_delta";
  text: string;
}

export interface ThinkingDeltaEvent {
  type: "thinking_delta";
  text: string;
}

export interface ToolCallProviderEvent {
  type: "tool_call";
  call: ToolCallBlock;
}

export interface DoneProviderEvent {
  type: "done";
  stop_reason: string;
  message: MessageOut;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read_tokens: number;
    cost_usd: number;
  };
}

export interface StreamErrorProviderEvent {
  type: "stream_error";
  code: string;
  detail: string;
  retryable: boolean;
}

/** Any provider event type not individually modeled above (e.g. the
 * server-side "content" event for non-text output blocks). */
export type ProviderEvent =
  | TextDeltaEvent
  | ThinkingDeltaEvent
  | ToolCallProviderEvent
  | DoneProviderEvent
  | StreamErrorProviderEvent
  | { type: string; [key: string]: unknown };

export type AgentEvent =
  | { type: "state_changed"; state: AgentState; detail?: string | null }
  | { type: "model_stream"; event: ProviderEvent }
  | { type: "tool_started"; call: ToolCallBlock }
  | { type: "tool_finished"; result: ToolResultBlock; seconds: number }
  | { type: "needs_confirmation"; call: ToolCallBlock }
  | { type: "run_done"; state: AgentState; final_message: MessageOut | null; spend: Spend };
