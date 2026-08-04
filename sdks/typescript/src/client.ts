import type {
  AgentEvent,
  ChatRequest,
  ChatResponse,
  DoctorCheck,
  ModelInfo,
  SaveConfigRequest,
  WsChatRequest,
} from "./types.js";

export interface SarvaClientOptions {
  /** Default: "http://localhost:8000" (`sarva serve`'s own default bind). */
  baseUrl?: string;
  /** Override for environments with no global `fetch` (older Node). */
  fetchImpl?: typeof fetch;
  /** Override for environments with no global `WebSocket` (Node < 22). */
  webSocketImpl?: typeof WebSocket;
}

export class SarvaApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: unknown,
  ) {
    super(`Sarva API request failed with status ${status}: ${JSON.stringify(body)}`);
    this.name = "SarvaApiError";
  }
}

export interface ChatStreamHandlers {
  onEvent?: (event: AgentEvent) => void;
  /** A malformed frame or a real transport-level error -- the stream
   * cannot usefully continue past this. */
  onError?: (error: Error) => void;
  onClose?: () => void;
}

/**
 * One `/ws/chat` connection: single turn per connection, exactly matching
 * the server's own documented contract (see `ws_chat`'s docstring in
 * core/sarva/server/app.py). Send the initial request once at
 * construction (handled internally, on open); if a `needs_confirmation`
 * event arrives and `auto` was not set, the *next* thing sent on this
 * socket must be a confirmation reply -- call `respondToConfirmation()`.
 */
export class SarvaChatStream {
  private ws: WebSocket;
  private sawRunDone = false;

  constructor(
    url: string,
    WebSocketImpl: typeof WebSocket,
    request: WsChatRequest,
    handlers: ChatStreamHandlers,
  ) {
    this.ws = new WebSocketImpl(url);
    this.ws.onopen = () => {
      this.ws.send(JSON.stringify(request));
    };
    this.ws.onmessage = (ev: MessageEvent) => {
      let parsed: AgentEvent;
      try {
        const raw = typeof ev.data === "string" ? ev.data : String(ev.data);
        parsed = JSON.parse(raw) as AgentEvent;
      } catch (e) {
        handlers.onError?.(
          new Error(`received a malformed (non-JSON) frame: ${e instanceof Error ? e.message : e}`),
        );
        return;
      }
      if (parsed.type === "run_done") {
        this.sawRunDone = true;
      }
      handlers.onEvent?.(parsed);
    };
    this.ws.onerror = () => {
      handlers.onError?.(new Error("WebSocket transport error"));
    };
    this.ws.onclose = (ev?: CloseEvent) => {
      // A real bug found by actually simulating a server-crash-mid-turn
      // (a state_changed frame arrives, then the connection drops
      // uncleanly with no run_done ever sent): the SDK's own documented
      // "resolve on run_done" consumer pattern hung forever with zero
      // signal that anything had gone wrong, since onclose carried no
      // information and nothing rejected the in-flight turn. Any
      // dropped connection, server restart, or reverse-proxy idle
      // timeout mid-turn would hang a real caller indefinitely -- this
      // is the one, narrow signal actually missing (not a speculative
      // timeout feature): if the socket closes and a terminal run_done
      // was never seen, that's unambiguously an error, surfaced via
      // onError before onClose fires. `ev` is read defensively (every
      // real WebSocket implementation passes a real CloseEvent, but
      // this must not itself become a new way to crash if some
      // WebSocketImpl doesn't).
      if (!this.sawRunDone) {
        handlers.onError?.(
          new Error(
            `WebSocket closed before a run_done event arrived (code=${ev?.code ?? "unknown"}, ` +
              `reason=${ev?.reason || "none"}) -- the turn did not complete`,
          ),
        );
      }
      handlers.onClose?.();
    };
  }

  /** Reply to a `needs_confirmation` event. Do NOT call this if the
   * request set `auto: true` -- per the server's own contract, nothing
   * is waiting to consume a reply in that mode, and sending one risks
   * being read as the answer to a later, unrelated prompt. */
  respondToConfirmation(approved: boolean): void {
    this.ws.send(JSON.stringify({ approved }));
  }

  close(): void {
    this.ws.close();
  }
}

/** A thin client for the Sarva REST + WebSocket API. Every method maps
 * directly to one real endpoint in core/sarva/server/app.py -- no
 * client-side retry/caching/state beyond what's documented here. */
export class SarvaClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private readonly webSocketImpl?: typeof WebSocket;

  constructor(options: SarvaClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://localhost:8000").replace(/\/+$/, "");
    const fetchImpl = options.fetchImpl ?? globalThis.fetch;
    if (!fetchImpl) {
      throw new Error(
        "no fetch implementation available -- pass options.fetchImpl in an environment with no global fetch",
      );
    }
    this.fetchImpl = fetchImpl;
    // Node's own built-in global WebSocket (undici) does NOT default here
    // the way fetch does -- confirmed live against a real `sarva serve`
    // process: Node's native WebSocket silently fails the handshake
    // against uvicorn's `websockets`-based ASGI implementation (opens,
    // sends the initial frame, then the connection drops with close code
    // 1006 before any server response ever arrives), while the same
    // request through the mature `ws` npm package, and through Python's
    // own `websockets` client, both work correctly -- this is a real,
    // narrow interop gap in Node's own implementation, not a bug in this
    // SDK or in Sarva's server. Rather than silently handing back a
    // WebSocket that will hang forever on every real request, Node
    // callers must explicitly opt in via `webSocketImpl` (`ws` is the
    // proven-working choice) -- "reject, don't guess" for a known-broken
    // default, the same discipline this project applies to malformed
    // CLI input elsewhere. Browsers are unaffected (a different,
    // long-established implementation) and keep defaulting to the global.
    const isNode =
      typeof process !== "undefined" && typeof process.versions?.node === "string";
    if (options.webSocketImpl) {
      this.webSocketImpl = options.webSocketImpl;
    } else if (!isNode) {
      this.webSocketImpl = (globalThis as { WebSocket?: typeof WebSocket }).WebSocket;
    }
  }

  private async requestJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    // A real bug found by actually running the compiled SDK against a
    // real server response: `response.json()` used to run unconditionally,
    // before `response.ok` was ever checked. The server's own global
    // exception handler only covers ConfigError (sarva.server.app) --
    // any OTHER unhandled exception (e.g. a PermissionError from
    // save_config() when ~/.sarva isn't writable, a realistic real-world
    // condition, not contrived) falls through to Starlette's default
    // handler, which returns a PLAIN-TEXT 500 body, not JSON. Calling
    // `.json()` on that raised a raw SyntaxError instead of the
    // documented SarvaApiError every caller is meant to catch --
    // discarding the real HTTP status and message entirely. Fixed by
    // reading the body as text exactly once, then trying to parse it as
    // JSON and falling back to the raw text on failure -- an error
    // response with a JSON body (the common case) still gets its parsed
    // object as `SarvaApiError.body`; a non-JSON body (this bug's case)
    // gets the raw string instead of an opaque parse error.
    const raw = await response.text();
    let body: unknown;
    try {
      body = raw ? JSON.parse(raw) : undefined;
    } catch {
      body = raw;
    }
    if (!response.ok) {
      throw new SarvaApiError(response.status, body);
    }
    return body as T;
  }

  async health(): Promise<{ status: string }> {
    return this.requestJson("/health");
  }

  async models(): Promise<ModelInfo[]> {
    return this.requestJson("/models");
  }

  async doctor(): Promise<DoctorCheck[]> {
    return this.requestJson("/doctor");
  }

  async saveConfig(req: SaveConfigRequest): Promise<DoctorCheck[]> {
    return this.requestJson("/config", { method: "POST", body: JSON.stringify(req) });
  }

  /** Single-turn, non-streaming, tool-free -- mirrors `sarva chat`
   * exactly (see ChatRequest/ChatResponse's own docstrings). For tool
   * use and streaming, use `chatStream()`. */
  async chat(req: ChatRequest): Promise<ChatResponse> {
    return this.requestJson("/chat", { method: "POST", body: JSON.stringify(req) });
  }

  /** Opens one `/ws/chat` connection for a single tool-using,
   * streaming turn -- mirrors `sarva run` (see `SarvaChatStream`'s own
   * docstring for the confirmation-reply protocol). */
  chatStream(request: WsChatRequest, handlers: ChatStreamHandlers = {}): SarvaChatStream {
    if (!this.webSocketImpl) {
      const isNode = typeof process !== "undefined" && typeof process.versions?.node === "string";
      throw new Error(
        isNode
          ? "no options.webSocketImpl was provided. Node's own built-in global " +
            "WebSocket does not interoperate correctly with Sarva's server " +
            "(confirmed live: the connection silently drops before any response " +
            "arrives) -- install the `ws` package and pass it explicitly: " +
            "new SarvaClient({ webSocketImpl: (await import('ws')).default })"
          : "no WebSocket implementation available -- pass options.webSocketImpl " +
            "in an environment with no global WebSocket",
      );
    }
    const wsUrl = `${this.baseUrl.replace(/^http/, "ws")}/ws/chat`;
    return new SarvaChatStream(wsUrl, this.webSocketImpl, request, handlers);
  }
}
