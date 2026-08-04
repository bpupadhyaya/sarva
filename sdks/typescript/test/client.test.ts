import { describe, expect, it, vi } from "vitest";
import { SarvaApiError, SarvaChatStream, SarvaClient } from "../src/client.js";
import { textFromContent } from "../src/types.js";

function fakeFetch(status: number, body: unknown): typeof fetch {
  // A real Response (not a partial { json: ... } stand-in): requestJson()
  // now reads the body via .text() first (see its own comment on the
  // real bug this closes -- calling .json() unconditionally, before
  // checking response.ok, threw a raw SyntaxError on any non-JSON error
  // body instead of the documented SarvaApiError). A hand-built object
  // exposing only .json() would silently mask that exact regression
  // returning here, the same way it did before this fix existed.
  //
  // mockImplementation, not mockResolvedValue: a real Response's body
  // can only be read once -- a shared mockResolvedValue instance reused
  // across multiple calls to the same mock (some tests below call the
  // client twice against one fakeFetch) would throw "Body is unusable:
  // Body has already been read" on the second call. A fresh Response
  // per invocation matches how a real server actually behaves.
  return vi
    .fn()
    .mockImplementation(async () => new Response(JSON.stringify(body), { status })) as unknown as typeof fetch;
}

describe("SarvaClient REST methods", () => {
  it("chat() posts the request and returns the parsed response", async () => {
    const responseBody = { state: "done", message: "hi there", spend: { model_calls: 1, total_tokens: 10, wall_seconds: 0.5, cost_usd: 0 }, detail: null };
    const fetchImpl = fakeFetch(200, responseBody);
    const client = new SarvaClient({ baseUrl: "http://example.com", fetchImpl });

    const result = await client.chat({ message: "hello" });

    expect(result).toEqual(responseBody);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://example.com/chat",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ message: "hello" }),
      }),
    );
  });

  it("chat() rejects with SarvaApiError on a non-2xx response, carrying the real status and body", async () => {
    const errorBody = { detail: "corrupted config" };
    const fetchImpl = fakeFetch(500, errorBody);
    const client = new SarvaClient({ baseUrl: "http://example.com", fetchImpl });

    await expect(client.chat({ message: "hello" })).rejects.toMatchObject({
      status: 500,
      body: errorBody,
    });
    await expect(client.chat({ message: "hello" })).rejects.toBeInstanceOf(SarvaApiError);
  });

  it("chat() rejects with SarvaApiError, not a raw SyntaxError, on a non-JSON error body", async () => {
    // A real bug found by actually running the compiled SDK against a
    // real server response: requestJson() used to call response.json()
    // unconditionally, before response.ok was ever checked. The
    // server's own global exception handler only covers ConfigError --
    // any OTHER unhandled exception (e.g. a real PermissionError from
    // save_config() when ~/.sarva isn't writable) falls through to
    // Starlette's default handler, which returns a PLAIN-TEXT 500 body,
    // not JSON. Calling .json() on that raised a raw SyntaxError
    // instead of the documented SarvaApiError every caller is meant to
    // catch, discarding the real status/message entirely. Confirmed
    // live against the real compiled dist/ before this fix.
    const fetchImpl = vi
      .fn()
      .mockImplementation(
        async () => new Response("Internal Server Error", { status: 500 }),
      ) as unknown as typeof fetch;
    const client = new SarvaClient({ baseUrl: "http://example.com", fetchImpl });

    const error = await client.chat({ message: "hello" }).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(SarvaApiError);
    expect((error as SarvaApiError).status).toBe(500);
    expect((error as SarvaApiError).body).toBe("Internal Server Error");
  });

  it("models() hits GET /models and returns the parsed list", async () => {
    const modelsBody = [{ id: "mock", display_name: "Mock", available: true }];
    const fetchImpl = fakeFetch(200, modelsBody);
    const client = new SarvaClient({ baseUrl: "http://example.com", fetchImpl });

    const result = await client.models();

    expect(result).toEqual(modelsBody);
    expect(fetchImpl).toHaveBeenCalledWith("http://example.com/models", expect.anything());
  });

  it("strips a trailing slash from baseUrl so paths never double up", async () => {
    const fetchImpl = fakeFetch(200, { status: "ok" });
    const client = new SarvaClient({ baseUrl: "http://example.com/", fetchImpl });

    await client.health();

    expect(fetchImpl).toHaveBeenCalledWith("http://example.com/health", expect.anything());
  });

  it("throws immediately if no fetch implementation is available anywhere", () => {
    const originalFetch = globalThis.fetch;
    // @ts-expect-error -- deliberately simulating an environment with no global fetch
    delete globalThis.fetch;
    try {
      expect(() => new SarvaClient({ baseUrl: "http://example.com" })).toThrow(/no fetch implementation/);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

/** A minimal, controllable stand-in for the browser WebSocket API --
 * the same testing shape the desktop app's own App.test.tsx already
 * established for this exact protocol (real delivery verified live
 * against a running `sarva serve` separately, per BUILD-JOURNAL.md). */
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  sent: string[] = [];
  closed = false;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: ((ev?: { code: number; reason: string }) => void) | null = null;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.closed = true;
    this.onclose?.();
  }

  /** Simulates the transport itself dropping the connection (server
   * crash, network blip, proxy timeout) -- as opposed to `close()`,
   * which mirrors the client calling the real WebSocket.close() (no
   * event argument, matching that API's own signature). */
  dropUncleanly(code: number, reason = "") {
    this.closed = true;
    this.onclose?.({ code, reason });
  }

  open() {
    this.onopen?.();
  }

  emit(event: unknown) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
}

describe("SarvaClient.chatStream", () => {
  it("connects to the ws:// counterpart of baseUrl and sends the request on open", () => {
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });

    client.chatStream({ message: "hi" });

    const ws = MockWebSocket.instances.at(-1)!;
    expect(ws.url).toBe("ws://example.com/ws/chat");
    ws.open();
    expect(ws.sent).toEqual([JSON.stringify({ message: "hi" })]);
  });

  it("parses incoming frames and dispatches them to onEvent", () => {
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    const events: unknown[] = [];

    client.chatStream({ message: "hi" }, { onEvent: (e) => events.push(e) });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.open();
    ws.emit({ type: "state_changed", state: "calling_model" });
    ws.emit({ type: "run_done", state: "done", final_message: null, spend: {} });

    expect(events).toEqual([
      { type: "state_changed", state: "calling_model" },
      { type: "run_done", state: "done", final_message: null, spend: {} },
    ]);
  });

  it("reports a malformed (non-JSON) frame via onError instead of throwing", () => {
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    const errors: Error[] = [];

    client.chatStream({ message: "hi" }, { onError: (e) => errors.push(e) });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.open();
    ws.onmessage?.({ data: "not valid json {{{" });

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toMatch(/malformed/);
  });

  it("respondToConfirmation() sends exactly {approved: bool} as the next frame", () => {
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });

    const stream = client.chatStream({ message: "hi" });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.open();
    stream.respondToConfirmation(true);

    expect(ws.sent[ws.sent.length - 1]).toBe(JSON.stringify({ approved: true }));
  });

  it("close() closes the underlying WebSocket", () => {
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });

    const stream = client.chatStream({ message: "hi" });
    stream.close();

    expect(MockWebSocket.instances.at(-1)!.closed).toBe(true);
  });

  it("reports an error via onError if the connection drops before run_done arrives", () => {
    // A real bug found by actually simulating a server-crash-mid-turn:
    // the SDK's own documented "resolve on run_done" consumer pattern
    // hung forever with zero signal that anything had gone wrong, since
    // onclose carried no information and nothing rejected the in-flight
    // turn. Any dropped connection, server restart, or reverse-proxy
    // idle timeout mid-turn would hang a real caller indefinitely.
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    const errors: Error[] = [];
    const closes: number[] = [];

    client.chatStream(
      { message: "hi" },
      { onError: (e) => errors.push(e), onClose: () => closes.push(1) },
    );
    const ws = MockWebSocket.instances.at(-1)!;
    ws.open();
    ws.emit({ type: "state_changed", state: "calling_model" });
    ws.dropUncleanly(1006, "");

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toMatch(/closed before a run_done event arrived/);
    expect(errors[0].message).toMatch(/code=1006/);
    expect(closes).toHaveLength(1); // onClose still fires too, in addition to onError
  });

  it("does NOT report an error when the connection closes cleanly after run_done", () => {
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    const errors: Error[] = [];

    const stream = client.chatStream({ message: "hi" }, { onError: (e) => errors.push(e) });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.open();
    ws.emit({ type: "run_done", state: "done", final_message: null, spend: {} });
    stream.close();

    expect(errors).toHaveLength(0);
  });

  it("handles a WebSocketImpl that calls onclose with no event argument at all", () => {
    // Defensive: every real WebSocket implementation passes a real
    // CloseEvent, but reading it must not itself become a new way to
    // crash if some WebSocketImpl doesn't provide one.
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    const errors: Error[] = [];

    client.chatStream({ message: "hi" }, { onError: (e) => errors.push(e) });
    const ws = MockWebSocket.instances.at(-1)!;
    ws.open();
    expect(() => ws.close()).not.toThrow();

    expect(errors).toHaveLength(1);
    expect(errors[0].message).toMatch(/code=unknown/);
  });

  it("throws a Node-specific, actionable error when running under Node with no explicit webSocketImpl", () => {
    // Real, confirmed behavior, not a hypothetical: Node's own built-in
    // global WebSocket does NOT interoperate with a real `sarva serve`
    // process (confirmed live -- the connection silently drops before
    // any response arrives, while the `ws` npm package and Python's own
    // websockets client both work correctly against the identical
    // server). This test runs under Node, so the client must refuse to
    // silently hand back a WebSocket known not to work, rather than
    // defaulting to the global the way it does in a browser.
    const client = new SarvaClient({ baseUrl: "http://example.com", fetchImpl: fakeFetch(200, {}) });
    expect(() => client.chatStream({ message: "hi" })).toThrow(/ws.*package|no options\.webSocketImpl/i);
  });

  it("does not default to the global WebSocket under Node even if one exists", () => {
    // The real regression this guards against: Node 22+ ships a real
    // global WebSocket, so a naive "use the global if present" check
    // would silently pick the known-broken implementation instead of
    // erroring -- confirmed this doesn't happen by leaving the global
    // WebSocket in place (not deleted) and still expecting the throw.
    expect(globalThis.WebSocket).toBeDefined(); // sanity: the global genuinely exists here
    const client = new SarvaClient({ baseUrl: "http://example.com", fetchImpl: fakeFetch(200, {}) });
    expect(() => client.chatStream({ message: "hi" })).toThrow();
  });

  it("still uses an explicitly-provided webSocketImpl under Node, global or not", () => {
    MockWebSocket.instances = [];
    const client = new SarvaClient({
      baseUrl: "http://example.com",
      fetchImpl: fakeFetch(200, {}),
      webSocketImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    expect(() => client.chatStream({ message: "hi" })).not.toThrow();
  });
});

describe("textFromContent", () => {
  it("concatenates every TextBlock's text, ignoring other block types", () => {
    const content = [
      { type: "text", text: "hello " },
      { type: "image", media_type: "image/png" },
      { type: "text", text: "world" },
    ];
    expect(textFromContent(content)).toBe("hello world");
  });

  it("returns an empty string for content with no text blocks", () => {
    expect(textFromContent([{ type: "tool_call", id: "x", name: "y", arguments: {} }])).toBe("");
  });
});

// Referenced only for type-checking the export surface exists; not asserted on directly.
void SarvaChatStream;
