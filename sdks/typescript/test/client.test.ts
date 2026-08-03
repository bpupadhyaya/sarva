import { describe, expect, it, vi } from "vitest";
import { SarvaApiError, SarvaChatStream, SarvaClient } from "../src/client.js";
import { textFromContent } from "../src/types.js";

function fakeFetch(status: number, body: unknown): typeof fetch {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  }) as unknown as typeof fetch;
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
  onclose: (() => void) | null = null;

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
