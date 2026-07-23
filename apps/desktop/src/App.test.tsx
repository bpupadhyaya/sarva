import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

/**
 * A minimal, controllable stand-in for the browser WebSocket API. Real
 * WebSocket delivery is proven end-to-end elsewhere (BUILD-JOURNAL.md —
 * a real `sarva serve` process was hit with a real `websockets` client);
 * this mock exists to drive App.tsx's own event-handling logic
 * deterministically, one simulated frame at a time.
 */
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.closed = true;
  }
}

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket);
  // App.tsx's mount effect calls GET /doctor to decide whether to show
  // the first-run Onboarding screen (see Onboarding.test.tsx for that
  // screen's own coverage) -- these chat-flow tests aren't testing that
  // decision, so every test here gets an "already configured" response
  // by default, skipping straight to the chat UI they actually exercise.
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [{ name: "Anthropic API key", ok: true, detail: "ANTHROPIC_API_KEY is set" }],
    }),
  );
});

function latestSocket(): MockWebSocket {
  const ws = MockWebSocket.instances.at(-1);
  if (!ws) throw new Error("no WebSocket was constructed");
  return ws;
}

/** Renders <App /> and waits for its mount-time GET /doctor check to
 * settle before returning, so callers land on the real chat UI instead
 * of the brief `showOnboarding === null` (renders nothing) state. */
async function renderApp() {
  render(<App />);
  await screen.findByPlaceholderText("Message Sarva…");
}

/** Simulate the server sending one AgentEvent JSON frame. Wrapped in act()
 * because App.tsx's onmessage handler updates React state, and — unlike
 * fireEvent, which wraps DOM events automatically — a call reached through
 * a plain mock callback isn't recognized by React as an update source
 * worth flushing synchronously without this. */
function emit(ws: MockWebSocket, event: unknown) {
  act(() => {
    ws.onmessage?.({ data: JSON.stringify(event) });
  });
}

function open(ws: MockWebSocket) {
  act(() => {
    ws.onopen?.();
  });
}

function triggerError(ws: MockWebSocket) {
  act(() => {
    ws.onerror?.();
  });
}

function submitMessage(text: string) {
  fireEvent.change(screen.getByPlaceholderText("Message Sarva…"), {
    target: { value: text },
  });
  fireEvent.click(screen.getByRole("button", { name: /send/i }));
}

describe("App", () => {
  it("shows the empty state before any message is sent", async () => {
    await renderApp();
    expect(screen.getByText("Say something to get started.")).toBeInTheDocument();
  });

  it("adds a user bubble and opens a WebSocket carrying the message on submit", async () => {
    await renderApp();
    submitMessage("hello sarva");

    expect(screen.getByText("hello sarva")).toBeInTheDocument();

    const ws = latestSocket();
    expect(ws.url).toMatch(/\/ws\/chat$/);
    open(ws);
    expect(JSON.parse(ws.sent[0])).toEqual({ message: "hello sarva", session: "web" });
  });

  it("accumulates text_delta events into the assistant bubble as they stream in", async () => {
    await renderApp();
    submitMessage("what's the weather?");

    const ws = latestSocket();
    open(ws);
    emit(ws, { type: "model_stream", event: { type: "text_delta", text: "It's " } });
    emit(ws, { type: "model_stream", event: { type: "text_delta", text: "sunny " } });
    emit(ws, { type: "model_stream", event: { type: "text_delta", text: "today." } });

    expect(screen.getByText("It's sunny today.")).toBeInTheDocument();
  });

  it("re-enables the composer and shows nothing extra on a clean run_done", async () => {
    await renderApp();
    submitMessage("hi");

    const ws = latestSocket();
    open(ws);
    emit(ws, { type: "model_stream", event: { type: "text_delta", text: "hello" } });
    emit(ws, { type: "run_done", state: "done", final_message: null });

    const input = screen.getByPlaceholderText("Message Sarva…") as HTMLInputElement;
    expect(input.disabled).toBe(false);
    expect(ws.closed).toBe(true);
    expect(screen.queryByText(/run ended/)).not.toBeInTheDocument();
  });

  it("shows an error message when the run ends in a non-done state", async () => {
    await renderApp();
    submitMessage("this will fail");

    const ws = latestSocket();
    open(ws);
    emit(ws, { type: "run_done", state: "failed", final_message: null });

    expect(screen.getByText(/run ended: failed/)).toBeInTheDocument();
  });

  it("disables the composer while a response is streaming", async () => {
    await renderApp();
    const input = screen.getByPlaceholderText("Message Sarva…") as HTMLInputElement;
    submitMessage("hi");

    expect(input.disabled).toBe(true);
    expect(screen.getByRole("button", { name: /thinking/i })).toBeDisabled();
  });

  it("shows a connection error and re-enables the composer on a socket error", async () => {
    await renderApp();
    submitMessage("hi");

    const ws = latestSocket();
    triggerError(ws);

    expect(screen.getByText(/connection error/)).toBeInTheDocument();
    const input = screen.getByPlaceholderText("Message Sarva…") as HTMLInputElement;
    expect(input.disabled).toBe(false);
  });

  it("shows an Approve/Deny card on needs_confirmation and sends the reply on Approve", async () => {
    await renderApp();
    submitMessage("delete something");

    const ws = latestSocket();
    open(ws);
    emit(ws, {
      type: "needs_confirmation",
      call: { id: "c1", name: "delete_thing", arguments: { path: "x.txt" } },
    });

    expect(screen.getByText("delete_thing")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /approve/i }));

    expect(JSON.parse(ws.sent.at(-1)!)).toEqual({ approved: true });
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("sends {approved: false} on Deny and dismisses the card", async () => {
    await renderApp();
    submitMessage("delete something");

    const ws = latestSocket();
    open(ws);
    emit(ws, {
      type: "needs_confirmation",
      call: { id: "c1", name: "delete_thing", arguments: {} },
    });
    fireEvent.click(screen.getByRole("button", { name: /deny/i }));

    expect(JSON.parse(ws.sent.at(-1)!)).toEqual({ approved: false });
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("renders tool_started and tool_finished as inline status lines", async () => {
    await renderApp();
    submitMessage("write a file");

    const ws = latestSocket();
    open(ws);
    emit(ws, {
      type: "tool_started",
      call: { id: "c1", name: "write_file", arguments: { path: "hi.txt" } },
    });
    emit(ws, { type: "tool_finished", result: { is_error: false }, seconds: 0.01 });

    expect(screen.getByText(/write_file/)).toBeInTheDocument();
    expect(screen.getByText(/ok/)).toBeInTheDocument();
  });

  it("clears the confirmation card on run_done even if never answered", async () => {
    await renderApp();
    submitMessage("delete something");

    const ws = latestSocket();
    open(ws);
    emit(ws, {
      type: "needs_confirmation",
      call: { id: "c1", name: "delete_thing", arguments: {} },
    });
    emit(ws, { type: "run_done", state: "done", final_message: null });

    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });
});
