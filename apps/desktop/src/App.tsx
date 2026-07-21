import { useCallback, useRef, useState } from "react";
import type { AgentEvent } from "./events";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}

interface PendingConfirmation {
  name: string;
  arguments: Record<string, unknown>;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingConfirmation | null>(null);
  const sessionRef = useRef("web");
  const socketRef = useRef<WebSocket | null>(null);

  const appendToLastAssistant = useCallback((suffix: string) => {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, text: last.text + suffix };
      return next;
    });
  }, []);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || streaming) return;

    setMessages((prev) => [...prev, { role: "user", text }, { role: "assistant", text: "" }]);
    setInput("");
    setStreaming(true);
    setError(null);
    setPending(null);

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/chat`);
    socketRef.current = ws;

    ws.onopen = () => {
      // auto: false (the default) — every destructive tool call pauses for
      // an explicit Approve/Deny in the UI before it runs. See
      // core/sarva/server/app.py's ws_chat docstring for the protocol.
      ws.send(JSON.stringify({ message: text, session: sessionRef.current }));
    };

    ws.onmessage = (raw: MessageEvent<string>) => {
      const event = JSON.parse(raw.data) as AgentEvent;

      if (event.type === "model_stream" && event.event.type === "text_delta" && event.event.text) {
        appendToLastAssistant(event.event.text);
      } else if (event.type === "tool_started") {
        appendToLastAssistant(`\n→ ${event.call.name}(${JSON.stringify(event.call.arguments)})`);
      } else if (event.type === "tool_finished") {
        appendToLastAssistant(event.result.is_error ? "  error" : "  ok");
      } else if (event.type === "needs_confirmation") {
        setPending({ name: event.call.name, arguments: event.call.arguments });
      } else if (event.type === "run_done") {
        setStreaming(false);
        setPending(null);
        if (event.state !== "done") {
          setError(`run ended: ${event.state}`);
        }
        ws.close();
      }
    };

    ws.onerror = () => {
      setError("connection error — is `sarva serve` running?");
      setStreaming(false);
      setPending(null);
    };
  }, [input, streaming, appendToLastAssistant]);

  const respondToConfirmation = useCallback((approved: boolean) => {
    socketRef.current?.send(JSON.stringify({ approved }));
    setPending(null);
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Sarva</h1>
        <p className="subtitle">An open, all-in-one multimodal AGI tool.</p>
      </header>

      <main className="messages">
        {messages.length === 0 && <p className="empty">Say something to get started.</p>}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <span className="role">{m.role === "user" ? "you" : "sarva"}</span>
            <p>{m.text || (streaming && i === messages.length - 1 ? "…" : "")}</p>
          </div>
        ))}
        {error && <p className="error">{error}</p>}
      </main>

      {pending && (
        <div className="confirm-card" role="alertdialog" aria-label="Confirm tool use">
          <p>
            Allow <code>{pending.name}</code>({JSON.stringify(pending.arguments)})?
          </p>
          <div className="confirm-buttons">
            <button type="button" className="approve" onClick={() => respondToConfirmation(true)}>
              Approve
            </button>
            <button type="button" className="deny" onClick={() => respondToConfirmation(false)}>
              Deny
            </button>
          </div>
        </div>
      )}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message Sarva…"
          disabled={streaming}
          autoFocus
        />
        <button type="submit" disabled={streaming || !input.trim()}>
          {streaming ? "Thinking…" : "Send"}
        </button>
      </form>
    </div>
  );
}
