import { useCallback, useRef, useState } from "react";
import type { AgentEvent } from "./events";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef("web");

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || streaming) return;

    setMessages((prev) => [...prev, { role: "user", text }, { role: "assistant", text: "" }]);
    setInput("");
    setStreaming(true);
    setError(null);

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/chat`);

    ws.onopen = () => {
      ws.send(JSON.stringify({ message: text, session: sessionRef.current }));
    };

    ws.onmessage = (raw: MessageEvent<string>) => {
      const event = JSON.parse(raw.data) as AgentEvent;

      if (event.type === "model_stream" && event.event.type === "text_delta" && event.event.text) {
        const delta = event.event.text;
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          next[next.length - 1] = { ...last, text: last.text + delta };
          return next;
        });
      } else if (event.type === "run_done") {
        setStreaming(false);
        if (event.state !== "done") {
          setError(`run ended: ${event.state}`);
        }
        ws.close();
      }
    };

    ws.onerror = () => {
      setError("connection error — is `sarva serve` running?");
      setStreaming(false);
    };
  }, [input, streaming]);

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
