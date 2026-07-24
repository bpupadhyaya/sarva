import { type ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import type { AgentEvent } from "./events";
import Onboarding, { DISMISSED_KEY, isAnyProviderConfigured } from "./Onboarding";
import type { DoctorCheck } from "./Onboarding";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
}

interface PendingConfirmation {
  name: string;
  arguments: Record<string, unknown>;
}

interface AttachedImage {
  base64: string;
  mediaType: string;
  name: string;
}

interface ModelInfo {
  id: string;
  display_name: string;
  available: boolean;
}

/** Reads a File's raw bytes and base64-encodes them without going through
 * FileReader.readAsDataURL -- avoids the extra data:...;base64, prefix
 * parsing step, and File.arrayBuffer() is the one path this project's own
 * jsdom-backed test environment and every real browser both support
 * identically. */
async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingConfirmation | null>(null);
  const [attachedImage, setAttachedImage] = useState<AttachedImage | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  // "" means auto (no override) -- the exact meaning omitting the CLI's
  // own --model flag has, kept consistent rather than inventing a
  // separate "auto" sentinel value the server would need to know about.
  const [selectedModel, setSelectedModel] = useState("");
  const sessionRef = useRef("web");
  const socketRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // null = still deciding (avoids a first-run screen flashing briefly for
  // an already-configured install while GET /doctor is in flight).
  const [showOnboarding, setShowOnboarding] = useState<boolean | null>(null);

  useEffect(() => {
    // localStorage can be unavailable in some embedded webview contexts
    // (and in this project's own test environment) -- treated as "not
    // dismissed" rather than crashing the whole app on a feature that's
    // a pure convenience, not a correctness requirement.
    let dismissed = false;
    try {
      dismissed = window.localStorage.getItem(DISMISSED_KEY) !== null;
    } catch {
      dismissed = false;
    }
    if (dismissed) {
      setShowOnboarding(false);
      return;
    }
    fetch("/doctor")
      .then((res) => (res.ok ? (res.json() as Promise<DoctorCheck[]>) : Promise.reject()))
      .then((checks) => setShowOnboarding(!isAnyProviderConfigured(checks)))
      .catch(() => setShowOnboarding(false)); // server unreachable -- don't block the chat UI on it
  }, []);

  useEffect(() => {
    // Best-effort: an empty list just means the "Model" picker only ever
    // offers "Auto" -- the same graceful-degradation instinct as the
    // doctor fetch above, not a reason to block the chat UI.
    fetch("/models")
      .then((res) => (res.ok ? (res.json() as Promise<ModelInfo[]>) : Promise.reject()))
      .then(setModels)
      .catch(() => {});
  }, []);

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

    const image = attachedImage;
    setMessages((prev) => [
      ...prev,
      { role: "user", text: image ? `${text} [${image.name}]` : text },
      { role: "assistant", text: "" },
    ]);
    setInput("");
    setAttachedImage(null);
    setStreaming(true);
    setError(null);
    setPending(null);

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/chat`);
    socketRef.current = ws;

    let lastDetail: string | null = null;
    // Set by run_done (clean or failed completion) or onerror (a real
    // WebSocket error) -- onclose only acts when NEITHER already fired,
    // the raw-close case (server process killed mid-stream, a proxy
    // idle timeout, a TCP reset) that used to leave `streaming` stuck
    // true forever with no onclose handler at all: every composer
    // control (text input, attach-image button, model picker, send) is
    // gated on it, so a hung raw close silently locked the entire UI
    // with no recovery short of a page reload.
    let settled = false;

    ws.onopen = () => {
      // auto: false (the default) — every destructive tool call pauses for
      // an explicit Approve/Deny in the UI before it runs. See
      // core/sarva/server/app.py's ws_chat docstring for the protocol.
      // image_base64/image_media_type/model are omitted entirely (not
      // sent as null) when unset, matching the REST /chat request
      // schema's own optional-field shape.
      ws.send(
        JSON.stringify({
          message: text,
          session: sessionRef.current,
          ...(image ? { image_base64: image.base64, image_media_type: image.mediaType } : {}),
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
      );
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
      } else if (event.type === "state_changed" && event.detail) {
        lastDetail = event.detail;
      } else if (event.type === "run_done") {
        settled = true;
        setStreaming(false);
        setPending(null);
        if (event.state !== "done") {
          // The real reason (e.g. an unknown --model equivalent) used to
          // be silently dropped here -- state_changed.detail carries it,
          // the same fix the CLI and /chat's own response got.
          setError(lastDetail ? `run ended: ${event.state} — ${lastDetail}` : `run ended: ${event.state}`);
        }
        ws.close();
      }
    };

    ws.onerror = () => {
      settled = true;
      setError("connection error — is `sarva serve` running?");
      setStreaming(false);
      setPending(null);
    };

    ws.onclose = () => {
      if (settled) return; // already handled by run_done or onerror above
      settled = true;
      setError("connection closed before the run finished — is `sarva serve` still running?");
      setStreaming(false);
      setPending(null);
    };
  }, [input, streaming, attachedImage, selectedModel, appendToLastAssistant]);

  const respondToConfirmation = useCallback((approved: boolean) => {
    socketRef.current?.send(JSON.stringify({ approved }));
    setPending(null);
  }, []);

  const handleFileChange = useCallback(async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file later
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError(`"${file.name}" doesn't look like an image (${file.type || "unknown type"})`);
      return;
    }
    const base64 = await fileToBase64(file);
    setAttachedImage({ base64, mediaType: file.type, name: file.name });
    setError(null);
  }, []);

  if (showOnboarding === null) {
    return null; // deciding — see the effect above for why this is brief and expected
  }
  if (showOnboarding) {
    return <Onboarding onComplete={() => setShowOnboarding(false)} />;
  }

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

      {attachedImage && (
        <div className="attached-image">
          <span>📎 {attachedImage.name}</span>
          <button type="button" onClick={() => setAttachedImage(null)}>
            Remove image
          </button>
        </div>
      )}

      <div className="model-picker">
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={streaming}
        >
          <option value="">Auto</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.display_name}
              {m.available ? "" : " (unavailable)"}
            </option>
          ))}
        </select>
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={{ display: "none" }}
          data-testid="attach-image-input"
        />
        <button
          type="button"
          disabled={streaming}
          onClick={() => fileInputRef.current?.click()}
          aria-label="Attach image"
        >
          📎
        </button>
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
