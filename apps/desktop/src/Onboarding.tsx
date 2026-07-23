import { useCallback, useEffect, useState } from "react";

/**
 * The first-run guided setup T4's own definition of done (and the
 * README's own quickstart text) has promised since T4: "guided setup
 * offers (a) 'Free & private' → pulls a local model, or (b) 'Frontier
 * quality' → paste an API key." Confirmed missing until now — this was
 * a real gap between what the docs promised and what a non-technical
 * user double-clicking the app actually got (a bare chat window with no
 * path to configure anything).
 *
 * Reachable state is decided by GET /doctor -- the exact same
 * `run_diagnostics()` sarva.runtime and `sarva doctor` already use, so
 * "is anything configured" can never drift between what this screen
 * decides and what the CLI would report for the same install.
 */

interface DoctorCheck {
  name: string;
  ok: boolean;
  detail: string;
}

const DISMISSED_KEY = "sarva_onboarding_dismissed";

type Provider = "anthropic" | "openai" | "gemini";

const PROVIDER_LABELS: Record<Provider, string> = {
  anthropic: "Anthropic (Claude)",
  openai: "OpenAI",
  gemini: "Google (Gemini)",
};

async function fetchDoctor(): Promise<DoctorCheck[]> {
  const res = await fetch("/doctor");
  if (!res.ok) throw new Error(`GET /doctor failed: ${res.status}`);
  return (await res.json()) as DoctorCheck[];
}

function isAnyProviderConfigured(checks: DoctorCheck[]): boolean {
  const relevant = new Set(["Anthropic API key", "OpenAI API key", "Google API key", "Ollama (local models)"]);
  return checks.some((c) => relevant.has(c.name) && c.ok);
}

export default function Onboarding({ onComplete }: { onComplete: () => void }) {
  const [checks, setChecks] = useState<DoctorCheck[] | null>(null);
  const [checking, setChecking] = useState(false);
  const [provider, setProvider] = useState<Provider>("anthropic");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setChecking(true);
    try {
      const result = await fetchDoctor();
      setChecks(result);
      if (isAnyProviderConfigured(result)) {
        onComplete();
      }
    } catch {
      // Server not reachable yet, or briefly restarting -- leave the
      // screen showing its last known state rather than crashing on it.
    } finally {
      setChecking(false);
    }
  }, [onComplete]);

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ollamaCheck = checks?.find((c) => c.name === "Ollama (local models)");

  const saveKey = useCallback(async () => {
    if (!apiKey.trim()) return;
    setSaving(true);
    setSaveError(null);
    try {
      const body =
        provider === "anthropic"
          ? { anthropic_api_key: apiKey.trim() }
          : provider === "openai"
            ? { openai_api_key: apiKey.trim() }
            : { gemini_api_key: apiKey.trim() };
      const res = await fetch("/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`save failed: ${res.status}`);
      const result = (await res.json()) as DoctorCheck[];
      setChecks(result);
      if (isAnyProviderConfigured(result)) {
        onComplete();
      } else {
        setSaveError("Saved, but that key doesn't look valid — double-check it and try again.");
      }
    } catch {
      setSaveError("Could not reach the server to save this key. Is `sarva serve` running?");
    } finally {
      setSaving(false);
    }
  }, [apiKey, provider, onComplete]);

  const skip = useCallback(() => {
    try {
      window.localStorage.setItem(DISMISSED_KEY, "1");
    } catch {
      // Non-fatal -- see App.tsx's matching guard around the read side.
    }
    onComplete();
  }, [onComplete]);

  return (
    <div className="onboarding">
      <header>
        <h1>Sarva</h1>
        <p className="subtitle">An open, all-in-one multimodal AGI tool.</p>
      </header>

      <main className="onboarding-body">
        <p className="onboarding-intro">Choose how Sarva should talk to a model:</p>

        <section className="onboarding-option">
          <h2>Free &amp; private</h2>
          <p>Runs entirely on this machine via Ollama — no API key, no data leaves your computer.</p>
          {ollamaCheck?.ok ? (
            <p className="onboarding-status ok">✓ Ollama is already reachable — you're set.</p>
          ) : (
            <>
              <p className="onboarding-status">
                Ollama isn't reachable yet. Install it from{" "}
                <a href="https://ollama.com" target="_blank" rel="noreferrer">
                  ollama.com
                </a>
                , then run:
              </p>
              <code className="onboarding-code">ollama pull qwen3:8b</code>
            </>
          )}
          <button type="button" onClick={refresh} disabled={checking}>
            {checking ? "Checking…" : "Check again"}
          </button>
        </section>

        <section className="onboarding-option">
          <h2>Frontier quality</h2>
          <p>Paste an API key from a frontier provider — saved locally, never sent anywhere but that provider.</p>
          <div className="onboarding-key-form">
            <select value={provider} onChange={(e) => setProvider(e.target.value as Provider)}>
              {(Object.keys(PROVIDER_LABELS) as Provider[]).map((p) => (
                <option key={p} value={p}>
                  {PROVIDER_LABELS[p]}
                </option>
              ))}
            </select>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Paste API key…"
            />
            <button type="button" onClick={saveKey} disabled={saving || !apiKey.trim()}>
              {saving ? "Saving…" : "Save & Continue"}
            </button>
          </div>
          {saveError && <p className="error">{saveError}</p>}
        </section>

        <button type="button" className="onboarding-skip" onClick={skip}>
          Skip for now (uses a free, offline demo model)
        </button>
      </main>
    </div>
  );
}

export { DISMISSED_KEY, isAnyProviderConfigured };
export type { DoctorCheck };
