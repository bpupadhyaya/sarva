import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Onboarding from "./Onboarding";

function mockFetchSequence(responses: Array<{ ok: boolean; json: unknown }>) {
  const fn = vi.fn();
  for (const r of responses) {
    fn.mockResolvedValueOnce({ ok: r.ok, json: async () => r.json });
  }
  vi.stubGlobal("fetch", fn);
  return fn;
}

const UNCONFIGURED = [
  { name: "Anthropic API key", ok: false, detail: "ANTHROPIC_API_KEY not set" },
  { name: "OpenAI API key", ok: false, detail: "OPENAI_API_KEY not set" },
  { name: "Google API key", ok: false, detail: "GEMINI_API_KEY/GOOGLE_API_KEY not set" },
  { name: "Ollama (local models)", ok: false, detail: "not reachable" },
  { name: "Foundry (local from-scratch models)", ok: false, detail: "not installed" },
];

describe("Onboarding", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows both setup options while nothing is configured", async () => {
    mockFetchSequence([{ ok: true, json: UNCONFIGURED }]);
    render(<Onboarding onComplete={vi.fn()} />);

    await screen.findByText("Free & private");
    expect(screen.getByText("Frontier quality")).toBeInTheDocument();
    expect(screen.getByText(/ollama pull qwen3:8b/)).toBeInTheDocument();
  });

  it("calls onComplete immediately if the initial doctor check already shows a configured provider", async () => {
    const onComplete = vi.fn();
    mockFetchSequence([
      { ok: true, json: [{ name: "Ollama (local models)", ok: true, detail: "reachable" }] },
    ]);

    render(<Onboarding onComplete={onComplete} />);

    await waitFor(() => expect(onComplete).toHaveBeenCalled());
  });

  it("shows Ollama as ready without instructions once it's reachable", async () => {
    mockFetchSequence([
      {
        ok: true,
        json: [
          ...UNCONFIGURED.filter((c) => c.name !== "Ollama (local models)"),
          { name: "Ollama (local models)", ok: true, detail: "reachable at http://localhost:11434" },
        ],
      },
    ]);
    render(<Onboarding onComplete={vi.fn()} />);

    await screen.findByText(/already reachable/);
    expect(screen.queryByText(/ollama pull/)).not.toBeInTheDocument();
  });

  it("re-checks Ollama reachability on demand via Check again", async () => {
    const fetchMock = mockFetchSequence([
      { ok: true, json: UNCONFIGURED },
      {
        ok: true,
        json: [{ name: "Ollama (local models)", ok: true, detail: "reachable at http://localhost:11434" }],
      },
    ]);
    const onComplete = vi.fn();
    render(<Onboarding onComplete={onComplete} />);
    await screen.findByText("Free & private");

    fireEvent.click(screen.getByRole("button", { name: /check again/i }));

    await waitFor(() => expect(onComplete).toHaveBeenCalled());
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("saves a pasted API key via POST /config and completes on success", async () => {
    const onComplete = vi.fn();
    const fetchMock = mockFetchSequence([
      { ok: true, json: UNCONFIGURED },
      {
        ok: true,
        json: [{ name: "Anthropic API key", ok: true, detail: "ANTHROPIC_API_KEY is set" }],
      },
    ]);
    render(<Onboarding onComplete={onComplete} />);
    await screen.findByText("Frontier quality");

    fireEvent.change(screen.getByPlaceholderText("Paste API key…"), {
      target: { value: "sk-ant-real-looking-key" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save & continue/i }));

    await waitFor(() => expect(onComplete).toHaveBeenCalled());
    const [, postCall] = fetchMock.mock.calls;
    expect(postCall[0]).toBe("/config");
    expect(JSON.parse(postCall[1].body)).toEqual({ anthropic_api_key: "sk-ant-real-looking-key" });
  });

  it("shows an error and does not complete when the saved key still isn't recognized", async () => {
    const onComplete = vi.fn();
    mockFetchSequence([
      { ok: true, json: UNCONFIGURED },
      { ok: true, json: UNCONFIGURED }, // still not ok after "saving" a bad key
    ]);
    render(<Onboarding onComplete={onComplete} />);
    await screen.findByText("Frontier quality");

    fireEvent.change(screen.getByPlaceholderText("Paste API key…"), {
      target: { value: "not-a-real-key" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save & continue/i }));

    await screen.findByText(/doesn't look valid/);
    expect(onComplete).not.toHaveBeenCalled();
  });

  it("shows a clear error when the server is unreachable while saving", async () => {
    const onComplete = vi.fn();
    const fetchMock = vi.fn();
    fetchMock.mockResolvedValueOnce({ ok: true, json: async () => UNCONFIGURED });
    fetchMock.mockRejectedValueOnce(new Error("network error"));
    vi.stubGlobal("fetch", fetchMock);

    render(<Onboarding onComplete={onComplete} />);
    await screen.findByText("Frontier quality");

    fireEvent.change(screen.getByPlaceholderText("Paste API key…"), { target: { value: "sk-test" } });
    fireEvent.click(screen.getByRole("button", { name: /save & continue/i }));

    await screen.findByText(/is `sarva serve` running/i);
    expect(onComplete).not.toHaveBeenCalled();
  });

  it("completes via Skip without ever calling POST /config", async () => {
    const onComplete = vi.fn();
    const fetchMock = mockFetchSequence([{ ok: true, json: UNCONFIGURED }]);
    render(<Onboarding onComplete={onComplete} />);
    await screen.findByText("Frontier quality");

    fireEvent.click(screen.getByRole("button", { name: /skip for now/i }));

    expect(onComplete).toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1); // only the initial /doctor check
  });

  it("does not crash and still completes via Skip when the initial /doctor call fails", async () => {
    const onComplete = vi.fn();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    render(<Onboarding onComplete={onComplete} />);

    await screen.findByText("Free & private");
    fireEvent.click(screen.getByRole("button", { name: /skip for now/i }));

    expect(onComplete).toHaveBeenCalled();
  });
});
