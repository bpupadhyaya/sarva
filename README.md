# Sarva

An open-source, all-in-one multimodal AGI tool — free for everyone.

Sarva leans on frontier LLMs today (Claude, GPT, Gemini) behind a
model-agnostic provider layer, so it absorbs whatever surpasses them
tomorrow. It's built to be read: every module has a matching chapter in the
docs, because teaching how to build a multimodal AGI tool is as much the
point as the tool itself.

## Status

v0.1.0 (draft release) — the core engine (provider layer with real
Anthropic/OpenAI/Google/Ollama adapters, agent loop, built-in tools
including session persistence and semantic memory recall,
image/audio/video degradation for models that can't handle a modality,
an MCP client so the ecosystem's tools plug in with `sarva run
--mcp-server`, and a benchmark harness that grades every model with the
same yardstick via `sarva eval`), the FastAPI server (REST + WebSocket,
with real tool-confirmation over the socket), and a web UI with a
working chat and tool-approval flow are scaffolded and tested; the CLI
works end to end. The desktop app (Tauri) bundles and auto-starts its
own backend, with real cross-platform release bundles
(macOS/Linux/Windows) — see below for the one-click flow and its known
gaps. The from-scratch model-training track (`foundry/`) has a runnable
(toy-scale) pretraining pipeline: local-file corpus sourcing (exact +
near-duplicate dedup, quality filtering, provenance/license tracking), a
byte-level BPE tokenizer, a dense decoder-only transformer (RoPE,
RMSNorm, SwiGLU, GQA) with an optional Mixture-of-Experts feedforward
(fine-grained experts, a shared expert, aux-loss-free load balancing),
and a training loop with a warmup+cosine LR schedule and bit-identical
checkpoint/resume — see `examples/04_pretrain_and_resume.py`, or
`examples/06_real_corpus_pretraining.py` for the same pipeline run
against real, sourced, licensed public-domain text instead of toy
sentences. Web-scale corpus sourcing and distributed training aren't
built yet. See `BUILD-JOURNAL.md` for progress.

## Quickstart

```bash
git clone https://github.com/bpupadhyaya/sarva
cd sarva
uv sync --all-packages --group dev

# Works with zero configuration (routes to an offline mock model):
uv run sarva chat "hello"

# With a real model:
export ANTHROPIC_API_KEY=sk-...
uv run sarva run "list the files in this directory"

# Or run the server (REST + WebSocket + a web UI at http://127.0.0.1:8000):
uv run sarva serve
```

### Building the web UI

`core/sarva/server/static/` (what `sarva serve` serves at `/`) is a
committed, pre-built copy of `apps/desktop/` — no Node needed to run
Sarva. If you change the UI's source, rebuild and re-copy it:

```bash
./scripts/build-web.sh
```

### Desktop app

```bash
uv sync --all-packages --group dev      # once, to get pyinstaller
./scripts/freeze-server.sh              # freeze the backend into a sidecar binary
cd apps/desktop && npx tauri build --no-bundle
./src-tauri/target/release/sarva-desktop
```

Launching `sarva-desktop` is now the whole experience: it starts its own
bundled backend as a sidecar process and stops it whether you close the
window or kill the app directly (macOS/Linux; Windows signal handling
isn't wired up yet). Real, installable bundles (`.dmg`/`.msi` or
`.exe`/`.AppImage`+`.deb`) for all three OSes come from
`.github/workflows/release-bundle.yml`, triggered manually via
`gh workflow run release-bundle.yml` — genuinely verified to produce
working installers on macOS, Linux, and Windows, not just compile.
Known gap: no code signing yet, so an unsigned build will trigger
Gatekeeper/SmartScreen warnings. See `BUILD-JOURNAL.md` for the full
picture.

## Repository layout

```
core/sarva/       # the reference implementation — providers, agent loop, tools, memory, server, CLI
apps/desktop/      # the web UI (React + TypeScript + Vite) + src-tauri/ (native desktop wrapper, early)
foundry/          # from-scratch model training code (tokenizer, transformer, pretraining, RL)
tests/            # conformance suites — the definition of done for each component
examples/         # small, runnable, graded examples
docs/             # the accompanying book: "Building a Multimodal AGI Tool"
scripts/          # build-web.sh, and future setup/release scripts
```

## License

MIT.
