# Sarva

An open-source, all-in-one multimodal AGI tool — free for everyone.

Sarva leans on frontier LLMs today (Claude, GPT, Gemini) behind a
model-agnostic provider layer, so it absorbs whatever surpasses them
tomorrow. It's built to be read: every module has a matching chapter in the
docs, because teaching how to build a multimodal AGI tool is as much the
point as the tool itself.

## Status

Early — the core engine (provider layer, agent loop, built-in tools,
session persistence), the FastAPI server (REST + WebSocket), and a first
web UI are scaffolded and tested; the CLI works end to end. The one-click
desktop app and the from-scratch model-training track (`foundry/`) are
ahead. See `BUILD-JOURNAL.md` for progress.

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

## Repository layout

```
core/sarva/       # the reference implementation — providers, agent loop, tools, memory, server, CLI
apps/desktop/      # the web UI (React + TypeScript + Vite); later wrapped in Tauri for the one-click app
foundry/          # from-scratch model training code (tokenizer, transformer, pretraining, RL)
tests/            # conformance suites — the definition of done for each component
examples/         # small, runnable, graded examples
docs/             # the accompanying book: "Building a Multimodal AGI Tool"
scripts/          # build-web.sh, and future setup/release scripts
```

## License

MIT.
