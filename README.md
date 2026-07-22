# Sarva

An open-source, all-in-one multimodal AGI tool — free for everyone.

Sarva leans on frontier LLMs today (Claude, GPT, Gemini) behind a
model-agnostic provider layer, so it absorbs whatever surpasses them
tomorrow. It's built to be read: every module has a matching chapter in the
docs, because teaching how to build a multimodal AGI tool is as much the
point as the tool itself.

## Status

Early — the core engine (provider layer, agent loop, built-in tools,
session persistence), the FastAPI server (REST + WebSocket, with real
tool-confirmation over the socket), and a web UI with a working chat and
tool-approval flow are scaffolded and tested; the CLI works end to end.
The desktop app (Tauri) now bundles and auto-starts its own backend — see
below for the real one-click flow and its known gaps. The from-scratch
model-training track (`foundry/`) has started: a byte-level BPE tokenizer
and a dense decoder-only transformer (RoPE, RMSNorm, SwiGLU, GQA) are
built and tested (see `examples/02_train_a_tokenizer.py` and
`examples/03_train_toy_transformer.py`); the pretraining data pipeline and
a real (non-toy) training loop aren't built yet. See `BUILD-JOURNAL.md`
for progress.

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
bundled backend as a sidecar process and stops it when you close the
window — no terminal, no manual `sarva serve` step. Known gaps: a
force-quit or crash (as opposed to closing the window) can leave the
sidecar process running; there's no code signing yet, so an unsigned
build will trigger Gatekeeper/SmartScreen warnings; and only macOS arm64
is verified so far. See `BUILD-JOURNAL.md` for the full picture.

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
