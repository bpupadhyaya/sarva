# Sarva

An open-source, all-in-one multimodal AGI tool — free for everyone.

Sarva leans on frontier LLMs today (Claude, GPT, Gemini) behind a
model-agnostic provider layer, so it absorbs whatever surpasses them
tomorrow. It's built to be read: every module has a matching chapter in the
docs, because teaching how to build a multimodal AGI tool is as much the
point as the tool itself.

## Status

Early — the core engine (provider layer, agent loop, built-in tools) is
scaffolded and tested; the CLI works end to end. Multimodal I/O, the server,
the one-click desktop app, and the from-scratch model-training track
(`foundry/`) are ahead. See `BUILD-JOURNAL.md` for progress.

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
```

## Repository layout

```
core/sarva/       # the reference implementation — providers, agent loop, tools, CLI
foundry/          # from-scratch model training code (tokenizer, transformer, pretraining, RL)
tests/            # conformance suites — the definition of done for each component
examples/         # small, runnable, graded examples
docs/             # the accompanying book: "Building a Multimodal AGI Tool"
```

## License

MIT.
