# Sarva — Building a Multimodal AGI Tool

Sarva is an open-source, all-in-one AGI tool, free for the global population,
and this site is the book that comes with it: how it's built, and how to
build one yourself.

## Part I — Foundations (in progress)

### Chapter 1: What "AGI tool" means today

An AGI *tool* is the system wrapped around a frontier or open model that
turns raw next-token prediction into something a person can actually use to
get work done: an agent loop that plans and verifies, a multimodal pipeline
that lets it see and hear, memory that persists across sessions, and tools
that let it act in the world. The model supplies intelligence; the tool
supplies everything that makes that intelligence *usable*.

Sarva leans on frontier LLMs (Claude, GPT, Gemini) for that intelligence
layer today, behind a provider abstraction that treats every model — hosted
or local — identically. That choice is deliberate and temporary: as models
more capable than today's frontier arrive, Sarva absorbs them as a one-entry
registry change, never a rewrite. See the roadmap for where this leads.

*(Chapters 2+ — the provider abstraction, model registry and routing, the
agent loop, multimodality, memory, and packaging for humans — land as the
core engine is built. Each chapter mirrors a module in `core/sarva/`.)*

## Quickstart

```bash
git clone https://github.com/bpupadhyaya/sarva
cd sarva
uv sync --all-packages --group dev
uv run sarva chat "hello"          # works with zero configuration
export ANTHROPIC_API_KEY=sk-...
uv run sarva run "list the files in this directory"
```
