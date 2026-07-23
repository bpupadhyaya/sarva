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

**[Chapter 2 — The Provider Abstraction, Model Registry, and
Routing](providers.md)** is live: the `Provider` protocol every backend
implements, five real adapters (Anthropic/OpenAI/Google/Ollama/Mock)
and the genuine wire-format differences writing them surfaced, and how
`models.yaml` + the router turn "absorb a new frontier model" into a
one-entry data edit instead of a rewrite.

**[Chapter 3 — The Agent Loop](agent-loop.md)** is live too: the
explicit plan/act/verify state machine every skin drives, concurrent
tool execution gated by one confirm policy, budgets as a clean stop
rather than an exception, and the opt-in multimodal-degradation
fallback — plus what's honestly not built yet (subagent fan-out, named
in the design but not in code).

**[Chapter 4 — Multimodality](multimodal.md)** is live: the typed
`ContentBlock` vocabulary every layer speaks, the three real
degraders (image/audio/video), and an honestly-named real gap
`degrade_message`'s own "never silently drop" guarantee doesn't
currently reach — `DocumentBlock` is typed but unprocessed by any
provider adapter.

*(Chapters 5+ — memory and packaging for humans — land as their own
chapters get written. Each chapter mirrors a module in `core/sarva/`;
see the nav sidebar for what's already up: memory, MCP, eval,
distillation, and the whole from-scratch foundry track.)*

## Quickstart

```bash
git clone https://github.com/bpupadhyaya/sarva
cd sarva
uv sync --all-packages --group dev
uv run sarva chat "hello"          # works with zero configuration
export ANTHROPIC_API_KEY=sk-...
uv run sarva run "list the files in this directory"
```
