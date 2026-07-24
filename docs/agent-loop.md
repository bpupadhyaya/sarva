# Chapter 3 — The Agent Loop

Chapter 2 covered how Sarva talks to a model. This chapter is about
what drives that conversation forward: `sarva.agent.loop.AgentLoop`,
the one loop every skin (CLI, server, desktop) runs underneath —
`sarva chat`, `sarva run`, and the WebSocket server all construct the
same `AgentLoop` with different tool lists and a different confirm
policy, never their own copy of the control flow.

## An explicit state machine, not implicit control flow

`AgentLoop.run()` is a state machine with its states and legal
transitions written down as data (`sarva.agent.events.LEGAL`), not
scattered across `if`/`elif` branches:

```
INIT -> CALLING_MODEL
CALLING_MODEL -> RUNNING_TOOLS | DONE | FAILED | INTERRUPTED | BUDGET_EXCEEDED
RUNNING_TOOLS -> AWAITING_CONFIRMATION | CALLING_MODEL | FAILED | INTERRUPTED | BUDGET_EXCEEDED
AWAITING_CONFIRMATION -> RUNNING_TOOLS | INTERRUPTED
```

Every transition goes through one function, `transition(to)`, which
asserts the move is actually legal before taking it. Every skin
consumes only the resulting `AgentEvent` stream (`StateChangedEvent`,
`ModelStreamEvent`, `ToolStartedEvent`/`ToolFinishedEvent`,
`NeedsConfirmationEvent`, `RunDoneEvent`) — none of them reach into
loop internals, and the transcript is the same append-only JSONL file
(`run_dir/transcript.jsonl`) regardless of which skin is driving.

The loop is the plan → act → verify cycle made literal: `CALLING_MODEL`
is the model deciding what to do next (plan), `RUNNING_TOOLS` is acting
on that decision, and looping back to `CALLING_MODEL` afterward — with
the tool results appended to the conversation — is how the model
verifies what just happened and decides whether it's actually done.

## Tool use: concurrent, typed, gated by one policy

A `Tool` is a small, explicit contract:

```python
class Tool(Protocol):
    spec: ToolSpec  # name, description, JSON Schema, and a `destructive: bool` flag
    async def run(self, args: dict, ctx: ToolContext) -> ToolResultBlock: ...
```

`BUILTIN_TOOLS` ships six: `ReadFileTool`, `WriteFileTool`,
`RunShellTool`, `WebFetchTool`, `RememberTool`, `RecallMemoryTool` (the
last two backed by the semantic memory store — see the memory
chapter). MCP-backed tools (see the MCP chapter) implement the exact
same `Tool` protocol, which is why the loop never needs to know or care
whether a given tool call is local Python or a round trip to a
subprocess speaking MCP.

When a model turn ends in `TOOL_USE`, every requested call runs
concurrently via `asyncio.gather` — not sequentially, and not with the
model waiting on one before deciding about the next. **Whether a tool
runs at all is a policy decision, never the tool's own to make**: a
tool declares `spec.destructive`; the loop — not the tool — decides
whether that triggers `AWAITING_CONFIRMATION` and a call out to the
caller's `ConfirmPolicy`. This is what makes `sarva run --auto`
(`always_allow`, never asks) and `sarva run`'s default (a real
`typer.confirm()` prompt per destructive call) the same loop with a
one-argument policy swap, not two different code paths to keep in sync.

A tool that raises never crashes the loop — the exception becomes an
`is_error=True` `ToolResultBlock` the model sees on its next turn, the
same as any other tool failure it needs to react to. An unrecognized
tool name gets the identical treatment rather than a hard stop.

### `WebFetchTool` and a real SSRF gap it had, found and closed

`WebFetchTool` is marked `destructive=False` — deliberately, since
fetching a URL changes no state — which means it runs with **zero
confirmation**, even in the CLI's default (non-`--auto`) mode. That
made a real, not hypothetical, SSRF (server-side request forgery) gap:
confirmed directly against a real local Ollama server running in this
environment, `web_fetch` on `http://127.0.0.1:11434/api/tags`
succeeded and returned the response straight into the model's own
context — the same shape of request would reach a cloud metadata
endpoint (`http://169.254.169.254/...`, a classic SSRF target for
exfiltrating cloud credentials) or any other internal service with
identical ease.

Closed with the standard mitigation: before every fetch, the target
hostname is resolved and every returned IP is checked against
`ipaddress`'s `is_global` (covers RFC 1918 private ranges, loopback,
link-local — which includes the metadata address — and other reserved
ranges, for both IPv4 and IPv6, in one check). `follow_redirects=True`
was replaced with a bounded manual redirect loop that re-validates the
target host on **every** hop, not just the caller-supplied URL — a
legitimate public site's own server issuing a redirect straight to an
internal address is exactly the bypass a validate-once-up-front check
would miss. Verified against real addresses (a real running local
Ollama server, the real cloud-metadata IP, a real private-range IP,
and a simulated redirect to an internal address) and against real
public traffic (a real `https://example.com` fetch, and a real
`http://github.com` → `https://github.com/` redirect chain) — both
still work exactly as before.

**The guard itself now lives in `sarva.multimodal.fetch`
(`ensure_public_host`), not duplicated here** — `resolve_media_bytes()`
(the multimodal chapter) is the *other* real url-fetching path in this
codebase, and it shares the identical function so the SSRF guard can
never drift out of sync between the two.

## Budgets: exceeding one is a clean stop, not an exception

```python
class Budget(BaseModel):
    max_model_calls: int = 50
    max_total_tokens: int = 2_000_000
    max_wall_seconds: float = 3600.0
    max_cost_usd: float = 10.0
```

`Spend.exceeded(budget)` is checked after every model turn; the first
budget that's actually crossed lands the loop in `BUDGET_EXCEEDED` —
a normal terminal state with a full `Spend` summary attached to
`RunDoneEvent`, not a raised exception a caller has to catch. A
runaway agent stops itself cleanly, with a receipt of exactly how much
it spent before stopping.

## Multimodal-aware routing, and degradation as an opt-in fallback

Before the first model call, the loop scans every message for the
modalities actually present (an image attached alongside text, say)
and asks the router for a model that supports all of them —
`needs=_required_modalities(messages)`. If no available model
qualifies, that's normally a hard `FAILED` — unless the loop was
constructed with `degraders` (see the multimodal degraders described
in the memory/eval chapters' sibling material): with degraders
supplied, the failure becomes *recoverable* — fall back to the best
available text-capable model, and degrade the unsupported content
(video → sampled frames → text, say) into something that model can
actually see, rather than refusing outright. Deliberately opt-in, not
a silent default: nobody gets a lower-fidelity response than they
explicitly asked for without having asked for exactly that tradeoff.

## Failure handling, named explicitly rather than left implicit

- A provider crash (any exception escaping `provider.generate()`, not
  just a well-formed `StreamErrorEvent`) is caught at the loop level
  and turned into `FAILED` — it never propagates up into a skin.
- A `StreamErrorEvent` marked `retryable` gets exactly one retry after
  a fixed backoff before the loop gives up; a non-retryable one is an
  immediate `FAILED`.
- `MAX_TOKENS` and `REFUSAL` stop reasons are both `FAILED`, with the
  specific reason recorded in the terminal event's `detail` field —
  distinguishable after the fact, not collapsed into one generic
  failure state.

## What's honestly not built yet

The design doc's own architecture section names "subagent fan-out" and
"verifier subagent" patterns alongside the loop this chapter describes.
**Neither exists in code yet** — `AgentLoop` today drives exactly one
model conversation with one flat tool list; there's no mechanism for
one agent run to spawn and coordinate others. Real, deferred work,
named here rather than implied to already exist because the design doc
mentions it.

## Build it yourself

- Read `tests/conformance/test_agent.py` — `MockProvider` scripts let
  you drive the loop through every state without a real model, which is
  exactly how this project tests budget exhaustion, tool errors,
  confirmation gating, and the degradation fallback without ever
  touching the network.
- Write a tool that raises on purpose and watch the loop keep going —
  the model sees the error and gets to react, the run doesn't crash.
- Construct a `Budget(max_model_calls=1)` and watch a multi-tool-call
  task land in `BUDGET_EXCEEDED` with a real `Spend` summary instead of
  running forever.
- Try `sarva run --auto "some destructive task"` vs. plain `sarva run`
  and watch the exact same loop take two different paths through
  `AWAITING_CONFIRMATION` based on nothing but which `ConfirmPolicy`
  was passed in.
