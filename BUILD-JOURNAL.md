# Sarva — Build Journal

One entry per milestone: what was built, what was verified, what's next.

## 2026-07-21 — T0/T1: core engine scaffold

**Built:**
- Monorepo: uv workspace (`core/` = `sarva`, `foundry/` = `sarva_foundry`).
- `sarva.multimodal.content` — the typed multimodal block model (text, thinking,
  image, audio, video, document, tool call/result), frozen + discriminated union,
  with a degradation registry for cross-modality fallback.
- `sarva.providers` — the provider contract (`Provider` protocol, streaming
  `ProviderEvent`s, typed errors), the model registry + router (`models.yaml`
  / `routing.yaml`, data-driven), a deterministic `MockProvider` (offline,
  scriptable — drives the whole test suite with no API key), and a first
  Anthropic adapter (adaptive thinking, streaming, effort).
- `sarva.agent` — the agent loop (explicit state machine, budgets, concurrent
  tool dispatch, confirm-gated destructive tools, append-only JSONL transcript),
  built-in tools (read/write file, shell — sandboxed to the working directory).
- `sarva` CLI (`sarva chat`, `sarva run`, `sarva models`) — zero-config by
  default (falls back to the offline mock model with no API key set).
- 25 conformance tests across content/provider/agent — **all passing**.
- CI (GitHub Actions: lint + format check + test), ruff-clean, ruff-formatted.
- MkDocs skeleton (Part I, Chapter 1) and `examples/01_hello_model.py`.

**Real bugs found and fixed while building (not just theorized):**
- Pydantic's default JSON `bytes` serialization tries UTF-8 decode, which
  breaks on genuine binary image/audio data — fixed with
  `ser_json_bytes/val_json_bytes="base64"` on every content block.
- Rich's `console.print` interprets literal `[`/`]` in dynamic text as markup
  tags and silently drops unrecognized ones — fixed by escaping dynamic CLI
  output and disabling markup parsing for raw model text.

**Known T1 simplifications (documented in code, not hidden):**
- Multimodal degradation isn't wired into the agent loop yet — T1 tools are
  text-only, so this doesn't bite until T2's multimodal I/O pipeline lands.
- Concurrent tool-call events (`tool_started`/`tool_finished`) are yielded in
  two grouped passes rather than true wall-clock interleaving. Tools still
  *execute* concurrently; only event ordering is batched. Tightens in T3
  when a live-progress UI needs it.
- The Anthropic adapter is written to the documented SDK pattern but has
  **not been exercised against a live API key** in this environment — needs
  a real run to fully validate (its conformance tests are mock-only for now).
- `sarva run` with only the mock provider available never calls tools — the
  mock is a dumb echo with no reasoning, so tool-calling only shows up with
  a real model. Expected, not a bug.

**Next:** Ollama local adapter, live-key validation of the Anthropic adapter,
more built-in tools (web fetch/search), then T2 (multimodal I/O pipeline).

## 2026-07-21 — T1 continued: Ollama adapter, web_fetch, graded examples

**Built:**
- `sarva.providers.ollama_provider.OllamaProvider` — talks to a local Ollama
  server's `/api/chat` (NDJSON streaming), tool-call translation, namespaced
  model-id stripping (`ollama/qwen3:8b` -> `qwen3:8b`). CLI now probes
  `localhost:11434` (fast, 0.3s timeout) and routes to it automatically when
  reachable — this is what makes the "free & private" tier real.
- `WebFetchTool` — non-destructive http(s) fetch tool (scheme-validated,
  truncated, error-handled), added to `BUILTIN_TOOLS`.
- `tests/live/` — a new marker tier (`@pytest.mark.live`, skipped by default
  via `-m 'not live'`) holding real-adapter conformance tests for Anthropic
  and Ollama, `skipif`-gated on credentials/reachability so CI stays green
  with zero secrets while still documenting what "done" means for these
  adapters once run against the real thing.
- `tests/conformance/test_tools.py` — file round-trip, path-escape rejection,
  URL-scheme rejection, one live-marked real fetch.
- **Examples 02–05**: tool use, budget-exceeded (clean stop, no hang),
  confirmation-gating (destructive tool denied, loop continues), and a
  real-model web-fetch demo (needs `ANTHROPIC_API_KEY`, degrades to a clear
  message without one). Examples 01–04 actually **executed** in this
  environment and produced the expected teaching output; 05 verified to
  fail gracefully with no key.

**Verified, not just written:** full lint (`ruff check` + `format --check`)
clean, 29/29 mock-tier tests passing (3 live tests correctly deselected),
all four offline examples run end-to-end with correct output.

**Known gaps (documented, not hidden):**
- Neither the Anthropic nor the Ollama adapter has been exercised against a
  real endpoint in this sandboxed environment — both are written to their
  documented API shapes and covered by `tests/live/`, but that suite has
  never actually run. Treat both as **unverified until someone runs them
  with real credentials / a real local server.**
- Ollama's `/api/chat` doesn't return token usage — `Usage()` defaults to
  zero for local models; cost tracking for local inference stays $0, which
  is correct, but "tokens used" will read 0 too, which understates true
  context consumption. Revisit if budget-by-tokens matters for local runs.

**Next:** T2 — multimodal I/O pipeline (wire `degrade_message` into the
agent loop; image input end-to-end; first audio path).

## 2026-07-21 — T2 started: image input end-to-end

**Built:**
- `AgentLoop.run()` gained `extra_content: list[ContentBlock] | None` — a
  purely additive parameter (every existing `task: str`-only call site is
  unaffected) that attaches non-text blocks to the initiating user turn.
- A new `_required_modalities()` helper scans the conversation for the
  modalities actually present and the loop now calls
  `router.pick(needs=...)` with it, instead of always assuming text-only.
  A message with an image now correctly routes to a vision-capable model.
- `AnthropicProvider._to_anthropic_message` now encodes `ImageBlock` into
  the real Anthropic API image content-block shape (base64 `source`) — this
  closed a real gap: the content model had `ImageBlock` since T0, but no
  adapter could actually *send* one until now.
- `sarva chat --image path.png "..."` — CLI support for attaching an image,
  with a friendly rejection (`typer.BadParameter`) for non-image files.
- Router-failure hardening: if no available model supports what the
  conversation needs (e.g. an image with only text-only models configured),
  the loop now yields a clean `FAILED` terminal event instead of letting
  `router.pick`'s `LookupError` escape the generator unhandled — a real bug
  that existed since T0/T1 and was only caught while building this feature.

**Verified, not just written:**
- 36/36 tests passing (7 new: modality computation, clean-failure-on-no-model,
  text-only regression guard, image-block base64 round-trip, tool call/result
  translation — all pure/offline, no network).
- **Ran the actual CLI against a real generated PNG** (`sarva chat --image`):
  confirmed the image is correctly routed to `mock` (registered as
  vision-capable) and the run completes `DONE` — the full
  CLI → ContentBlock → loop → `_required_modalities` → `Router.pick` →
  provider pipeline verified working, not just unit-tested in isolation.
- Verified the non-image-file rejection path produces a clean CLI error
  (exit code 2, readable message) rather than a stack trace.
- Fixed a ruff false-positive (B008 on `typer.Option` defaults — required by
  typer's own introspection, not a mutable-default bug) via
  `extend-immutable-calls` rather than suppressing the rule wholesale.

**Known gaps (documented, not hidden):**
- This is *routing* awareness, not full *degradation* — `degrade_message`
  (the recursive video→frames→text style fallback from spec-02) still isn't
  called anywhere. Today: either a model supports what's in the message, or
  the run fails cleanly. Graceful downgrading (e.g. auto-describing an image
  for a text-only model) is real T2 remaining work.
- `ImageBlock` only supports `data`/`path` sources end-to-end — a `url`
  source still raises in `resolve_bytes()` (unimplemented `fetch` module, as
  spec-02 already documented).
- No audio or document (PDF) path yet — image was the first modality wired
  because it's the one with a real vision-capable model already registered.

**Next:** first audio path (transcription-based degradation for text-only
models), or content-level degradation for image→text as the alternative to
routing failure. Then continue toward T3 (server + web UI).

## 2026-07-21 — Session persistence (memory, first slice)

**Built:**
- `sarva.memory.session.SessionStore` — file-based session persistence
  (`~/.sarva/sessions/<name>.json`, one JSON file per session, plain and
  greppable per the design doc's memory philosophy). Session names are
  validated against `[A-Za-z0-9_-]+` and **rejected** (not silently
  stripped) if invalid — silent sanitization risked two distinct names
  colliding onto the same file and corrupting history.
- `sarva chat --session <name>` now remembers: loads prior history before
  the run, appends the new user+assistant turn, saves after. Omitting
  `--session` keeps the original one-shot behavior unchanged (no regression).
- `sarva sessions list` / `sarva sessions clear <name>` — inspect and manage
  saved sessions.
- 8 new conformance tests (round trip, binary content survives, missing
  session behavior, name validation, clear/list).

**Verified, not just written:** ran two genuinely separate CLI process
invocations against a scratch `$HOME`, confirmed the second call actually
loaded the first call's history (4 total messages after 2 calls, correct
role/content), confirmed `sessions list`/`clear` operate on the real file.

**Scope, stated plainly:** this is proven correct only for `sarva chat`,
which never uses tools — the full turn is provably exactly `[user message,
final assistant message]`, safe to reconstruct from `RunDoneEvent`.
`sarva run` (which does use tools) is **not** wired for `--session` yet:
reconstructing history across multiple model/tool rounds needs either a
richer return value from the loop or a transcript replay, and building
either without getting the ordering subtly wrong deserved its own slice of
work rather than a rushed add-on here.

**Next:** extend session persistence to `sarva run` (likely via transcript
replay, since every run already writes `transcript.jsonl`), or move to T3
(FastAPI server + web UI) — whichever proves more valuable next iteration.

## 2026-07-21 — T3 started: FastAPI server (REST + WebSocket)

**Built:**
- `sarva.runtime` — extracted the provider/router wiring (Ollama-reachability
  probe, `build_router()`, `build_providers()`) out of `cli.py` into a shared
  module. The server needed the exact same "zero-config, auto-detect
  Ollama" logic as the CLI, and duplicating it would have let the two skins
  drift out of sync on what "available" means — refactored before adding
  the second consumer, not after.
- `sarva.server` — a FastAPI app (`create_app()`) with:
  - `GET /health`, `GET /models` (registry + availability)
  - `POST /chat` — non-streaming, mirrors `sarva chat` exactly (same
    session load/save semantics)
  - `WS /ws/chat` — streams the same `AgentEvent`s the CLI renders, one
    JSON frame per event, ending with `run_done`; single turn per
    connection
- `sarva serve [--host] [--port]` — CLI command, runs the server via
  uvicorn.
- 7 new conformance tests (health, models, chat zero-config, session
  persistence via both REST and WS, WS event streaming) using FastAPI's
  in-process `TestClient` — no real network needed for CI.

**Verified beyond the test suite:** started the actual `sarva serve`
process in the background (real uvicorn, real socket), then hit it with
real `curl` (`/health`, `/models`, `/chat`) and a real `websockets` Python
client against `/ws/chat` — confirmed genuine end-to-end behavior over an
actual TCP connection, not just FastAPI's in-process test transport. Server
process cleanly stopped afterward.

**Known gaps (documented, not hidden):**
- `/ws/chat` is single-turn per connection (matches `sarva chat`'s
  tool-free scope) — no tool-using WS endpoint yet, same limitation as
  `sarva run --session` noted in the previous entry.
- No CORS configuration yet — irrelevant for the CLI-driven smoke tests
  here, but will matter the moment a browser-based web UI (T3's other half)
  tries to call this server from a different origin.
- Picked up a Starlette deprecation warning during testing
  (`httpx`-via-`starlette.testclient` → recommends `httpx2`) — noted, not
  chased; `httpx2` isn't yet an established replacement worth pinning to
  mid-implementation.

**Next:** the web UI (React) that talks to this server, or extend
`/chat`/`/ws/chat` to accept tools (closing the `sarva run` session gap for
both CLI and server at once).

## 2026-07-21 — Closed the tool-use session-persistence gap

**Built:**
- `AgentLoop.run()` gained `transcript_out: list[Message] | None` — purely
  additive (default `None`, every existing call site unaffected). If given,
  it's extended in place with the complete final message list — history
  plus every turn this run appended, including intermediate
  tool-call/tool-result messages — at whatever terminal state the run
  reaches. This is the piece that was missing twice now (flagged in both
  the session-persistence and server-FastAPI journal entries): recovering
  a tool-using run's full history without changing the frozen
  `RunDoneEvent` shape (`final_message` alone only ever carries the *last*
  turn).
- **Found and fixed a real bug while wiring this**, not before: the
  loop only appended the model's message to its own internal `messages`
  list on the `TOOL_USE` path — a plain successful `END_TURN` run never
  added its own final answer to that list. Harmless before now (nothing
  read `messages` from outside), but it would have silently produced
  *wrong* transcripts — missing exactly the final turn — the moment
  anything depended on it. Fixed by moving the append to happen once,
  unconditionally, right after the budget check.
- `sarva chat --session` and `sarva run --session` both switched to
  `transcript_out`, removing the old manual `[history, user, final]`
  reconstruction that only happened to be correct because `chat` never
  used tools. `sarva run --session` **now works for tool-using runs** —
  the gap flagged in the previous two entries is closed.
- The server's `/chat` and `WS /ws/chat` switched to the same pattern for
  consistency (still `tools=[]` — server tool support is a separate,
  bigger decision: confirmation prompts don't have an obvious answer over
  a stateless REST call, and deserves its own design pass, not a rushed
  add-on here).
- 4 new loop tests: full tool-use-round reconstruction, plain-success
  reconstruction (the regression test for the bug above), failure-path
  population, and a not-passed-is-a-no-op guard (16/16 in this file, 55/55
  total).

**Verified, not just written:** the tool-use-round reconstruction is
proven by a dedicated test using a scripted mock (deterministic — no real
model can be made to reliably choose to call a tool, so this is the
correct verification tool for this specific claim). Separately, ran
`sarva run --session` through two real, separate CLI process invocations
and confirmed history persists correctly for the (mock-driven, tool-free)
path the CLI can actually exercise without a live model. Both forms of
verification are honestly reported as what they are — a unit test for the
tool-round mechanics, a live CLI run for the process-level plumbing —
rather than overstating either as covering the other.

**Next:** the web UI (React), or a considered design for server-side tool
confirmation (REST vs. a stateful WS round-trip) before adding tools to
`/chat`/`/ws/chat`.

## 2026-07-21 — T3: the web UI, and `sarva serve` becomes a complete browser experience

**Built:**
- `apps/desktop/` — a real React + TypeScript + Vite app (per the design
  doc's decided stack), hand-written rather than scaffolded from a
  template, kept minimal and readable: a chat UI that opens `/ws/chat`,
  streams `text_delta` events into a live-growing assistant bubble, and
  ends cleanly on `run_done`. Dark-mode aware via `prefers-color-scheme`.
- A small local `events.ts` mirroring `sarva.agent.events.AgentEvent`'s
  JSON shape — scoped deliberately to this app for now rather than
  factored into the design doc's planned `sdks/typescript/` package, since
  it has exactly one consumer today; noted as the natural next home once a
  second one shows up.
- **`sarva serve` now serves the whole thing.** `core/sarva/server/app.py`
  conditionally mounts a built UI at `/` (`StaticFiles`, only if
  `sarva/server/static/` exists — API-only mode still works if it
  doesn't). The static assets are the *committed, built output* of
  `apps/desktop/`, copied into the Python package so `pip install sarva`
  users get a working web UI without needing Node installed. This is a
  **manual step for now** (build, then copy) — a real release pipeline
  (T4/CI territory) should automate rebuilding on every release instead;
  documented as a known limitation, not silently glossed over.
- `.gitignore`: added `node_modules/` and `*.tsbuildinfo`. The generic
  `dist/` rule already inherited from the Python template happens to also
  cover Vite's build output — verified with `git check-ignore`, not
  assumed.

**Verified — real build, real server, real routing, not just code review:**
- Ran `npm install` + `tsc -b` + `vite build` for real: 27 modules
  transformed, zero type errors, a genuine production bundle produced.
- Started the actual `sarva serve` process (again, not the in-process test
  client) and confirmed with `curl`: `/health` and `/models` (explicit API
  routes) are **not shadowed** by the `/` static mount; `/` correctly
  serves the built `index.html`; the exact hashed asset paths Vite
  generated (`/assets/index-*.js`, `/assets/index-*.css`) resolve with
  `200` through the FastAPI mount — proving the asset-linking actually
  works end-to-end through this serving path, not just under Vite's own
  dev server.
- Full test suite still green afterward (55/55) — the static mount didn't
  regress anything.

**Known gaps (documented, not hidden):**
- The chat UI is text-only — no image attach button yet (the CLI's
  `--image` flag has no UI equivalent), and no tool-use rendering (the
  backend doesn't support tools over `/ws/chat` yet either — see the
  previous entry's note on needing a real confirmation-over-WS design).
- No `npm test`/component tests for the React app yet — verified via a
  real build + real server + real HTTP requests this round, which is
  meaningfully better than nothing, but not the same as unit-level
  coverage of the UI's own logic (e.g. the streaming-delta accumulation).
- Static-asset build is a manual, un-automated step (see above) — a stale
  `sarva/server/static/` after a UI source change is a real risk until a
  CI job (or at minimum a `Makefile`/script) rebuilds and re-copies it
  automatically.

**Closed within this same entry:** added `scripts/build-web.sh` (build +
copy in one command, actually run and verified to reproduce the identical
build) — the "manual step" risk above is now "run one script before
committing" rather than "remember several commands in the right order."
Still not CI-automated, but meaningfully lower-risk.

**Next:** UI component tests, or the tool-confirmation-over-WS design
needed before tool-using conversations can reach the browser.
