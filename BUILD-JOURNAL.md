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

## 2026-07-21 — UI component tests + CI now covers the web app

**Built:**
- Vitest + React Testing Library wired into `apps/desktop/` (`npm run
  test`). 7 tests covering `App.tsx`'s own logic: empty state, WebSocket
  URL/payload on send, streaming-delta accumulation into the assistant
  bubble, clean success, failure-state error display, composer
  disabled-while-streaming, and connection-error handling. A small mock
  `WebSocket` class drives these deterministically — real WebSocket
  delivery is already proven end-to-end (previous entry: a real
  `sarva serve` process hit with a real `websockets` client), so this mock
  exists to test the UI's *reaction* to events, not to re-prove transport.
- CI (`.github/workflows/ci.yml`) gained a second job, `web`: npm install,
  typecheck, test, build — the frontend was completely unverified in CI
  until now. Also fixed a real gap in the existing Python job: `examples/`
  was linted locally every milestone but never actually in CI's lint
  command — added it.
- **A CI check with teeth for the exact risk flagged last entry:** after
  building, CI now diffs the fresh `dist/` against the committed
  `core/sarva/server/static/` and fails with a clear message if they
  differ — turning "a human might forget to run `scripts/build-web.sh`
  before committing" from a documented risk into something CI actually
  catches.

**Real bugs found while writing these tests, not before:**
- Testing Library's DOM auto-cleanup between tests silently doesn't
  register without Vitest's `globals: true` (which this project
  deliberately doesn't use, preferring explicit imports) — every test
  after the first was finding duplicate elements from prior tests' unmounted
  DOM. Fixed with an explicit `afterEach(cleanup)` in `setupTests.ts`.
- Manually invoking the mock WebSocket's `onmessage`/`onopen`/`onerror`
  callbacks from test code doesn't reliably flush the resulting React state
  update before the next assertion runs — these calls aren't recognized as
  React-managed events the way `fireEvent` is. Fixed by wrapping each
  simulated callback in `act()`.

**Verified the CI check isn't just decorative:** actually broke the static
bundle on purpose (changed visible UI text, rebuilt, diffed) and confirmed
the check catches it with a clear failure message, then reverted and
confirmed it passes clean again — the same discipline applied to every
claim of "this works" all week, now applied to a CI check about CI checks.

**Next:** the tool-confirmation-over-WS design needed before tool-using
conversations can reach the browser, or continue toward T4 (Tauri desktop
wrapper).

## 2026-07-21 — Tool confirmation over WebSocket: the browser can now act, safely

**Built:**
- `/ws/chat` now runs with `BUILTIN_TOOLS` enabled (mirrors `sarva run`,
  not `sarva chat` — documented explicitly, since `/chat` stays tool-free:
  a stateless REST request can't naturally pause mid-request for a
  confirmation round-trip, which is exactly why this lives on the
  WebSocket). Client sends `{"message", "session", "auto"}`; a destructive
  tool call pauses the run and sends `needs_confirmation`, then the
  *next* value the client sends — `{"approved": bool}` — is consumed as
  the answer before the loop continues. `"auto": true` mirrors
  `sarva run --auto`.
- **A real protocol subtlety, found and documented, not glossed over:**
  `needs_confirmation` is emitted by the loop whenever a destructive call
  happens *at all* — it is not suppressed by `auto`. What changes is the
  confirm *policy* (`always_allow`, which never reads from the socket). A
  client in auto mode must treat the event as informational only and must
  NOT reply to it — there's nothing waiting to consume a reply, and
  sending one risks it being misread as the answer to a later, real
  prompt. Documented in the handler's docstring for whoever builds the
  next client.
- **The web UI now has real confirmation UI**, not just backend plumbing:
  an Approve/Deny card renders on `needs_confirmation` and blocks further
  input until answered; `tool_started`/`tool_finished` render as inline
  status lines in the assistant bubble.
- 3 new server tests (approve runs the tool, deny skips it, auto never
  blocks waiting for a reply) + 4 new UI tests (confirmation card renders
  and responds correctly, tool status lines render, card clears on
  run_done even if never answered) — 58 Python + 11 web tests, all
  passing.

**A real bug in my own first test, caught by actually running it:** my
first "auto mode" test asserted `needs_confirmation` would never be sent
at all — wrong assumption about the wire protocol, not a code bug. Writing
a test that failed for the *right* reason (a false assumption, not a
missed edge case) led directly to documenting the actual, correct
behavior above instead of shipping an incorrect mental model into the
docstring.

**Verified beyond the test suite (again, real process not just TestClient):**
started the actual `sarva serve`, confirmed `/health`, `/models`, and a
real WebSocket connection with `BUILTIN_TOOLS` wired in all still work
over an actual TCP socket (the mock provider can't self-initiate tool
calls to prove the confirmation round-trip this way — that's what the
scripted-provider pytest tests are for, and they exercise the identical
server code path via ASGI transport, not a mock of it).

**Next:** T4 — the Tauri desktop wrapper (the one-click app for
non-developers), or extending `/chat` (REST) with a "define outcome"
style async pattern if tool use is ever needed there too.

## 2026-07-21 — T4 started: Tauri desktop wrapper (step 1 of 2)

**Scope decision, stated up front:** the design doc's north-star is
"double-click an icon, no terminal" for non-developers. Fully delivering
that needs a Python runtime bundled *inside* the app (a Tauri sidecar) so
the app can start its own backend — real, separate work (cross-platform
Python packaging, code-signing, sidecar process management). Rather than
half-build that under time pressure, this entry ships **step 1 only**: a
real native window that loads the existing web UI from a `sarva serve`
backend the user starts themselves. Honestly, this is *not yet* one-click
for non-developers — it's a native shell around what already works,
with the remaining gap named precisely (not implied away) so nobody
mistakes "it runs" for "it's done."

**Built:**
- `apps/desktop/src-tauri/` — Tauri 2 (Rust) scaffold via `tauri init`,
  then hand-cleaned: package/lib renamed from generic `app`/`app_lib` to
  `sarva-desktop`/`sarva_desktop_lib`, a real identifier
  (`io.github.bpupadhyaya.sarva`, not the generated placeholder), removed
  `beforeDevCommand`/`beforeBuildCommand` (pointless here since
  `frontendDist`/`devUrl` point directly at the FastAPI server, not a
  locally-built or served asset — Tauri isn't serving anything itself in
  this architecture, just displaying it).
- `lib.rs` carries an explicit doc comment stating the step-1/step-2 split
  above, so the gap is visible in the code itself, not just this journal.
- CI gained a `desktop` job: `cargo check --locked` on every push (fast
  compile/borrow-check regression coverage). Deliberately **not** a full
  release build or cross-platform bundle — that's real infrastructure
  (multi-OS runners, code signing, `.dmg`/`.msi`/`.AppImage` artifacts)
  that belongs in step 2's own entry, not bolted on here to make this one
  look more finished than it is.

**Verified — this is the part that matters most for a desktop app:**
not just `cargo check`. Ran a real `tauri build --no-bundle`, producing an
actual 8.3MB arm64 Mach-O executable. Started a real `sarva serve` backend
and then **launched the built binary as a real OS process** — confirmed
it spawned genuine WebKit XPC helper processes (WebContent, GPU,
Networking — exactly what happens when a native macOS app creates a real
`WKWebView`), and confirmed in the backend's own access log that the
webview actually requested and received `GET /`, the JS bundle, and the
CSS bundle, all `200 OK`. That's the complete load pipeline, verified
through a genuine native app process — not a browser, not a test client.

**A real environment hiccup, handled correctly rather than worked around
carelessly:** port 8000 (the default) was already occupied by an
unrelated, pre-existing process in this environment (not something this
session started). Rather than kill an unknown process I don't own, the
verification above used a separately-confirmed-free port for the test,
then restored the committed config to the correct, standard default
(8000) afterward — the shipped config is correct; only the *verification
run* used a different port to get a clean result.

**Known gaps (the honest heart of this entry):**
- No bundled Python backend — the biggest remaining piece of the
  one-click promise. Tracked explicitly, not glossed over.
- Icons are Tauri's generated placeholders, not real Sarva branding.
- No code signing / notarization — an unsigned build will trigger
  Gatekeper warnings on macOS and SmartScreen warnings on Windows.
- CI checks compile correctness only, not that a real bundle builds on
  every platform.

**Next:** the Python sidecar (step 2 — the actual one-click unlock), or
real branding/icons, or cross-platform bundle CI. Sidecar is the one that
actually completes the mission's stated promise, so it's the natural next
priority when picked up.

## 2026-07-21 — T4 step 2: Python sidecar — the one-click unlock

The gap named at the end of step 1: bundle the Python backend itself so
launching the desktop app is the entire install, no terminal, no manual
`sarva serve`. This entry closes it.

**Built:**
- `scripts/freeze-server.sh` — PyInstaller `--onefile` freeze of the
  `sarva` CLI into a standalone executable, named per Tauri's sidecar
  convention (`sarva-server-<rust-target-triple>`) and dropped into
  `apps/desktop/src-tauri/bin/` (gitignored — a rebuilt-on-demand native
  binary, not source, so it isn't committed, unlike `core/sarva/server/static/`
  which is committed because it lets the app run with zero Node at
  install time; a frozen native binary has no equivalent "just works from
  source" fallback, so committing it would only bloat the repo with
  something CI/release should produce instead).
- Two `--add-data` flags bundle the non-Python files the backend reads at
  runtime — `core/sarva/providers/data/*.yaml` (the model registry) and
  `core/sarva/server/static/` (the web UI) — into the frozen archive at
  the same relative path `Path(__file__).parent / ...` already expects.
  PyInstaller's import analysis only follows Python imports; it does not
  discover data files a module reads at runtime, so without this the
  frozen binary starts but 500s on first real request.
- `apps/desktop/src-tauri/Cargo.toml` — added `tauri-plugin-shell`.
  `tauri.conf.json` — declared `bundle.externalBin: ["bin/sarva-server"]`.
  `capabilities/default.json` — scoped `shell:allow-execute` permission to
  exactly the `sarva-server` sidecar with a fixed `serve` arg (not a
  general shell-exec grant).
- `lib.rs` — `run()` now spawns the sidecar in `setup()`, logs its
  stdout/stderr through the app's own logger (so a startup failure is
  visible, not silently eaten), and kills it on the window's
  `CloseRequested` event.

**Real bug found and fixed while building (not just theorized):** the
first frozen-binary test (`--help` only) looked clean, but a full `serve`
run 500'd on `/models` and `/chat` with
`FileNotFoundError: .../_MEI.../sarva/providers/data/models.yaml` — exactly
the data-file risk named as a known unknown in the prior planning. Fixed
with the `--add-data` flags above; re-verified `/health`, `/models`, `/`,
and a real `/chat` round-trip all succeed from the frozen binary alone.

**Verified — the actual one-click path, not just the freeze:** ran a real
`tauri build --no-bundle`, then launched the resulting
`sarva-desktop` binary as a standalone OS process with **no `sarva serve`
running and no source repo on the loader's `sys.path`**. Confirmed via
`pgrep` that the app itself spawned `sarva-server serve` as a child
process, and confirmed over HTTP that `/health`, `/models`, and `/chat`
(a real mock completion) all responded correctly through it — the
complete one-click path, from double-click to a working chat response,
with zero manual steps.

**A real gap found, not papered over:** killing the app process directly
with `kill` (SIGTERM) — as opposed to closing its window — does **not**
run the `CloseRequested` handler, so the sidecar is orphaned and keeps
running. This was caught by testing the shutdown path explicitly (`kill
$APP_PID` then `pgrep sarva-server`), not assumed to work because the
happy path did. This matches the standard, documented caveat for Tauri's
sidecar pattern generally (window-close events don't fire on external
signals to any GUI app, not a bug specific to this code) — normal users
quitting via the window or Cmd+Q are unaffected, but a force-quit,
`pkill`, or crash leaves an orphaned backend process. Not fixed in this
entry; tracked as a known gap rather than silently shipped.

**Known gaps:**
- Orphaned sidecar on ungraceful app termination (above) — a real fix
  needs OS-level process-group or signal-handler work, not a quick patch.
- Still no code signing/notarization on the sidecar binary itself, in
  addition to the app bundle gap noted in step 1.
- `freeze-server.sh` and the sidecar wiring are verified on macOS
  arm64 only; Linux/Windows freezing and the `.exe` sidecar suffix
  convention are untested.
- The `desktop` CI job still only runs `cargo check` — it doesn't freeze
  the Python backend or build a real bundle, so this entire path has no
  CI coverage yet. A real release pipeline needs a job that runs
  `freeze-server.sh` before `tauri build`.

**Next:** fix the orphaned-sidecar gap (likely a `SIGTERM`/`SIGINT`
handler on the main process that also kills the sidecar), then real
branding/icons, then cross-platform release-bundle CI covering the full
freeze → bundle → sign pipeline on all three OSes.

## 2026-07-21 — T4 step 2 follow-up: fix the orphaned sidecar

Closed the gap named at the end of the previous entry, and found a
second, deeper bug while verifying the fix.

**Built:**
- `#[cfg(unix)]` `SIGINT`/`SIGTERM` handler (`signal-hook`, a dedicated
  OS thread blocking on `Signals::forever()`) that kills the sidecar and
  exits before the process dies from the signal. Covers force-quit,
  `pkill`, and `kill` — not just the graceful window-close path.

**Real bug found while verifying the fix (not just theorized):** after
wiring the signal handler, `kill $APP_PID` still left a `sarva-server`
process holding the port. Root cause, confirmed with
`ps -o pid,ppid,pgid`: PyInstaller's `--onefile` bootloader — the process
Tauri actually spawns and tracks as the sidecar `CommandChild` — forks a
**second** process to run the real frozen app and waits on it.
`child.kill()` only ever reaped the bootloader; the grandchild (the
actual running `uvicorn` server) was untouched and kept the port bound.
This affected **both** shutdown paths equally (window-close and the new
signal handler use the same `child.kill()` call) — it was latent in the
sidecar work shipped in the previous entry, not introduced by this one;
it only surfaced now because this entry specifically tested the
shutdown path end-to-end instead of assuming it worked. Fixed with a
`kill_sidecar()` helper that `pgrep -P`s the sidecar's own children and
kills them before killing the sidecar itself, called from both shutdown
paths.

**A red herring, run down and ruled out rather than assumed:** midway
through this fix, the sidecar appeared to stop binding its port at all,
even with the fix reverted — looked like a real regression. Root-caused
by polling with a fixed sleep instead of retrying: PyInstaller
`--onefile` re-extracts its payload to a temp directory on *every*
launch (no cache across runs), and under the machine's load at the time
(load average ~4.1) that extraction occasionally took longer than the
few seconds the earlier tests happened to wait. Confirmed by polling
with a longer timeout, which showed the exact same binary succeeding
consistently once given enough time. No code change was needed for this
part — worth recording so a future session doesn't chase the same ghost.

**Verified:** rebuilt, waited for the sidecar to bind (polling, not a
fixed sleep, after the above), confirmed `/health` responds, captured the
full process tree (bootloader + grandchild), sent `kill` to the app
process, and confirmed via `pgrep` that **no** `sarva-server` process
survives — the fix closes the gap for both the direct child and the
grandchild.

**Known gaps carried forward:**
- Windows has no equivalent signal handling yet (untested platform).
- `kill_sidecar` shells out to `pgrep`/`kill` rather than using a Rust
  process-group API — pragmatic given `tauri-plugin-shell` doesn't expose
  the underlying `std::process::Command` needed to set up a real process
  group at spawn time, but worth revisiting if that changes.
- Still no CI coverage for the freeze → sidecar → shutdown path.

**Next:** real branding/icons, then cross-platform release-bundle CI
covering the full freeze → bundle → sign pipeline on all three OSes.

## 2026-07-21 — F0: foundry track starts — a from-scratch BPE tokenizer

Every prior entry was `core/sarva`, the engine that leans on frontier
models. This one starts the other half of the mission — §3.6 of the
design of record, "no black boxes": Sarva must also carry the model-level
code, not just orchestrate someone else's model. First component: a
trainable byte-level BPE tokenizer, no HuggingFace `tokenizers`, no
`tiktoken`.

**Built:**
- `foundry/sarva_foundry/tokenizer/bpe.py` — `ByteLevelBPETokenizer`,
  implemented from first principles: a reversible byte↔Unicode-symbol
  mapping (the same trick GPT-2 uses) gives every possible byte value a
  dedicated vocabulary entry, so encoding never produces `<unk>` — any
  text, including scripts/emoji never seen during training, round-trips
  losslessly. A stdlib-`re`-only pretokenizer approximates GPT-2's regex
  (documented in the module docstring exactly where it diverges, rather
  than claimed identical). Training repeatedly merges the most frequent
  adjacent symbol pair until the requested vocab size is reached; encoding
  replays those merges in learned order. Special tokens (e.g.
  `<|endoftext|>`) are reserved ids, matched atomically before byte-level
  splitting. JSON save/load for trained tokenizers.
- `tests/foundry/test_tokenizer.py` — 10 conformance tests: round-trip on
  ASCII and on unseen Unicode/emoji, empty-input edge case, vocab-size
  budget respected, invalid vocab size rejected, merges actually compress
  a training sentence, training is deterministic (same corpus → identical
  merges/vocab), special tokens stay atomic and round-trip, save/load
  round-trip. All passing.
- `examples/02_train_a_tokenizer.py` — trains on a four-sentence toy
  corpus and prints both the compression (`"the quick brown fox"`: 19
  byte-level tokens → 4 trained tokens) and a round-trip proof on
  `"héllo wörld —日本語 🎉🚀"`, text the tokenizer never saw.
- `docs/foundry/tokenizer.md` — the matching docs chapter (design
  principle: every module gets one), covering why byte-level, how
  training works, and how to run the example. Wired into `mkdocs.yml`'s
  nav (validated the YAML parses correctly; `mkdocs` itself isn't a
  project dependency yet, so the actual site build is still unverified —
  named honestly rather than assumed to work).

**Real bug found and fixed while building (not just theorized):** the
first draft of the merge step rebuilt the word-frequency table with a
dict comprehension (`{merge(word): freq for word, freq in ...}`), which
silently drops frequency mass whenever two distinct pre-merge words
collide into the same tuple after a merge — the last one wins, the rest
vanish, and training silently learns a slightly wrong distribution with
no error or test failure to catch it. Fixed by accumulating into a
`Counter` with `+=` before any test ran against it, so it never shipped.

**Known gaps:**
- Tokenizer only — no model architecture, pretraining loop, or anything
  else from §3.6 yet. This is the first component of a large track.
- `mkdocs` isn't installed/pinned as a project dependency, so the docs
  site itself has never actually been built, only the YAML validated.
- No data-pipeline code yet — training above used an inline toy corpus,
  not the corpus-sourcing/cleaning/dedup pipeline §3.6(c) describes.

**Next:** the from-scratch transformer architecture (attention, RoPE,
RMSNorm, SwiGLU, GQA — the teaching-baseline dense decoder from §3.6a),
or continue rounding out desktop (branding, release CI). Foundry is the
harder, more novel work and was the natural next pick this iteration;
either track can lead next.

## 2026-07-21 — F0 continued: the from-scratch transformer

The teaching-baseline dense decoder from §3.6a: attention, RoPE, RMSNorm,
SwiGLU, GQA — the architecture every current LLaMA/Qwen/Mistral-class
model is a variation of, implemented directly from the math rather than
imported from `transformers`.

**Built:**
- `foundry/sarva_foundry/model/layers.py` — `RMSNorm` (root-mean-square
  norm, float32-upcast for stability); `precompute_rope`/`apply_rope`
  (rotary position embeddings, rotate-half convention); `SwiGLU` (gated
  feedforward) with `default_swiglu_hidden_dim` matching LLaMA's
  parameter-matched sizing convention.
- `foundry/sarva_foundry/model/attention.py` — `GroupedQueryAttention`:
  query heads split into groups sharing one KV head each (`repeat_kv`
  expands the shared KV heads to match), RoPE applied to q/k before
  attention, causal masking enforced unconditionally — no non-causal mode
  exists to accidentally select. The module docstring states explicitly
  where "from scratch" stops: `nn.Linear`/`nn.Embedding` and PyTorch's
  fused `scaled_dot_product_attention` kernel are commodity substrate
  (same tier as `torch.matmul`), not model logic.
- `foundry/sarva_foundry/model/transformer.py` — `TransformerBlock`
  (pre-norm residual composition) and `DecoderOnlyTransformer` (tied
  embedding/unembedding, token ids in → logits out).
- `tests/foundry/test_model.py` — 13 conformance tests, two of which are
  the actual point of this entry: `test_rope_encodes_relative_not_absolute_position`
  verifies RoPE's defining mathematical property directly (rotated q·k
  depends only on relative offset) rather than trusting a correct-looking
  implementation, and `test_causal_masking_prevents_attending_to_future_tokens`
  perturbs only the last token in a sequence and asserts every earlier
  position's output is bit-for-bit unchanged — the only test that can
  actually catch causal masking silently not masking, since a broken
  causal flag still produces plausible, right-shaped logits. Also: a
  full end-to-end trainability test (loss decreases over 50 optimizer
  steps on a toy task) that exercises gradient flow through every layer
  at once.
- `examples/03_train_toy_transformer.py` — wires the tokenizer (previous
  entry) into the transformer: trains on real token ids, 200 CPU steps,
  greedy-decodes a continuation.
- `docs/foundry/transformer.md` — the matching docs chapter, including
  both bugs below as worked examples of why shape-correct isn't the same
  as correct.

**Two real bugs found and fixed while building (not just theorized):**
1. The tokenizer's merge step (carried over from the previous entry) was
   already fixed; this entry's own bug: `precompute_rope`'s cos/sin
   tables are finite (bounded by `max_seq_len`), and `GroupedQueryAttention.forward`
   sliced them with no bounds check. Found by actually running the
   generation loop in example 03 — not by any unit test, since every test
   used a fixed sequence length — which grows the sequence past
   `max_seq_len` one token at a time. Slicing past a tensor's length
   doesn't raise in Python, it just returns something shorter, so the
   real failure surfaced several calls later as a confusing
   shape-mismatch deep inside `apply_rope` instead of at the actual
   misconfiguration. Fixed with an explicit, immediate bounds check at
   the top of `forward()`, and pinned with
   `test_forward_raises_a_clear_error_past_max_seq_len`.

**Known gaps:**
- Teaching baseline only — no MoE routing, long-context scaling, or
  native multimodal input yet (§3.6a's "frontier-class" extensions).
- No pretraining data pipeline (§3.6c) — training so far uses inline toy
  corpora, not real corpus sourcing/cleaning/dedup.
- No checkpointing/resume or distributed training (§3.6d) — everything
  verified so far is single-process CPU, seconds-scale.

**Next:** either the pretraining data pipeline + a real (checkpointed)
training loop, or continue rounding out desktop (branding, release CI).

## 2026-07-21 — F0 continued: dataset chunking + checkpoint/resume training loop

The last piece needed before the foundry track has a genuinely runnable
(if toy-scale) pretraining pipeline: corpus → batches (§3.6c, the
chunking mechanism) and a training loop that can actually survive being
interrupted (§3.6d).

**Built:**
- `foundry/sarva_foundry/data/dataset.py` — `tokenize_corpus` (encodes a
  corpus and concatenates it with `<|endoftext|>` document separators, so
  the model learns document boundaries instead of treating unrelated
  documents as one continuous stream) and `TextChunkDataset` (fixed-length
  `(input, target)` chunks, target shifted right by one — standard
  next-token-prediction framing; the trailing leftover tokens that don't
  fill a whole chunk are dropped, not padded, and that behavior is
  tested, not just assumed).
- `foundry/sarva_foundry/train/trainer.py` — `Trainer`: a training step,
  gradient clipping, and `save_checkpoint`/`load_checkpoint` that persist
  **optimizer state** (AdamW's per-parameter momentum/variance), not just
  model weights — the module docstring states directly why this matters:
  a checkpoint that only restores weights silently restarts momentum
  from zero, training differently from the run it claims to resume, with
  no exception to catch the difference.
- `tests/foundry/test_dataset.py` (6 tests) and `tests/foundry/test_trainer.py`
  (3 tests) — the trainer tests are the ones that matter most here:
  `test_checkpoint_resume_is_bit_identical_to_uninterrupted_training`
  proves resume actually resumes (10 uninterrupted steps vs. 5 steps →
  checkpoint → fresh `Trainer` loaded from disk → 5 more steps produce
  identical final weights), paired with a **negative control**,
  `test_checkpoint_without_optimizer_state_would_diverge`, that
  deliberately reintroduces the bug the module warns about (swaps in a
  fresh optimizer post-load) and asserts the result *does* diverge —
  without this control, the positive test wouldn't prove much, since the
  toy task could coincidentally converge to the same point regardless of
  optimizer state.
- `examples/04_pretrain_and_resume.py` — the full pipeline built across
  three entries, run together: tokenizer → dataset → transformer →
  trainer, 30 steps, checkpoint, a *fresh* model/trainer resuming for 30
  more steps. Loss descends smoothly across the checkpoint boundary
  instead of spiking — the visible proof, not just the test's numeric
  assertion.
- `docs/foundry/training.md` — the matching docs chapter, including the
  positive/negative test pairing as a worked example of why a passing
  checkpoint test alone doesn't prove correctness.

**A real bug introduced and caught by this entry's own verification
step, not shipped:** fixing a `ruff` B008 lint warning (mutable/call
default argument — `TrainerConfig()` as a literal default value) by
switching to `config: TrainerConfig | None = None` left the constructor
body still reading the old parameter name (`config.lr`) instead of
`self.config.lr`, which is `None` post-refactor — an `AttributeError` on
every `Trainer()` call. The lint fix looked complete (ruff was clean,
the diff looked like a mechanical rename); the bug was invisible to
`ruff check` and would have been invisible to a review that didn't
re-run the tests after the "trivial" fix. Caught immediately because
this session runs the full test suite after every change without
exception, not just after the change that looks risky.

**Known gaps:**
- No real corpus sourcing (web/code/books/math crawling, cleaning,
  dedup, quality filtering) — `tokenize_corpus` is the chunking mechanism
  §3.6c needs, not the sourcing pipeline.
- No distributed training (FSDP/3D parallelism) or loss-spike handling —
  everything verified is single-process CPU, seconds-scale.
- No learning-rate schedule (warmup/decay) — `Trainer` uses a flat LR.

**Next:** real branding/icons + cross-platform release CI for the
desktop app (still open from T4), or scaling the foundry pipeline up
from toy-corpus to a real small dataset with an actual LR schedule.

## 2026-07-22 — CI: cross-platform matrix, and a real CI-only regression found and fixed

Extended the `desktop` job to a `[macos-latest, ubuntu-latest,
windows-latest]` matrix — the T4 sidecar work had only ever been verified
on macOS arm64, and a `cargo check`-level regression on Linux/Windows had
no way to surface before this. This entry also caught and fixed a real
bug that had been silently breaking CI for two prior commits.

**Built:**
- CI matrix for the `desktop` job across all three target OSes, with
  Tauri's documented Linux system-package prerequisites
  (webkit2gtk/appindicator/etc.) installed on `ubuntu-latest` first.

**A real bug found immediately after pushing — by actually watching CI,
not by assuming a green local run meant CI was fine too:** `gh run list`
showed the *previous two* commits' CI runs had failed on the `desktop`
job — going back to the T4 step-2 sidecar commit. Root cause:
`tauri-build`'s build script validates that every `bundle.externalBin`
path exists on disk, and fails the **entire compile** — `cargo check`
included, not just a real `tauri build` — if it doesn't. The sidecar
binary (`scripts/freeze-server.sh`'s output) is correctly gitignored as a
large, per-platform artifact this repo deliberately doesn't commit, which
means CI has never had one on disk since `bundle.externalBin` was added,
and the `desktop` job has been failing on every single push since —
invisible because nothing in this session's workflow had checked `gh run
list` after those two prior pushes, only local `cargo check`, which
always had the real binary present locally.

**Fixed** with a CI step that creates an empty placeholder file at the
exact target-triple path Tauri's build script checks for
(`bin/sarva-server-<target-triple>[.exe]`, computed via `rustc -vV`),
before `cargo check` runs. This is proportionate to what the job actually
checks (compile correctness, per its own existing comment — never meant
to verify the sidecar itself, which is verified locally and recorded
earlier in this journal) rather than trying to run a full PyInstaller
freeze inside a job whose entire value proposition is being fast and
cheap. Verified the fix two ways before trusting it: (1) reproduced the
exact failure locally by moving the real sidecar binary aside and
re-running `cargo check`, confirming the identical `resource path ...
doesn't exist` error; (2) applied the same placeholder-file logic
locally, confirmed `cargo check` then passed, restored the real binary,
confirmed it *still* passed. Only then pushed, and watched the real CI
run (`gh run watch`) to completion — genuinely green across `core`,
`web`, and all three `desktop` OS variants, not inferred from the fix
"looking right."

**The lesson, stated plainly because it's worth remembering beyond this
one bug:** this session's discipline of running local tests/lint before
every commit is necessary but was not sufficient — it caught every
Python-side regression this session but had no way to catch a
CI-environment-specific failure (missing file on a fresh checkout) that
only manifests where the working tree doesn't already have local,
gitignored build artifacts sitting around. `gh run list`/`gh run watch`
after a push that touches CI-relevant files (or any push, periodically)
is now part of how this loop verifies "pushed" actually means "working,"
not just "compiled locally."

**Known gaps:**
- Still no real cross-platform **bundle** CI (`.dmg`/`.msi`/`.AppImage`)
  or code signing — this entry only closes the compile-check gap.
- Real app branding/icons still outstanding (Tauri's generated
  placeholders, per T4 step 1's entry).

**Next:** real branding/icons, or scaling the foundry pipeline up from
toy-corpus to a real small dataset with an actual LR schedule.

## 2026-07-22 — Core: url-sourced media blocks actually work now

A gap in the type system's own stated design, not a new feature: since
T0, `_MediaBlock.resolve_bytes()`'s docstring has said "url sources must
be fetched via `sarva.multimodal.fetch` (not implemented here)" — that
module never existed, so any `ImageBlock`/`AudioBlock`/etc. constructed
with a `url` source (as opposed to `data` or `path`) was unusable
end-to-end. Closed that gap.

**Built:**
- `core/sarva/multimodal/fetch.py` — `fetch_bytes(url)`: async, streams
  the response rather than trusting `Content-Length` (a misbehaving or
  malicious server can omit or lie about it), enforcing `max_bytes` from
  actual bytes counted while streaming, and restricts schemes to
  `http`/`https` (rejects `file://`, `ftp://`, etc. — this resolves URLs
  that arrive as declared media sources, so scheme hygiene matters even
  though there's no untrusted-user-input path to it yet). Accepts an
  optional `client: httpx.AsyncClient` so production call sites can share
  one client and tests can inject an `httpx.MockTransport` — no real
  network I/O anywhere in this entry's test suite.
  `resolve_media_bytes(block)` is the dispatcher: `data`/`path` sources
  resolve exactly as the existing sync `resolve_bytes()` already does,
  `url` sources go through `fetch_bytes`. Deliberately kept **out** of
  `content.py` itself — that module is the dependency-light type
  vocabulary every layer imports, and pulling `httpx` into it would
  couple the universal content model to a network library it has no
  other reason to need.
- Wired into `sarva.providers.anthropic_provider`: `_to_anthropic_message`
  is now `async def` and awaits `resolve_media_bytes` instead of calling
  the block's own `resolve_bytes()` directly, so an `ImageBlock` with a
  `url` source now actually reaches the Anthropic API instead of raising
  at request-build time. The one caller (`generate()`) already ran inside
  an async context, so this required no wrapper/anti-pattern — genuinely
  awaiting network I/O where the code was already async throughout.
- `tests/conformance/test_fetch.py` (7 tests) — response-body round-trip,
  scheme rejection, HTTP-error-status handling, the streamed size-cap
  (verified against a handler that doesn't even set `Content-Length`, so
  the cap can't be passing by accident via header-trusting), and all
  three `resolve_media_bytes` dispatch paths (data/path/url), all via
  `httpx.MockTransport` — no real network call anywhere in this suite.
- Updated `tests/conformance/test_anthropic_provider.py` for the new
  `async def` signature (its own docstring already called out that these
  tests use only in-memory `data` sources, so no I/O actually runs here
  either — the `await` exists because the function's shape changed, not
  because these particular tests exercise the network path).

**Known gaps:**
- No degrader implementations still ship (per the earlier codebase
  survey that identified this gap) — `fetch.py` makes url-sourced bytes
  loadable, it doesn't add image resizing/reformatting or audio
  transcription. That's the next natural piece if this area gets picked
  up again.
- `fetch_bytes` has no retry/backoff — a transient network blip surfaces
  as a `FetchError` immediately rather than retrying.

**Next:** a concrete image degrader (resize/reformat via Pillow for
provider context limits) to give the degradation registry its first real
converter, or continue elsewhere (branding, foundry scale-up).

## 2026-07-22 — Core: the degradation registry's first real converter

`sarva.multimodal.content.Degrader` has been a proven, tested framework
since T0 (`degrade_message`, recursive dispatch, depth-capped, never
silently drops content) — but zero concrete implementations shipped
anywhere until this entry, confirmed by grepping the whole `core/`
tree: `Degrader`/`degrade_message` were referenced only inside
`content.py` itself and its own tests (which use a fake `_EchoDegrader`).
Closed that gap with the first real one.

**Built:**
- `core/sarva/multimodal/degraders/image.py` — `ImageToTextDegrader`:
  turns an `ImageBlock` a text-only model can't consume into a
  `TextBlock`. Deliberately does **not** attempt to describe the image's
  actual visual content — that would require a vision-capable model call,
  which is a decision for the router/agent loop to make explicitly (route
  to a vision model, or don't), not something that should happen as an
  implicit side effect buried inside content-degradation plumbing.
  Instead it reports only objectively verifiable metadata decoded
  directly from the bytes (dimensions, format, size) via Pillow — new
  dependency, added to `core/pyproject.toml`, used here purely as a
  commodity image-decoding library (same tier as `httpx` for network
  I/O), not model logic. This keeps "content is never silently dropped"
  honest in the specific way that matters: the target model learns an
  image was present and what it technically was, with nothing fabricated
  about what it contains.
  Uses `resolve_media_bytes` (last entry's `sarva.multimodal.fetch`), so
  it handles url-sourced images too, not just data/path — the first real
  caller of that module.
- `tests/conformance/test_degraders.py` (6 tests) — correct
  dimensions/format extraction, correct byte-size reporting, a test that
  directly pins the "no fabrication" design principle (asserts the
  honesty disclaimer is present, not just that *some* text came out), a
  corrupt-bytes case that must raise clearly rather than degrade into
  something wrong-but-plausible, the path-source dispatch path, and —
  the one that matters most — an end-to-end test through the *real*
  `degrade_message` recursive dispatcher (not just calling `.degrade()`
  directly), proving the concrete implementation actually satisfies the
  `Degrader` protocol and works through the framework, not just in
  isolation.

**Known gaps:**
- Still the only concrete degrader — audio, video, and document have no
  converters yet.
- Not wired into the agent loop's model-selection fallback path. The
  loop's own docstring already states this scope boundary explicitly
  ("T2 wires *routing*, not yet *degradation*") — today, `router.pick()`
  requires a model that already supports every modality present and
  raises if none exists; teaching the loop to fall back to the
  best-available model plus degradation is a real, separate design
  decision (when to prefer "wait/fail" vs. "silently degrade and
  continue") deliberately left out of this entry rather than folded in
  as a side effect.
- No image resizing/reformatting for provider size/dimension limits
  (the original framing for this entry) — decoding+reporting metadata
  turned out to be the correctly-scoped first piece; resizing is a
  reasonable next one.

**Next:** wire `ImageToTextDegrader` into the agent loop's fallback path
(the real remaining design decision named above), or continue elsewhere
(branding, foundry scale-up, audio/video degraders).

## 2026-07-22 — Core: degradation wired into the agent loop as an opt-in fallback

The design decision the previous entry deliberately deferred: when should
the loop prefer degrading content over failing outright? Answered as
**opt-in, not automatic** — a caller who doesn't ask for it gets exactly
today's behavior; a caller who supplies `degraders` gets a real fallback
attempt before failing.

Before touching `core/sarva/agent/loop.py` (part of the FROZEN spec-03),
re-read `sarva-specs/spec-03-agent-loop.md`: what's frozen is the state
machine, event vocabulary, budget model, and tool contract — the loop's
own module docstring already documents T2 extending routing behavior
beyond the spec's literal code via new optional `run()`/`__init__()`
parameters (`extra_content`, `transcript_out`, both prior entries). This
change follows that exact established pattern — a new optional
constructor parameter, zero change to any state transition, event shape,
or budget check — rather than treating it as a spec change requiring
escalation.

**Built:**
- `AgentLoop.__init__` gained `degraders: dict[Modality, Degrader] | None
  = None`. Empty/absent (the default) is byte-for-byte the old behavior —
  confirmed by the pre-existing `test_image_content_with_no_vision_capable_model_fails_cleanly`
  passing completely unchanged.
- The `LookupError` handler in `run()` (previously: fail immediately) now
  tries a fallback when `degraders` is non-empty: pick the best available
  model needing only `Modality.TEXT` (guaranteed to exist in any real
  configuration, since the mock provider is always available), degrade
  every message down to what that model actually supports via the real
  `degrade_message` dispatcher, and proceed with that model. Any failure
  in the fallback itself (no text-capable model either, or a degrader
  registered but not for the modality actually present) falls through to
  the original `FAILED` state — the exact behavior from before this
  entry, not a new failure mode.
- Confirmed `router.pick()`'s `override` parameter always bypasses the
  modality check entirely and never raises `LookupError` — meaning
  reaching this fallback path at all is only possible when the caller
  passed no explicit `model_override`, so there's no scenario where this
  fallback could silently contradict an explicit model choice.
- 5 new tests in `tests/conformance/test_agent.py`: the fallback actually
  succeeding (verified by echoing the *degraded* text back through
  echo-mode `MockProvider` and asserting the degrader's own metadata
  string appears in the response — not just that the run ended `DONE`
  for some unrelated reason); a non-empty `degraders` dict that doesn't
  cover the modality actually present still failing cleanly (proves the
  fallback checks coverage, not just dict-truthiness); a regression guard
  that the fallback never triggers when a directly vision-capable model
  is already available (the registry's own `mock` entry supports images);
  and the degenerate double-failure case (no models available at all)
  still terminating cleanly in `FAILED` rather than raising out of the
  generator.

**Known gaps:**
- Still only image degradation exists — audio/video/document content
  with no covering degrader still fails outright, same as before.
- No signal is surfaced to the caller/UI that a run actually degraded
  (vs. routed to a fully-capable model normally) beyond inspecting which
  `model.id` ended up in the transcript — deliberately left out to avoid
  overloading the shared `StateChangedEvent.detail` field's semantics in
  the same change; a dedicated signal is reasonable follow-up work.

**Next:** real desktop branding/icons, continued foundry scale-up, or an
audio/video degrader now that the loop actually knows what to do with one.

## 2026-07-22 — Desktop: real app branding, replacing Tauri's placeholders

Closes the branding gap named honestly since T4 step 1's very first
entry ("Icons are Tauri's generated placeholders, not real Sarva
branding") and repeated as a known gap in every desktop entry since.

**Built:**
- `scripts/generate-icon.py` — generates the 1024x1024 source icon with
  pure Pillow shape-drawing (no font/system dependency, so it's
  reproducible on any platform with the project's own dependencies
  installed): a solid off-white circle — *sarva* (सर्व) meaning "all /
  whole" — centered on a solid indigo rounded square. Deliberately the
  simplest possible design: one shape, one contrast, nothing that gets
  lost at 16x16. Framed honestly in the script's own docstring as a
  first real, deliberate mark, not professional final branding.
- Ran Tauri's own `tauri icon` CLI against that source to regenerate the
  entire platform icon set (32x32 through the Windows Store tile sizes,
  `.icns`, `.ico`) — the officially supported path, far more reliable
  than hand-building multi-resolution container formats. Its default
  output also included iOS/Android asset sets; removed those since
  neither platform is in scope yet (design doc: mobile is explicitly
  "later phase," not v1) — regenerable from the same source icon when
  that phase actually starts, not needed as speculative scope now.

**Verified, not just generated:** `cargo check` still passes, and — the
part that actually proves the icon is wired in, since `--no-bundle`
skips macOS's bundling step entirely — ran a real `tauri build` (with
bundling) and confirmed `icon.icns` is genuinely embedded in the
resulting `Sarva.app/Contents/Resources/` and referenced correctly by
`Info.plist`'s `CFBundleIconFile`, not just sitting in the source tree
unused. Visually checked the icon at both 128x128 and 32x32 to confirm
it stays legible at the sizes that actually matter (Dock/taskbar,
window title bar) before treating it as done.

**Known gaps:**
- A simple geometric mark, not professional graphic design — a real
  brand identity (typography, color system, app-store assets) is
  future work if/when the project wants one.
- No app-store screenshots/marketing assets — out of scope for this
  entry, which only closes the "the icon itself is a placeholder" gap.

**Next:** continued foundry scale-up, an audio/video degrader, or
cross-platform release-bundle CI (still the one T4 gap this session
hasn't touched: `.dmg`/`.msi`/`.AppImage` artifacts + code signing).

## 2026-07-22 — Core: an audio degrader, and closing the "actually reachable" gap

Two pieces. The second turned out to matter more than the first.

**Built:**
- `core/sarva/multimodal/degraders/audio.py` — `AudioToTextDegrader`,
  the second concrete `Degrader`. Same honesty principle as
  `ImageToTextDegrader` (report only what's verifiably known, never
  fabricate content), but a **deliberately different failure-handling
  tradeoff**, documented directly in the module: Pillow reliably decodes
  nearly every real-world image format, so the image degrader treats
  undecodable bytes as a genuine error. Real-world audio is
  overwhelmingly compressed (MP3/AAC/OGG/M4A) — stdlib `wave` only
  parses uncompressed WAV, and pulling in ffmpeg/pydub isn't justified
  for a metadata-only converter — so "not WAV" is the *expected* case
  here, not an error: it falls back to whatever the block already
  declares (`media_type`, `duration_s` if set, and the always-knowable
  byte size) instead of raising.
- `sarva.multimodal.degraders.default_degraders()` — the shared
  `{IMAGE: ImageToTextDegrader(), AUDIO: AudioToTextDegrader()}` set
  every skin now wires in, so "what does Sarva degrade out of the box"
  lives in exactly one place.
- **The gap that actually mattered:** grepped every `AgentLoop(...)`
  construction site — `cli.py`'s `chat`/`run` commands, `app.py`'s
  `/chat` and `/ws/chat` — and found **none of the four** passed
  `degraders=`. Last entry's opt-in fallback was fully built, fully
  tested, and completely unreachable by any real user; only custom code
  calling `AgentLoop` directly could ever have used it. Wired
  `degraders=default_degraders()` into all four.
- 13 tests: the audio degrader's real-WAV-decode path (proves it
  actually reads bytes, not just declared metadata, for the one format
  it can), the undecodable-format fallback, the "nothing knowable at
  all" case, the no-fabrication principle, an end-to-end test through
  the real `degrade_message` dispatcher, and coverage for
  `default_degraders()` itself (correct modality set, correct types,
  no shared-mutable-dict surprise across callers).

**Honest note on what "wired in" currently means in practice:** the
fallback only ever *triggers* when the router can't find a model
supporting every modality present — and today's default registry
(`models.yaml`) gives the always-available `mock` provider full
`[text, image, document]` support, so with zero configuration the
fallback path is wired correctly but practically dormant; there's
always a directly-capable model. It becomes live in any deployment
whose actually-available models don't all cover every modality (e.g.
only a text-only local model, or a future registry entry that's
narrower) — confirmed correct by the loop-level tests using a
purpose-built text-only router, not glossed over as "done" just because
the plumbing compiles.

**Known gaps:**
- No video/document degraders yet.
- No signal surfaced to callers that a request path actually is
  running with degradation live vs. dormant (same known gap named in
  the wiring entry, still unaddressed).

**Next:** continued foundry scale-up, cross-platform release-bundle CI,
or a video degrader (frame-sampling + the now-existing image degrader
composed together, per §3.3's stated video->frames+text path).

## 2026-07-22 — Cross-platform release-bundle CI: real installers, all three OSes

The T4 gap named in nearly every desktop entry since it started: `cargo
check` proved the Rust compiles everywhere, but nothing had ever produced
an actual installable artifact on Linux or Windows — only ever a real
macOS `.app`/`.dmg`, built and verified by hand.

**Built:**
- `.github/workflows/release-bundle.yml` — manual-trigger
  (`workflow_dispatch`) workflow, matrixed across macOS/Linux/Windows:
  freeze the Python sidecar (`scripts/freeze-server.sh`), run a real
  `tauri build` (not `--no-bundle`), upload whatever installer format
  each OS produces as a build artifact. Deliberately not on every push —
  a full PyInstaller freeze + real bundle per OS is genuinely slow,
  meaningful only when actually cutting a release. Unsigned by design (no
  signing certificates exist yet, a separate tracked gap); an unsigned
  build a maintainer can download and run is real progress over no
  release pipeline at all.

**Three real, previously-undiscovered cross-platform bugs found and
fixed in `scripts/freeze-server.sh` — each one only surfaced by actually
running a Windows GitHub Actions job, not by local reasoning alone (this
script had only ever executed on macOS since it was written):**
1. uv venvs use `.venv/Scripts` on Windows, not `.venv/bin`, and every
   executable in it (including PyInstaller's own frozen output) gains a
   `.exe` suffix; PyInstaller's `--add-data` separator is also
   platform-dependent (`os.pathsep`: `:` on POSIX, `;` on Windows).
2. The `sarva` console-script entry point `uv sync` installs is a plain
   readable `.py` file with a shebang on macOS/Linux — PyInstaller can
   analyze that directly — but a *compiled* `.exe` launcher stub on
   Windows, which isn't an analyzable script at all
   (`Script file '...\sarva.exe' does not exist`). Fixed by freezing a
   new, tiny, repo-owned wrapper (`scripts/_freeze_entrypoint.py`) that's
   a real `.py` file on every platform, instead of the installed,
   platform-varying launcher.
3. Git Bash's (MSYS2) automatic POSIX↔Windows path conversion turned out
   to be actively harmful either way it was set: left enabled, it mangled
   `--add-data`'s semicolon-joined `SRC;DEST` value into garbage
   (`D:/a/sarva/...` → `\\d\\a\\sarva\\...`); disabled outright
   (`MSYS_NO_PATHCONV=1`, the first fix attempted), plain single-path
   arguments like the script path stopped being converted at all, so
   PyInstaller — a native Windows program with no idea what MSYS's
   internal `/d/a/...` paths mean — reported them as not existing either.
   Fixed by not relying on MSYS's heuristic at all: resolve every path
   PyInstaller receives to native Windows form explicitly via `cygpath
   -m` (a no-op passthrough on macOS/Linux, where the command doesn't
   exist).

**Verified, iteratively, against real CI — not fixed once and assumed
correct:** each of the three fixes above was diagnosed from an actual
failed Windows Actions run's log, fixed, re-verified on macOS locally
(confirming the fix didn't regress the platform that already worked),
pushed, and re-triggered via `gh workflow run` + `gh run watch` until the
Windows job genuinely passed. One of those verification passes also
caught a false alarm worth recording rather than mis-diagnosing: a
`--help` invocation that appeared to hang for several seconds during
local re-testing turned out to be the same PyInstaller onefile
re-extraction latency under system load already documented earlier in
this journal — waited it out and confirmed correct output instead of
"fixing" a nonexistent regression. Final result, confirmed by inspecting
the actual uploaded artifacts (not just green checkmarks): all three OSes
produced real, substantial bundle artifacts in one workflow run —
`sarva-macos-latest` (65MB), `sarva-windows-latest` (80MB),
`sarva-ubuntu-latest` (478MB, larger because Linux's bundle target
includes both `.AppImage` and `.deb`).

**Known gaps:**
- No code signing/notarization — artifacts trigger Gatekeeper/SmartScreen
  warnings, expected and documented, not silently glossed over.
- Manual trigger only, not wired to git tags/releases yet — that's the
  natural next step once the project actually wants to cut a v0.1.0.

**Next:** continued foundry scale-up, a video degrader, or wiring
release-bundle.yml to version tags for real automated releases.

## 2026-07-22 — F0 continued: a real learning-rate schedule

`Trainer` used a flat LR — named honestly as a known gap in the entry
that shipped it. Closed it with the standard shape essentially every
real pretraining run uses: linear warmup, then cosine decay.

**Built:**
- `foundry/sarva_foundry/train/schedule.py` — `WarmupCosineSchedule`, a
  pure function of step count (`lr_at(step)`), not mutable schedule
  state. That design choice is the point: `Trainer.train_step` calls it
  fresh on every step, so the *existing* checkpoint/resume machinery —
  which already restores `self.step` — resumes the LR curve correctly
  for free. There's no separate schedule state that could drift out of
  sync with the checkpointed step count, because there's no separate
  state at all.
- `TrainerConfig` gained an optional `schedule` field (default `None` =
  the original flat-LR behavior, unchanged) and `train_step` now sets
  `optimizer.param_groups[...]["lr"]` from the schedule before each step
  when one is configured.
- `examples/04_pretrain_and_resume.py` now trains with a schedule and
  prints the LR alongside loss — visibly ramping through warmup, then
  decaying smoothly *through* the checkpoint boundary rather than
  resetting, and loss converges noticeably faster than the flat-LR
  version from the prior entry (reaches ~0.27 by step 51 here; the
  flat-LR run took till step 199 to reach near-zero on a similar toy
  task).
- 12 new tests: `test_schedule.py` (10) covering the warmup ramp, cosine
  decay shape, the post-`total_steps` floor (a run that overshoots its
  planned length must degrade to `min_lr`, not have cosine's periodicity
  ramp back up), and input validation; `test_trainer.py` (2) covering
  that `train_step` actually pulls a fresh LR every call (not just once
  at construction) and — the one that matters most —
  `test_checkpoint_resume_is_bit_identical_with_a_schedule_active`,
  proving resume continues the LR curve exactly rather than restarting
  warmup or jumping to some other point on it.

**A test bug found and fixed by the test suite itself, not shipped:**
the first draft of `test_lr_never_exceeds_peak_or_drops_below_min`
asserted `min_lr` bounds the *entire* schedule, including warmup — it
failed immediately (LR of 0.1 when `min_lr=0.2`, during warmup). The
implementation was correct; the test's assumption wasn't: `min_lr` is a
floor for the post-warmup decay phase, not the whole curve — the
standard convention (matching NanoGPT/Megatron-style schedules)
deliberately ramps warmup from near-zero. Fixed by splitting the
assertion into what's actually guaranteed during warmup (no negative
LR, never exceeds peak) versus after it (bounded by `min_lr`/`peak_lr`
both ways) — a real example of a failing test correctly catching a wrong
assumption in the test itself, not a bug in the code under test.

**Known gaps:**
- No other schedule shapes (linear decay, constant-with-warmup) — only
  warmup+cosine, the most common default.
- Still no real corpus sourcing or distributed training (§3.6c/d) —
  unchanged from prior entries.

**Next:** continued foundry scale-up (real corpus sourcing), a video
degrader, or wiring release-bundle.yml to version tags.

## 2026-07-22 — F0 continued: real corpus sourcing (load, dedup, filter)

Every training run so far used an inline Python list of toy sentences —
honest as a proof-of-concept, but not the sourcing/cleaning/dedup slice
of §3.6c the design of record actually calls for. This entry closes the
first real piece of that gap, at the scale this project can run and
test today: a local directory of text files, not Common Crawl.

**Built:**
- `foundry/sarva_foundry/data/corpus.py` — three composable stages:
  `load_text_files` (reads a directory's files as one document each,
  sorted for deterministic ordering, **raises** rather than silently
  skipping a file it can't decode — a bad file should be a loud, fixable
  problem, not quietly missing data no one notices until the model
  trained on it behaves strangely); `dedup_documents` (exact-duplicate
  removal by content hash, first-occurrence order preserved —
  near-duplicate detection via minhash/simhash is real, separate scope,
  named rather than silently assumed covered); `filter_by_length` (drops
  documents outside a `[min_chars, max_chars]` range — the crudest real
  quality filter, and the one every larger pipeline layers richer
  heuristics on top of, not a replacement for them).
- 11 tests in `tests/foundry/test_corpus.py`, including one that proves
  the three new stages compose into the *existing* tokenize/chunk
  pipeline as a real end-to-end flow — two files that are exact
  duplicates of each other collapse to one document, a too-short file
  gets filtered before it ever reaches the tokenizer, and what survives
  successfully trains a tokenizer and produces a real `TextChunkDataset`
  — not three functions that merely happen to share a module.
- `docs/foundry/training.md` — a new "Sourcing" section ahead of the
  existing chunking section, and the "What's next" list updated to stop
  claiming corpus sourcing doesn't exist at all (it does now, at local
  scale — provenance/license tracking and web/code/books/math-scale
  sourcing still don't).

**Known gaps:**
- Still local-files-only — no web/code/books/math crawling, no
  provenance or license tracking, no mixing recipes across sources.
- Near-duplicate detection (minhash/simhash) not implemented — only
  exact-match dedup.
- No distributed training (§3.6d) — unchanged from prior entries.

**Next:** a video degrader, wiring release-bundle.yml to version tags,
or continuing to deepen the foundry pipeline (near-duplicate dedup,
provenance tracking, or scaling the toy examples up to a real small
public-domain corpus).

## 2026-07-22 — Core: `VideoToTextDegrader`, completing the degrader trio

The third and, for now, final concrete `Degrader` — image, audio, video
all now have real converters, and all three are wired into every real
`AgentLoop` call site via `default_degraders()`.

**Built:**
- `core/sarva/multimodal/degraders/video.py` — `VideoToTextDegrader`.
  Same honesty principle as the other two (report only what's verifiably
  known, never fabricate content), but simpler than audio's: there's no
  standard-library module that can decode *any* real-world video
  container at all (unlike audio, where `wave` genuinely handles the one
  common uncompressed case), so this degrader never attempts byte-level
  decoding — it always reports the block's declared `media_type`,
  `duration_s` if set, and the always-knowable byte size.
- **Named, not silently skipped:** `Degrader`'s own docstring in
  `content.py` uses "video -> [image frames + text transcript]" as its
  motivating example — this degrader deliberately does *not* do that.
  Real frame sampling into `ImageBlock`s needs a video-decoding
  dependency (ffmpeg/opencv) this project doesn't carry yet, and adding
  one wasn't justified for a metadata-only converter. Documented
  directly in the module as real, deferred follow-up work, not quietly
  declared "the video degrader" and left at that.
- `default_degraders()` now includes `Modality.VIDEO`, so all four real
  `AgentLoop` call sites (from the wiring entry two commits ago) pick it
  up automatically with no further changes needed there.
- 6 new tests, mirroring the audio degrader's structure: declared
  duration reported, unknown-duration fallback, actual byte size (not a
  guess), the no-fabrication principle, an end-to-end test through the
  real `degrade_message` dispatcher, and `default_degraders()` coverage
  updated to expect all three modalities.

**Known gaps:**
- No real frame extraction — the deferred scope named above.
- No document degrader (`DocumentBlock` still has no converter).

**Next:** wiring release-bundle.yml to version tags, continuing to
deepen the foundry pipeline (near-duplicate dedup, provenance tracking,
scaling toy examples to a real small corpus), or real frame-sampling
video degradation if a video-decoding dependency becomes justified.

## 2026-07-22 — CI: version-tag releases, with a deliberate safety boundary

`release-bundle.yml`'s own known-gaps list has said "manual trigger
only, not wired to git tags/releases yet" since the entry that shipped
it. Closed — with one deliberate line not crossed.

**Built:**
- `release-bundle.yml` now also triggers on `push: tags: ["v*"]`. A new
  `publish-release` job (`needs: bundle`, so it only runs after all
  three OSes bundle successfully) downloads every platform's artifacts,
  flattens out just the real installer files (`.dmg`/`.msi`/`.exe`/
  `.AppImage`/`.deb` — `actions/download-artifact` also recreates
  non-file bundle output like the raw `.app` directory, which `gh
  release` can't attach as an asset at all), and creates a GitHub
  Release via `gh release create` (the CLI directly, not a third-party
  action — consistent with using `gh` throughout this session already,
  and one fewer external trust boundary for something that publishes to
  the public repo).
- **The deliberate safety boundary:** the release is created with
  `--draft --prerelease`. A draft is invisible to the public and sends
  no notification to watchers until a maintainer explicitly clicks
  "Publish release" in the GitHub UI. Pushing a version tag — an action
  that could happen accidentally, or during testing — must never be
  enough, on its own, to make something publicly visible; only an
  additional, deliberate human action does that. This matters
  specifically because a real GitHub Release (unlike the `workflow_dispatch`
  runs used to verify this pipeline all along) is genuinely public,
  shared state — the same category of action this project's own working
  practice treats as requiring explicit confirmation, not something to
  automate all the way to "live" without a human in the loop.

**What's verified vs. not, stated precisely rather than blurred
together:** the `bundle` job itself (shared, unchanged code) was
re-verified with a fresh `workflow_dispatch` run after this change,
confirming the new trigger didn't regress anything already proven
working. The new `publish-release` job's actual behavior — the `gh
release create` step, the installer-flattening `find` — has **not**
been live-tested against a real tag push, deliberately: doing so would
create a real (if draft) Release object and push a real tag to the
public repo, both real, visible actions on shared state that this
session's own operating practice reserves for explicit user
confirmation rather than autonomous action mid-loop. Verified instead by
careful reading and by confirming `publish-release`'s `if:` condition
correctly evaluates false (and the job is skipped) on the
`workflow_dispatch` run just used to re-verify `bundle`.

**Known gaps:**
- The `publish-release` job's own logic is unverified against a real
  tag push (see above) — the next real verification opportunity is
  whenever a maintainer actually decides to cut a version and push a
  tag, at which point the draft-release output should be checked before
  publishing it.
- Still unsigned (unchanged from prior entries).

**Next:** the actual first version tag, whenever a maintainer decides
it's time (that decision, and pushing the tag, is deliberately not this
session's to make autonomously) — or continuing foundry depth /
video frame-sampling in the meantime.

## 2026-07-22 — F0 continued: near-duplicate detection via MinHash

`dedup_documents`'s own docstring named the gap and deferred it: exact-hash
dedup only catches byte-identical documents. Real corpora have
near-duplicates too — a re-published article with one word edited, a
scraped page with a different timestamp. Closed with MinHash.

**Built:**
- `foundry/sarva_foundry/data/near_dedup.py` — `dedup_near_duplicates`:
  reduces each document's character-shingle set to a fixed-size MinHash
  signature (one minimum hash value per hash function, `hashlib.sha256`
  salted per function — no external minhash/datasketch dependency, the
  algorithm is the contribution, not the hash primitive underneath it),
  then estimates Jaccard similarity from the fraction of matching
  signature positions between two documents, dropping anything at or
  above `threshold` similarity to an earlier-kept document. Documented
  as O(kept²) pairwise comparison — fine at this project's scale, named
  honestly as needing an LSH banding index to go further, not silently
  implied to scale to a web-sized corpus.
- 13 tests in `tests/foundry/test_near_dedup.py`, including the actual
  algorithmic properties (deterministic signatures, identical shingle
  sets produce identical signatures, identical signatures estimate
  similarity 1.0), the dedup behavior itself (drops a real near-dup,
  keeps genuinely different documents, respects `threshold`, keeps
  first-occurrence order, handles empty documents), and composition with
  the existing exact-hash `dedup_documents`.

**A test-calibration bug caught by actually computing ground truth, not
shipped:** the first draft's "near-duplicate" test document was the
original text with a whole extra sentence appended (modeling "an article
republished with one more paragraph"). Empirically computing the *true*
Jaccard similarity for that pair — not just assuming a threshold would
obviously pass — showed only ~0.66 similarity, well below any reasonable
dedup threshold: appending new content dilutes shingle-set Jaccard far
more than intuition suggests, because Jaccard divides by the *union*,
and a whole new sentence adds a large batch of shingles no version of
the document shared before. The MinHash *implementation* was correct the
whole time (its estimate tracked the true value closely, ~0.62 vs.
~0.66); the test's mental model of "what counts as near-duplicate in
shingle-similarity terms" was wrong. Fixed by using a small in-place
edit (one word changed) instead, which is both a more realistic
near-duplicate scenario and empirically scores ~0.85 — comfortably
above threshold. Documented directly in `docs/foundry/training.md`, not
quietly corrected and forgotten.

**Known gaps:**
- O(kept²) — no LSH banding index, so this doesn't scale to a web-sized
  corpus as-is.
- Character shingles only; word-level or sentence-level shingling (a
  different tradeoff — more robust to word-order-preserving paraphrase,
  less robust to typos) isn't implemented.

**Next:** the actual first version tag (still the user's call), real
frame-sampling video degradation, or provenance/license tracking for
the corpus-sourcing pipeline.

## 2026-07-22 — F0 continued: provenance/license tracking, and a refactor first

Closes the last of §3.6c's explicitly-named requirements this session
has been working through ("each recipe documented with provenance and
license notes"). Getting there cleanly took a refactor of already-shipped
code first.

**Built:**
- Refactored `dedup_documents`, `filter_by_length` (`corpus.py`), and
  `dedup_near_duplicates` (`near_dedup.py`) into thin wrappers around new
  generic `_dedup_by_key`/`_filter_by_length_key`/`_dedup_near_duplicates_by_key`
  helpers (PEP 695 generic syntax — `def _dedup_by_key[T](...)`, matching
  ruff's `UP047` for this Python 3.12+ project), each parameterized by a
  `key: Callable[[T], str]` extractor. Re-ran the full existing
  `test_corpus.py`/`test_near_dedup.py` suites immediately after — 24
  tests, all passing unchanged — to confirm this was a genuine
  behavior-preserving refactor before building anything on top of it.
- `foundry/sarva_foundry/data/provenance.py` — `SourcedDocument` (frozen:
  `text`, `source_path`, `license`) plus `load_text_files_with_provenance`,
  `dedup_sourced_documents`, `filter_sourced_documents_by_length`,
  `dedup_near_duplicate_sourced_documents`. Each of the three dedup/filter
  functions calls the *exact same* generic helper the plain-`str`
  pipeline uses — keyed on `lambda d: d.text` instead of `lambda d: d` —
  not a reimplementation, and deliberately not "run the string pipeline
  separately, then guess which output belongs to which input," which
  breaks the moment two *different* source files happen to contain
  identical text.
- `sarva_foundry.data.corpus`/`near_dedup`'s existing plain-`str`
  functions are completely untouched from a caller's perspective —
  provenance is an additive, opt-in layer, not a breaking change to
  code three prior entries already shipped and tested.
- 9 new tests in `tests/foundry/test_provenance.py`, including the one
  that actually justifies the "don't reconstruct, key through instead"
  design: two different source files with byte-identical text — the
  correct behavior is dropping the second file while keeping the
  *first* file's provenance, verified directly rather than assumed.

**Known gaps:**
- `load_text_files_with_provenance` applies one `license` string
  uniformly per call — real per-file license variation within one
  directory needs a manifest (path → license mapping), not implemented.
- Same O(kept²) near-dup scaling limit as the plain-string version,
  inherited by construction since they share the same underlying helper.

**Next:** the actual first version tag (still the user's call), real
frame-sampling video degradation, or a per-file license manifest for
directories with mixed sources.

## 2026-07-22 — F0 continued: per-file license manifest

Closes the known gap the provenance entry named: `load_text_files_with_provenance`
applies one license uniformly per call, which doesn't cover a directory
with genuinely mixed sources.

**Built:**
- `load_text_files_from_manifest` (`provenance.py`) — reads a JSON
  manifest mapping each document's path to its own license string, paths
  resolved relative to the *manifest's own directory* so the manifest
  travels with its corpus without path edits. Validates every entry
  rather than trusting it: raises clearly on a malformed manifest (not a
  JSON object), a missing file, or a path traversal attempt.
- **Caught, not just handled defensively:** a real pathlib gotcha —
  `Path("/safe/dir") / "/etc/passwd"` silently *discards* the base and
  evaluates to `/etc/passwd` alone, since joining an absolute path onto
  any base always wins. A manifest entry that's absolute (by accident,
  or by injection if a manifest is ever untrusted input) would otherwise
  read a file nowhere near the corpus with no error at all. The
  traversal check validates the final *resolved* path against the
  manifest's directory rather than pattern-matching the raw string
  (e.g. checking for `".."`), so it catches this exact case — pinned by
  a dedicated test (`test_load_from_manifest_rejects_an_absolute_path_entry`)
  distinct from the plain `"../"` traversal test, since a naive
  string-based guard would pass the absolute-path case while still
  looking like it handles "path traversal."
- 7 new tests in `tests/foundry/test_provenance.py`: per-file license
  assignment, path resolution relative to the manifest, the missing-file
  and malformed-manifest error paths, both traversal cases above, and
  composition with the existing dedup/filter functions.

**Known gaps:**
- One manifest per directory tree, no glob patterns or wildcard license
  assignment — every file needs an explicit manifest entry.
- No manifest *generation* tooling (e.g. scaffolding one from a
  directory listing) — authored by hand for now.

**Next:** the actual first version tag (still the user's call), real
frame-sampling video degradation, or scaling the toy pipeline examples
up to a real small public-domain corpus now that the sourcing side is
fully built out.

## 2026-07-22 — Core: semantic memory (TF-IDF + cosine similarity), and wired in

`sarva.memory`'s own module docstring named this as future work since
T0: "a vector index or database-backed store can layer on top later
without changing this contract." Built it — and, having learned the
exact lesson from an earlier entry (a fully-tested feature that sat
completely unreachable because nothing actually called it), wired it
into the agent's real tool runtime in the same entry rather than as an
afterthought.

**Built:**
- `core/sarva/memory/vector.py` — `VectorMemoryStore`: SQLite for
  storage, TF-IDF + cosine similarity for retrieval, entirely from
  scratch (no external ML/vector-search library). Deliberately not
  neural embeddings: a real embedding pipeline needs a live
  embedding-model API this project has no configured provider for, and
  building against one now would be unverifiable without credentials —
  the same trap a web-search tool would fall into, which is why this
  entry is a memory store instead of that. TF-IDF is a genuine first
  tier, not a toy stand-in: a real sparse vector representation scored
  with the same cosine-similarity metric dense embeddings use, fully
  local and fully testable today. Deliberately not `sqlite-vec` either
  (the design doc's stated choice) — that extension indexes *dense*
  vectors for approximate nearest-neighbor search at scale; these are
  sparse, per-query-computed vectors scored exactly, which doesn't need
  an ANN index at this project's memory-store size.
- `RememberTool`/`RecallMemoryTool` (`core/sarva/agent/tools.py`), added
  to `BUILTIN_TOOLS` — the model can explicitly save a note and later
  search for it, both real tool calls, not a hidden background process.
- 13 tests in `test_vector_memory.py`, including the one that actually
  matters most: a real relevance-ranking test (topically related "fox"/
  "dog" entries score above an unrelated "quarterly revenue" entry for
  a fox/dog query) — proving the retrieval genuinely works, not just
  that it runs without crashing.

**A real bug caught before shipping, not after:** the first draft
constructed each tool's default `VectorMemoryStore` eagerly in
`__init__`. `BUILTIN_TOOLS` is a module-level list — `RememberTool()`
and `RecallMemoryTool()` get constructed once, at *import* time. Eager
construction would have made merely `import sarva.agent.tools` open (and,
via the store's own `mkdir`), create a real file at `~/.sarva/memory.db`
on every machine that ever imports the module — including test/CI runs
that touch no filesystem otherwise. Fixed by deferring store construction
into a `_get_store()` helper called from `run()`, not `__init__`.
Verified two ways: a hermetic unit test asserting `tool._store is None`
immediately after construction (checking the actual internal state, not
a fragile `Path.home()`-monkeypatch proxy — `DEFAULT_MEMORY_DB_PATH` is
a module-level constant already bound at import time, so patching
`Path.home` afterward wouldn't have caught anything), and an empirical
check: imported the real module fresh and confirmed
`~/.sarva/memory.db` genuinely does not exist before or after.

**Known gaps:**
- No per-session isolation for the default store — every entry lands in
  one shared `"default"` bucket. Needs the CLI's `--session` flag
  threaded through `ToolContext`, which doesn't expose a session
  identifier to tools at all today; a real, separate design decision.
- No neural-embedding tier — see above for why, and what would need to
  change to add one (a configured embeddings provider).
- No automatic "remember this" — memory only grows via an explicit
  `remember` tool call the model itself decides to make.

**Next:** the actual first version tag (still the user's call), threading
session identity through `ToolContext` so memory tools can be genuinely
per-session, or real frame-sampling video degradation.

## 2026-07-22 — Core: session identity threaded through ToolContext

Closes the exact known gap the memory entry named: every `remember`/
`recall_memory` call landed in one shared `"default"` bucket, since
nothing threaded the CLI's `--session` flag (or the server's `session`
field) down into a tool's `ToolContext` at all.

**Built:**
- `ToolContext` gained an optional `session_id: str | None = None` field
  — backward compatible, every existing construction site unaffected.
- `AgentLoop.run()` gained a matching optional `session_id` parameter,
  threaded straight into the `ToolContext` it constructs — additive,
  following the exact pattern `extra_content`/`transcript_out` already
  established for extending `run()`'s signature beyond spec-03's frozen
  literal code (same reasoning as the earlier degradation-fallback
  entry: this is the loop's established, precedented way of growing new
  capability without touching what's actually frozen — the state
  machine, events, budgets, tool contract).
- `RememberTool`/`RecallMemoryTool` now prefer `ctx.session_id` over
  their own constructor-time `session_id` default — the live session a
  run actually belongs to wins over a static fallback.
- All four real `AgentLoop.run()` call sites (CLI's `chat`/`run`,
  server's `/chat` and `/ws/chat`) updated to pass their already-existing
  `session`/`req.session` value straight through as `session_id=` —
  each of them already had this value in scope for `SessionStore`
  load/save, just never forwarded it to the loop.
- 5 new tests: `ctx.session_id` winning over the tool's fallback,
  falling back correctly when `ctx.session_id` is `None`, session-scoped
  recall actually excluding another session's entries — plus, the one
  that matters most, two integration tests in `test_agent.py` using a
  tool that echoes `ctx.session_id` straight back through a *real*
  `AgentLoop.run(session_id=...)` call: proof the value genuinely
  reaches a tool's context end to end, and a regression guard that a
  run with no `session_id` leaves `ctx.session_id` as `None` exactly as
  before this entry, not some accidental new default.

**Known gaps:**
- No neural-embedding tier still (unchanged — see the prior entry for
  why).
- No automatic "remember this" still — memory only grows via an
  explicit `remember` tool call.

**Next:** the actual first version tag (still the user's call), real
frame-sampling video degradation, or scaling the foundry pipeline
examples to a real small public-domain corpus.
