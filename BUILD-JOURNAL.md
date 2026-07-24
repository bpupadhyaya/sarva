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

## MCP client — the ecosystem's tools plug in with no Sarva-specific glue

§3.5's tool runtime list named this from the start ("MCP client support
so the ecosystem's tools plug in without Sarva-specific glue") and it
had been unaddressed until now — confirmed by grep, not assumed, same
discipline as the earlier degrader-registry entry.

`sarva.mcp_client` uses the official `mcp` Python SDK's `ClientSession`
— the same "official SDK, not hand-rolled protocol" pattern the
provider adapters already follow for anthropic/openai/google-genai, not
a from-scratch JSON-RPC client (that's reserved for the foundry's model
math, a different tier of "from scratch" entirely).

Only the stdio transport is wired up, deliberately: it's what the
majority of real MCP servers speak today (`npx`/`uvx`-launched local
processes), and it's the one transport genuinely verifiable offline —
spawn a real local subprocess, speak real MCP over its stdin/stdout, no
network call anywhere. HTTP/SSE transports are real, named, deferred
scope, not silently assumed covered.

`McpToolAdapter` wraps a single remote tool as an ordinary Sarva `Tool`
(same `spec` + `async def run(args, ctx)` shape every built-in uses), so
the agent loop, confirmation policy, and transcript logging don't need
to know or care that a given tool call is actually a subprocess round
trip. `list_mcp_tools(session)` lists everything a connected server
exposes, each already wrapped and ready to hand to `AgentLoop(tools=...)`.
Content conversion follows the same honesty principle as the degraders:
text/image content converts directly, anything else (audio, resource
links, embedded resources) reports its own declared MCP content type
rather than being silently dropped.

Wired into `sarva run --mcp-server "command args..."` (repeatable,
`shlex`-split, connected via an `AsyncExitStack` so every server's
subprocess is torn down cleanly at the end of the run) — merged into
the same flat tool list as the built-ins, so the model sees one
registry with no way to tell which tools came from where.

**Verified with a real server, not a mock of the protocol:**
`tests/fixtures/mcp_echo_server.py` is a genuine MCP server built with
the SDK's own `FastMCP`, launched as a real subprocess over real stdio
by `tests/conformance/test_mcp_client.py`. Covers: tool listing against
the real server, a successful call, a failing call (proving MCP error
propagation reaches `ToolResultBlock.is_error` — confirmed empirically
first that FastMCP turns a raised exception into `isError=True` with the
exception message in the content, not a protocol-level error), and —
the test that actually proves the integration rather than just the
wrapper in isolation — a real `AgentLoop.run()` driven by a
`MockProvider` script that calls the MCP-backed tool and gets back the
exact text the real subprocess produced. Also smoke-tested through the
actual CLI (`sarva run ... --mcp-server "..."`, real connect + tool
listing + clean shutdown, no source repo shortcuts). 196 Python tests
total now (192 → 196), all real, no protocol mocking anywhere in this
feature.

**Next:** the actual first version tag (still the user's call), real
frame-sampling video degradation, or scaling the foundry pipeline
examples to a real small public-domain corpus.

## Real frame-sampling video degradation, closing the last-named degrader gap

`VideoToTextDegrader`'s own docstring (and `Degrader`'s in content.py,
which uses "video -> [image frames + text transcript]" as its
*motivating example*) had named this as real, deferred work since the
degrader trio first shipped — closed now, not left as a permanent
disclaimer.

Uses **PyAV** (`av`), not a system `ffmpeg` binary: PyAV statically
bundles its own decoder libraries into the wheels it publishes on PyPI
for macOS/Linux/Windows, so there's no repeat of the cross-platform CI
availability gamble this project already paid for once, the hard way,
getting the Windows sidecar freeze working. The audio degrader's
stdlib-only tradeoff (documented in its own module) was made when the
realistic choices were "stdlib `wave`, can't touch compressed audio" or
"a heavy dependency not justified for a metadata-only converter" — a
genuinely portable, self-contained decoding library changes that
calculus for video, where there's no stdlib fallback at all and
sampling actual frames is the entire point of the modality.

On a genuinely decodable video: decodes real frames, samples up to 4
evenly spaced across the whole video (bounding output size regardless of
source length, same spirit as the corpus pipeline's length filters), and
reports the **real decoded duration** — proven with a test that
deliberately sets a wrong `duration_s` on the block and confirms the
real decoded value wins, not just that some duration string appears.
Same honesty principle as always: sampled frames are real pixels, never
a fabricated description of what they show. Undecodable bytes (corrupt
data, an unsupported container, a zero-frame stream) fall back cleanly
to the original metadata-only report rather than raising — "couldn't
decode this particular file" is an expected case for a byte-agnostic
converter, not a bug.

**Verified with real encode+decode round trips, not fixture files
committed to the repo:** tests synthesize tiny mp4s directly with PyAV
(distinct solid-colored frames so a test can tell them apart, not just
count them), decode them back through the real degrader, and confirm
sampled frames are genuinely Pillow-openable PNGs at the right
resolution. One test proves the full documented chain end to end
through `degrade_message`'s own recursion: video → sampled image frames
→ (a text-only target still can't see images either) → text — not just
that `VideoToTextDegrader` emits `ImageBlock`s in isolation. A dedicated
zero-frame-stream test guards the one PyAV edge case that's decodable as
a container but has nothing to sample. 196 → 201 Python tests.

**Known gap:** no audio-track extraction from video yet (frames only) —
named, not silently assumed covered.

**Next:** the actual first version tag (still the user's call), or
scaling the foundry pipeline examples to a real small public-domain
corpus.

## Foundry examples scaled to a real, small, public-domain corpus

Every foundry example so far (02-04) trained on four hardcoded toy
sentences — enough to prove the mechanics (tokenizer, transformer,
checkpoint/resume) but never actually exercising the corpus-sourcing
pipeline (`sarva_foundry.data.corpus`/`.near_dedup`/`.provenance`) on
real text.

`examples/06_real_corpus_pretraining.py` fetches three short, genuinely
public-domain texts from Project Gutenberg (*A Modest Proposal*, *The
Hunting of the Snark*, *The Time Machine* — picked small on purpose:
this is a laptop-scale demo, not a run meant to produce a useful model),
runs them through the real pipeline this project actually has —
`load_text_files_with_provenance` → `dedup_sourced_documents` →
`dedup_near_duplicate_sourced_documents` → `filter_sourced_documents_by_length`
— with an honestly-stated real license (`"Public Domain (Project
Gutenberg, US)"`) attached to every surviving document, then trains the
same tokenizer/transformer/`Trainer` stack example 04 exercises, now on
~90K real tokens instead of a few dozen synthetic ones.

**Verified by actually running it, with real timing, not assumed:** full
run (download 3 texts, ~250KB total → BPE-train a 1200-vocab tokenizer →
200 training steps on a 4-layer/128-dim transformer) completes in ~12.5s
wall-clock on this machine. Loss goes 116.5 → ~8 over the run — real
learning on real prose, not a toy string repeated until memorized.
Network access is required for the download step only; gated the same
way `examples/05_web_fetch.py` gates on a missing API key — a clear
message and a clean exit if Project Gutenberg can't be reached, not a
stack trace.

**Deliberately not a pytest test:** like every other example script in
this repo, it's a runnable demonstration, not conformance-tested — the
corpus pipeline's own unit tests (dedup/near-dedup/filter/provenance) are
already covered hermetically without network in `tests/foundry/`; this
example's job is proving those tested pieces compose correctly against
real, larger, externally-sourced text, which a unit test with synthetic
strings structurally can't prove.

**Next:** the actual first version tag (still the user's call).

## v0.1.0 tagged and released (draft) — first version tag, with explicit go-ahead

Every earlier mention of this had deferred it as "the user's own
decision, not mine to take autonomously" — the whole reason
`publish-release`'s tag-triggered path existed but had never been live-
tested. Asked directly whether to cut it now that every other named
milestone was shipped; got an explicit go-ahead. Tagged the current
commit as `v0.1.0` and pushed the tag, triggering `release-bundle.yml`'s
real tag path for the first time (previously only exercised via
`workflow_dispatch`, where `publish-release` correctly *skips*).

**Verified the real thing, not just green checkmarks:** all three
`bundle` jobs succeeded (windows 5m10s, ubuntu 8m18s, macos 2m9s), then
`publish-release` ran for real and `gh release view v0.1.0 --json
isDraft,isPrerelease,assets` confirmed `isDraft: true`, `isPrerelease:
true` (the safety boundary held — invisible to the public until a
maintainer explicitly publishes it) with all 5 real installer assets
attached (macOS `.dmg` 53MB, Linux `.AppImage` 167MB + `.deb` 92.5MB,
Windows `.exe` 71MB + `.msi` 71.8MB). One cosmetic detail, not a bug:
the release's URL shows a placeholder `untagged-<hash>` slug rather than
`v0.1.0` — normal GitHub behavior for an unpublished draft; the API's
`tagName` field already correctly reports `v0.1.0`.

**Still the maintainer's call, unchanged:** actually clicking "Publish
release" in the GitHub UI.

## OpenAI provider adapter — closing T1's other named provider gap

T1's own roadmap line has always read "Provider layer (Anthropic+OpenAI+
Google+Ollama)" — Anthropic, Ollama, and Mock existed; OpenAI didn't.
`sarva.providers.openai_provider.OpenAIProvider` implements the same
`Provider` protocol via OpenAI's Chat Completions streaming API, same
"thin adapter, same contract" pattern as the other two.

The one genuinely novel piece of logic, called out directly in the
adapter's own docstring: OpenAI streams a tool call's `arguments` as
string fragments across many chunks, keyed by `index` — unlike Anthropic
(whose SDK hands back an already-assembled `get_final_message()`) or
Ollama (whose chat API sends each tool call complete in one chunk). Got
this wrong once in the sense of not trusting it enough on the first
pass: wrote a hermetic test specifically interleaving *two concurrent*
tool calls' argument fragments chunk-by-chunk to prove index-keyed
accumulation doesn't cross-contaminate them — the one place a
live-only test wouldn't reliably force the bug, since a live model might
never happen to interleave two calls in exactly the order that would
expose an index mistake. A separate test proves malformed/truncated
argument JSON degrades to an empty dict rather than crashing the
adapter.

**A real, deliberate scope boundary, not an oversight:** no entries
added to `providers/data/models.yaml`. That file's own header states
it's "re-validated at every release," and this project's honesty
principle — no fabricated content anywhere, the same rule the degraders
live by — applies to a registry file exactly as much as to model output.
A web search for "current OpenAI model + 2026 pricing" turned up nothing
trustworthy enough to write into a file explicitly meant to be accurate
(low-authority SEO aggregator sites naming implausible model variants,
the pattern of AI-generated pricing-page spam, not OpenAI's own
documentation) — writing that data in anyway would have been exactly
the kind of fabrication this codebase explicitly refuses everywhere
else. The adapter is real and complete; wiring a specific verified model
in is the one-entry config change the registry design was built for, left
for whoever has that data. `runtime.py`'s `build_providers`/`build_router`
wire it in behind `OPENAI_API_KEY`, same guard shape as the Anthropic
adapter — currently a no-op until a real `provider: openai` entry exists,
by design.

Following the established Anthropic/Ollama precedent (Ollama has zero
unit tests, only a live-gated one; Anthropic unit-tests only its pure
translation function): `test_openai_provider.py` covers
`_to_openai_messages` hermetically (5 tests, including the one dedicated
to something Anthropic doesn't need — OpenAI requires a *separate*
role="tool" message per tool_call_id, unlike Anthropic's single
content-array-with-multiple-tool-results shape), and a live-gated test
was added to `tests/live/test_live_providers.py` (skipped without
`OPENAI_API_KEY`, model id overridable via `OPENAI_TEST_MODEL` since no
verified-current model id is hardcoded anywhere). 201 → 209 Python
tests, all passing; `sarva.runtime.build_providers()` empirically
confirmed to still return only `{"mock": ...}` with no keys set — no
import-time side effects, no accidental client construction.

## Google Gemini provider adapter — T1's provider layer now complete

Closes the last of T1's four named providers (Anthropic+OpenAI+Google+
Ollama). `sarva.providers.google_provider.GoogleProvider` implements the
`Provider` protocol via `google-genai`'s async streaming API, same thin-
adapter contract as the other three.

The one genuinely novel, adapter-specific piece — and a real bug caught
and fixed before shipping, not hypothetical: Gemini reports
`finish_reason=STOP` even when the response includes `function_call`
parts. There is no distinct "made a tool call" finish reason at all,
unlike Anthropic's `tool_use` or OpenAI's `tool_calls`. The first draft
trusted `finish_reason` the same way the other two adapters correctly
do, which would have silently misreported *every* Gemini tool-use turn
as `END_TURN` — a structural bug that would make the agent loop treat a
tool-call request as if the model were simply done. Caught by writing a
hermetic test first (`test_tool_call_infers_tool_use_despite_stop_finish_reason`)
using duck-typed fake chunks with `finish_reason="STOP"` *and* a
function_call part together — the exact shape Gemini actually sends —
rather than trusting it would come up in a live run. Fixed by inferring
`TOOL_USE` from the presence of a tool-call block first, falling back to
the raw `finish_reason` mapping only when there isn't one.

A second real shape difference required its own translation logic:
Gemini's `FunctionResponse` requires a `name` field, but Sarva's
`ToolResultBlock` (like every other provider's tool-result shape) only
carries `tool_call_id` — resolved via `_tool_call_names()`, which scans
every `ToolCallBlock` across the whole request once per `generate()`
call to build an id→name map, rather than assuming the caller supplies
it.

Same deliberate scope boundaries as the OpenAI adapter, for the same
reasons: no `models.yaml` entries (no verified-current Gemini model
catalog/pricing to add responsibly), and `GenerateConfig.effort`/
`.thinking` left unmapped (Gemini's `thinking_config` shape is
unverified against a live model in this session). Also named honestly,
not silently assumed handled: no dedicated network-connection-failure
exception type was found documented for `google-genai` the way
`anthropic`/`openai` both document `APIConnectionError` — only HTTP-level
`ClientError`/`ServerError` are caught; a real connection failure
surfaces uncaught until verified live.

Wired into `runtime.py` behind `GEMINI_API_KEY`/`GOOGLE_API_KEY`, a
no-op until a real `provider: google` registry entry exists, same shape
as the other two additions. `test_google_provider.py` covers
`_to_gemini_content`/`_tool_call_names` hermetically (7 tests), a live-
gated test was added to `tests/live/test_live_providers.py` (model id
overridable via `GOOGLE_TEST_MODEL`). 209 → 219 Python tests, all
passing.

## Mixture-of-Experts — the first frontier-class architecture extension (§3.6a)

T1's provider layer being done freed up the next real, well-scoped,
locally-verifiable piece: §3.6a's "frontier-class architecture" line
names Mixture-of-Experts explicitly — "the K3/DeepSeek-class design:
fine-grained experts, shared experts, aux-loss-free load balancing" —
and it's genuinely testable at toy scale on a laptop, unlike the
distributed-training slice of F1 this same section defers to real
compute.

`sarva_foundry.model.moe.MoEFeedForward` swaps in for `SwiGLU` via a new
`TransformerConfig.moe: MoEConfig | None` field (default `None`, dense
baseline completely unchanged — 13 existing `test_model.py` tests still
pass untouched). All three named ideas, not a generic MoE strawman:
fine-grained experts (many smaller FFNs vs. a few large ones), an
always-active shared expert, and aux-loss-free load balancing via a
`register_buffer` bias (never a `Parameter` — can't accumulate a
gradient) added to router logits for *selection* only, updated after
each forward by a fixed arithmetic rule (`update_expert_bias()`), never
by an auxiliary loss term competing with the real training objective.

**The one detail that makes "aux-loss-free" real rather than a relabeled
aux loss, pinned by a dedicated test:** selection uses `gate_logits +
bias`, but the *weight* applied to a selected expert's output comes from
softmax over the *raw*, unbiased logits of just the selected experts —
`test_route_bias_changes_selection_but_not_weight_of_a_selected_expert`
forces a different expert to be selected via a large bias and confirms
its weight is identical to what an unbiased selection of it would have
produced.

**A real test-construction bug caught by running it, not shipped:** the
first draft of the load-balancing convergence test froze the gate at
all-zero to "isolate" the bias's effect — this produced the *opposite*
of convergence, a winner-take-all oscillation where literally every
token piled onto whichever single expert currently had the highest bias,
flipping to a *different* single expert each round as the bias update
caught up (same load std-dev before and after, just relabeled — caught
by printing per-round loads, not by the assertion alone, exactly the
"verify a test's assumption empirically before trusting it" pattern this
project keeps re-learning). Root cause: a real gate gives different
tokens different per-token preferences, which is what lets tokens peel
off to alternative experts gradually as the bias narrows the gap between
over/underloaded experts — the graceful rebalancing the mechanism is
designed to produce. Fixed by using a real (untouched) random gate;
`test_load_balancing_converges_toward_balance_over_repeated_updates` now
shows the load's standard deviation shrinking by more than half over 50
update cycles from a deliberately fully-skewed start.

11 new tests total in `tests/foundry/test_moe.py`, including a full
trainability test (gradients flow through the router, every selected
expert, and the shared expert; loss decreases on a toy task, mirroring
the dense transformer's own trainability test) and a config-swap test
proving `DecoderOnlyTransformer` picks `MoEFeedForward` vs `SwiGLU`
purely from `TransformerConfig.moe` with identical output shapes either
way. `examples/07_moe_transformer.py` runs the same toy training loop as
example 03 with `update_expert_bias()` called after every optimizer
step, printing each layer's per-expert token counts every 50 steps —
real, visible convergence toward balance on an actual training run, not
just inside an isolated test. 230 Python tests total now (219 → 230).

**Honestly scoped, not silently implied broader:** dense per-expert
loop (`index_add_`), not scatter/gather or grouped-GEMM kernels — correct
and simple at this project's training scale, the same "commodity
substrate" boundary `layers.py` draws around `nn.Linear`, drawn here on
the routing/balancing math's side of it instead. `update_expert_bias()`
is a method the caller invokes, not auto-wired into `Trainer` — real,
deferred integration work, named rather than assumed.

**Next:** F1's real (non-toy) training infrastructure, an eval harness
(§3.6g), or the remaining §3.6a extensions (long-context scaling, native
multimodal input).

## Eval harness — grading every model with the same yardstick (§3.6g)

Closes §3.6g's named gap: "benchmark harness shared with the registry
(grades our models and third-party models with the same yardstick)."

`sarva.eval.harness.run_benchmark(benchmark, provider, model)` is
deliberately built against the `Provider` protocol, not any specific
backend — the same abstraction that already makes Anthropic/OpenAI/
Google/Ollama/Mock interchangeable everywhere else in this codebase
(the agent loop, the router, the CLI). One function call grades any of
them identically; the moment §3.1's planned foundry adapter exists (a
foundry-trained checkpoint plugged into the registry as a real
`Provider` — not built yet, named honestly as real deferred work), it
becomes gradable by this exact same harness with zero changes here.
Reuses `sarva.providers.base.complete()` (the existing "drain the
stream, get the `DoneEvent`" helper) instead of reimplementing stream
draining.

`ARITHMETIC`: a bundled, ten-case benchmark — real arithmetic problems,
each answer computed and hand-checked, not generated and assumed
correct. Deliberately small, not a claim to GSM8K-scale coverage, same
"real, not a fabricated placeholder" discipline as the corpus pipeline's
length filter and the degraders' metadata-only reports elsewhere in this
project. `contains_match` (checking whether the expected answer appears
anywhere in the output) is the default grader rather than
`exact_match`, since real models rarely answer with *only* the expected
string — an exact-match default would mostly measure formatting luck.
A case whose request fails (`ProviderError` — rate limit, auth, etc.) is
scored incorrect with the error recorded as its output, rather than
aborting the whole benchmark run; one flaky case shouldn't hide every
other case's real result.

Wired into the CLI as a real, runnable command: `sarva eval [--model
ID]` grades every available model (or one, filtered) against the bundled
benchmark and prints accuracy + correct/total side by side — the
roadmap language made concrete rather than left as a library-only
capability. 8 new tests in `tests/conformance/test_eval_harness.py`
(grader correctness, scoring math, the provider-error path, empty-report
edge case, custom-grader support). 230 → 238 Python tests.

**Next:** F1's real (non-toy) training infrastructure, or the remaining
§3.6a extensions (long-context scaling, native multimodal input).

## Long-context RoPE scaling — linear interpolation and NTK-aware scaling (§3.6a)

Closes the second item on §3.6a's "position-interpolation/NTK scaling"
line (MoE closed the first, native multimodal input remains). Two real,
distinct, named techniques, not a single generic "scale factor" knob —
implemented as `RopeScalingConfig(method="linear"|"ntk", factor=...)`,
threaded through `precompute_rope` → `GroupedQueryAttention` →
`TransformerConfig.rope_scaling` (default `None`, output bit-identical
to before this feature existed — confirmed by a dedicated regression
test, not just assumed from the diff).

**Linear** (Chen et al. 2023, position interpolation): divides every
position by `factor` before computing rotation angles.
**NTK-aware** (bloc97): raises the RoPE base `theta` itself instead of
touching positions. The distinguishing, testable property: the
highest-frequency dimension's rotation rate is `theta^0 = 1` regardless
of `theta`, so NTK leaves it **exactly** bit-identical to the unscaled
table at every position, while linear scaling (which divides every
position uniformly) visibly changes it — two dedicated tests prove this
directly rather than asserting it in a docstring, plus a matching pair
proving both techniques *do* stretch the lowest-frequency (long-range)
dimension, so neither is a no-op either. Relative-position invariance
(the property RoPE exists for) is verified to still hold under both
scaling configs, same style as the existing unscaled-table test.

**A real numeric-precision lesson while writing `examples/08_long_context_rope_scaling.py`:**
the first draft printed `cos(angle)` at a handful of short positions to
show the effect — and for the lowest-frequency dimension, every column
printed the same value to 4 decimal places, because real RoPE
frequencies are tiny by design (that's *why* long-context scaling
matters at all — the effect only becomes visually significant over
thousands of positions, not dozens). Caught by actually running the
draft and looking at the output, not assumed correct from the code.
Fixed by printing what's honestly demonstrable at toy scale instead:
raw per-dimension frequency *ratios* for NTK (position-independent,
exactly 1.0 at dim 0 and exactly `1/factor` at the lowest dimension —
visible without needing any position at all) and the exact
position-index equivalence linear scaling produces (`cos` at scaled
index `i*factor` matching `cos` at unscaled index `i` to 6 decimal
places). A real forward pass through an NTK-scaled model closes the
example, proving the config wiring produces a runnable model, not just
a standalone table function.

9 new tests in `tests/foundry/test_rope_scaling.py`, including a full
trainability test with linear scaling active. 238 → 247 Python tests.

**Next:** F1's real (non-toy) training infrastructure, or native
multimodal input — the last named piece of §3.6a's architecture list.

## Native multimodal input — the last named piece of §3.6a's architecture list

§3.6a: "native multimodal (vision encoder + projector; audio later)."
With MoE and long-context RoPE scaling already shipped, this closes
every named architecture item — audio stays explicitly future work per
the design doc's own wording.

`sarva_foundry.model.vision` adds three real, standard LLaVA-class
pieces, each reusing already-tested substrate: `PatchEmbed` (strided-
conv "patchify," proven mathematically identical to manual
flatten+linear, not just shape-checked), `VisionEncoder` (patchify + N
*bidirectional* transformer blocks, reusing `GroupedQueryAttention`/
`RMSNorm`/`SwiGLU` with a new `causal: bool = True` parameter —
`causal=False` for vision, the text decoder's default completely
unchanged), and `Projector` (2-layer MLP + GELU, the LLaVA-1.5-style
connector).

`DecoderOnlyTransformer.forward_multimodal(token_ids, image_embeds,
image_token_id)` is the splice point: every `image_token_id` position
gets replaced by the next projected image embedding, then the *same*
causal decoder body (refactored behind a shared `_forward_embeds`
helper so text-only `forward` and multimodal `forward_multimodal` are
provably the same code path past embedding construction) runs
unmodified.

**Two tests worth naming for what they specifically guard against, not
just "more coverage":**
- `test_vision_encoder_is_genuinely_bidirectional_not_accidentally_causal`
  — the mirror image of the existing causal-masking test: perturbing the
  *first* patch must change the *last* patch's output, proving
  `causal=False` is genuinely wired through rather than silently
  ignored (a shape-only test can't distinguish a correctly bidirectional
  encoder from an accidentally-still-causal one).
- `test_full_stack_is_trainable_gradients_flow_through_vision_and_text`
  — asserts every parameter across the vision encoder, the projector,
  AND the text decoder receives a real gradient during a real training
  step. A broken splice (e.g. an accidental `.detach()` in
  `embed_multimodal`) would silently zero out the vision/projector
  gradients while still producing plausible-shaped logits — exactly the
  kind of bug a shape-only or even a loss-decreases-only test can miss,
  since the text decoder alone could still fit *some* pattern from the
  placeholder positions' fixed (untrained) embeddings.

`examples/09_multimodal_vision_transformer.py` trains the full stack on
a deliberately trivial but real task: a solid red image should make the
model predict token 5, a solid blue image token 10, with *identical*
surrounding text tokens in both cases — getting both right after
training is only possible if the model is genuinely using image
content, since there's no way to guess correctly from text alone. Loss
goes from 64.7 to 0.0000; the model correctly predicts both tokens.

**Honestly named simplification, not silently assumed equivalent:** the
vision encoder reuses 1D RoPE over the flattened patch sequence — same
mechanism the text decoder already has, not a 2D-aware positional scheme
(2D RoPE or learned 2D embeddings) a production vision encoder would use
to actually encode row/column structure.

13 new tests in `tests/foundry/test_vision.py`. 247 → 260 Python tests.
§3.6a's full named architecture list (dense baseline, MoE, long-context
RoPE scaling, native multimodal input) is now built.

**Next:** F1's real (non-toy) training infrastructure, or F2
post-training (§3.6e: SFT/DPO/agentic RL).

## Supervised fine-tuning — the first F2 post-training piece (§3.6e)

§3.6e: "SFT -> DPO/RLHF -> agentic RL... this, not pretraining, is what
turns a base model into a Fable/K3-class agent." SFT is that line's
first piece, built as a composable addition to the *existing* `Trainer`
rather than a parallel implementation — same pattern MoE and RoPE
scaling established for the model architecture side, now applied to the
training loop: `Trainer.train_step` gained one optional `loss_mask`
parameter, and that alone is the entire difference between pretraining
and SFT here. `loss_mask=None` (the default, and every call site before
this existed) is exactly the original unmasked behavior — confirmed
bit-identical by a dedicated regression test, not just "the diff looks
safe."

`sarva_foundry.train.sft` builds that mask from `(prompt, response)`
pairs: `encode_sft_example` concatenates prompt + response +
`end_of_turn` (reusing `DOCUMENT_SEPARATOR`, the same document-boundary
special token plain pretraining already uses, rather than inventing a
second one for the same role), with the mask covering only the response
(so the model learns to predict answers, never to reproduce prompts —
the entire point of SFT versus plain next-token training on the same
text). `build_sft_batch` pads a batch to its longest example and shifts
for next-token prediction; right-padding is safe under causal attention
by construction (the causal mask already guarantees a padded position
can't influence an earlier real position), not by convention.

**The property the tests check directly, not just "shapes are right":**
two training batches whose targets differ *only* at masked-out (prompt)
positions must produce bit-identical loss —
`test_loss_mask_makes_masked_target_values_irrelevant_to_the_loss`
proves this directly by corrupting only masked-out targets and
confirming the loss doesn't move. The complementary test confirms an
*unmasked* target change does move the loss, ruling out the trivial
failure mode where a mask that excludes everything would "pass" the
first test while making SFT training a complete no-op.

**A real, non-mocked pathological-input test:** `build_sft_batch`
guards against a batch too short to form even one training pair after
the shift. Rather than fabricate that scenario, the test constructs it
for real — `SFTExample(prompt="", response="")` genuinely encodes to
exactly one token (the `end_of_turn` marker; both `encode("")` calls
return empty lists), which is a real, reachable input through the
documented API, not a contrived mock.

`examples/10_sft_toy_assistant.py` runs the full two-stage pipeline on a
toy model: after plain pretraining alone, greedy-decoding from any of
three different questions produces the *same* generic babbled
continuation (the base model has no notion of "answer this specific
question" yet); after SFT on three `(prompt, response)` pairs,
greedy-decoding from each distinct prompt produces its own distinct,
correct response — real, printed proof the model learned to condition
its answer on the actual question, not just memorize one fixed
continuation regardless of what's asked.

11 new tests (7 in `test_sft.py`, 4 in `test_trainer.py`). 260 → 271
Python tests.

**Next:** F1's real (non-toy) training infrastructure, or the rest of
§3.6e's post-training line (DPO/RLHF, agentic RL) beyond SFT.

## DPO — the second F2 post-training piece (§3.6e)

§3.6e: "SFT -> DPO/RLHF -> agentic RL." Direct Preference Optimization
(Rafailov et al. 2023) teaches a model to *prefer* one response over
another for the same prompt using nothing but which one was chosen — no
reward model, no RL rollouts. `sarva_foundry.train.dpo.build_dpo_batch`
reuses SFT's own `build_sft_batch` rather than a parallel encoding path
(a DPO preference triple is exactly two SFT-shaped pairs sharing one
prompt); `Trainer.dpo_step` is a new method rather than another
`train_step` parameter, since DPO genuinely needs four forward passes
(policy × {chosen, rejected}, frozen reference × {chosen, rejected})
instead of `train_step`'s one, but shares the same optimizer/grad-clip/
step-counting machinery.

**The strongest test in this batch, and it's not a trainability test —
it's an exact algebraic fixed point:** when the policy is literally
identical to the reference model (true at DPO's very first step, before
any update), the chosen and rejected log-ratio terms are identical, so
the loss must equal `-log(sigmoid(0)) = ln(2) ≈ 0.6931` — not
approximately, exactly, straight from the formula.
`test_dpo_step_initial_loss_is_exactly_ln2_when_policy_equals_reference`
checks this on the full `dpo_step` path (real model forward passes on a
real tiny transformer, not an isolated-tensor version of the formula)
and it holds to `1e-4`. Verified empirically before writing the test,
not just derived on paper: a standalone script confirmed
`trainer.dpo_step(...)` returns `0.6931471824645996` against
`math.log(2) == 0.6931471805599453` on the very first call.

Two more properties tested directly: the reference model's forward pass
genuinely runs frozen (`p.grad is None` for every reference parameter
after a step, regardless of the caller's own `requires_grad` settings —
`dpo_step` wraps the reference forward in `torch.no_grad()` itself
rather than trusting the caller froze it), and after real training the
policy's preference *margin* (chosen log-probability minus rejected)
must be strictly larger than at initialization — the actual thing DPO
training exists to accomplish, not just "loss went down." 7 new tests
in `test_dpo.py`.

`examples/11_dpo_preference_tuning.py` makes the effect visible end to
end: SFT first on *both* candidate responses (so the model can already
produce either one, leaving preference close to neutral — the printed
margin after SFT alone was `-0.003`), then DPO on a single preference
pair shifts the margin to `+65.742` — no reward model, no sampled
rollouts, one preference pair. The initial DPO loss printed by the
example is exactly `0.6931`, the same fixed point the test pins.

278 Python tests total now (271 → 278).

**Next:** F1's real (non-toy) training infrastructure, or the last
piece of §3.6e's post-training line — agentic RL (RL on long-horizon
tool-use tasks, sandboxed coding-environment harness, distillation from
frontier models).

## Distillation — frontier-as-teacher synthetic data (§3.6c), and core meets foundry for the first time

§3.6c: "synthetic-data generation (frontier-as-teacher via the provider
layer)." `sarva.distill` (core) generates `(prompt, completion)` pairs
from any real `Provider` — the same abstraction `sarva.eval.harness.
run_benchmark` already uses to grade every registered model
identically, reused here to *generate* data instead of scoring it, so
whichever provider is configured (Anthropic, OpenAI, Google, a local
Ollama model) can serve as the teacher with zero backend-specific code.

**A real architectural decision, made deliberately, not by default:**
`distill()` returns plain `DistillationRecord`s (prompt/completion/
model), not `sarva_foundry.train.sft.SFTExample` objects directly.
`core`'s and `sarva_foundry`'s `pyproject.toml`s name completely
disjoint dependency sets (`anthropic`/`openai`/`google-genai`/`fastapi`/
... vs. `torch`/`numpy`), and until now neither package has ever
imported the other. Keeping it that way here means a caller who wants
foundry SFT data writes one line of glue
(`SFTExample(prompt=r.prompt, response=r.completion)`) in their own
script rather than either package taking on the other's entire
dependency tree just to pass a dataclass across a boundary.
`examples/12_distillation_to_sft.py` is that glue script, made runnable
end to end — the first example in this project to import from both
`sarva` and `sarva_foundry` in the same file, at the script level where
that kind of composition belongs.

**A deliberate difference from the eval harness's error handling, named
explicitly, not just implemented differently:** `run_benchmark` scores
a failing case as incorrect and keeps going, since one bad benchmark
case shouldn't hide every other case's real result. `distill()` does
the opposite — a `ProviderError` on any prompt propagates immediately.
Distillation output becomes training data; a silently-missing or
garbage record is a worse outcome than a loud failure a caller can
retry or investigate.

Wired into the CLI as `sarva distill prompts.txt --model ID --out
out.jsonl`, smoke-tested for real end to end against the zero-config
Mock provider (no API key needed to verify the command itself works —
`sarva distill` correctly read a 2-line prompts file, generated 2
records, and wrote valid JSONL). 7 new tests in `test_distill.py`,
covering generation ordering, model-id tagging, the error-propagation
difference from the eval harness, and a JSONL round-trip.

**Honestly scoped, not silently claimed verified:** `examples/12`
itself requires a real API key this environment doesn't have — unlike
every provider adapter before its first live run, it can't be exercised
end to end here. Verified everything that's verifiable without one: the
no-key path prints a clear message and exits cleanly (matching
`examples/05_web_fetch.py`'s established gating pattern), and every
cross-package import (`sarva.distill`, `sarva.providers.
anthropic_provider`, `sarva_foundry.*` all in one file) resolves with no
import errors — confirmed by actually running the script and observing
it reach the API-key check, not just reading the code and assuming it
would.

285 Python tests total now (278 → 285).

**Next:** F1's real (non-toy) training infrastructure, or the last
piece of §3.6e's post-training line — agentic RL (RL on long-horizon
tool-use tasks, sandboxed coding-environment harness).

## Docs Chapter 2 — the provider abstraction, model registry, and routing (T5)

The core engine's provider layer went from zero adapters to five real
ones (Anthropic, OpenAI, Google, Ollama, Mock) over this session without
ever getting a dedicated docs chapter — `docs/index.md`'s own text had
named "the provider abstraction, model registry and routing" as
"Chapter 2" since T0, still marked "(in progress)" for the whole of Part
I. With the code now genuinely substantial and stable, this was the
right time to write it, not before.

`docs/providers.md` covers the `Provider` protocol contract itself,
then — the part worth actually teaching, matching this project's
stated purpose that "teaching how to build a multimodal AGI tool is as
much the point as the tool itself" — the real, hard-won differences
between backends that writing four live adapters surfaced: OpenAI's
incremental (fragmented, index-keyed) tool-call argument streaming vs.
Anthropic's already-assembled final message vs. Ollama's complete-per-
chunk calls; Gemini's complete absence of a distinct "made a tool call"
finish reason (the real bug this caused and how it was caught, told as
the teaching example it is); and the three different shapes providers
use for tool-result messages. Closes with the model registry
(`models.yaml`) and `Router.pick()`'s availability/modality-aware
fallback — the literal mechanism behind "absorb a new frontier model =
one YAML entry" — and repeats, in the chapter's own words, why no
OpenAI/Google model entries exist yet (no verified-current catalog data
to add responsibly, the same principle applied when those adapters
shipped).

**Verified rather than assumed correct:** every code sample and every
specific claim (line counts, the `Router.pick()` signature, `GenerateConfig`'s
fields, the `ProviderEvent` union's members, the exact routing.yaml
content) was checked against the current source before writing it into
the chapter, not recalled from memory of writing that code earlier in
this session — caught one real inaccuracy this way (a first draft
claimed `run_benchmark`/`distill` were both "under 100 lines"; `wc -l`
showed `harness.py` is 104, fixed to a claim that's actually true).

`docs/index.md` updated to link the new chapter and stop describing it
as pending. No code changes this entry — pure documentation, verified
against the code it describes rather than written from memory of having
written that code.

**Next:** F1's real (non-toy) training infrastructure, agentic RL (the
last named piece of §3.6e), or Chapter 3 (the agent loop) continuing
the book.

## Docs Chapter 3 — the agent loop, and a real stale-docstring bug caught while writing it

Continuing the book started with Chapter 2. `docs/agent-loop.md` covers
`AgentLoop`'s explicit state machine (`LEGAL`, the transition table as
data rather than scattered `if`/`elif` control flow), concurrent tool
execution gated by exactly one `ConfirmPolicy`, budgets as a clean
`BUDGET_EXCEEDED` terminal state with a `Spend` receipt rather than a
raised exception, and the multimodal-aware routing + opt-in degradation
fallback.

**A real bug caught in the course of verifying every claim against
current source before writing it, not assumed correct from memory:**
`loop.py`'s own module docstring still read "T2 wires *routing*, not
yet *degradation*" — stale since the degradation-fallback entry shipped
earlier this session. Re-reading the actual code (the `degraders`
constructor parameter, the `LookupError` fallback path) before writing
the chapter surfaced the mismatch directly; fixed the docstring in the
same commit rather than writing a chapter that would have repeated a
now-false claim the code itself still made.

**Honestly scoped:** the chapter explicitly names what's *not* built —
the design doc's own architecture section names "subagent fan-out" and
"verifier subagent" patterns, and neither exists in code; `AgentLoop`
today drives exactly one model conversation with one flat tool list.
Stated directly in the chapter rather than silently omitted, matching
this project's discipline of naming gaps rather than letting a reader
assume more coverage than actually exists.

No test changes — one doc-adjacent code fix (a docstring, not
behavior) plus new documentation, verified by re-reading the real
`AgentLoop`/`Budget`/`Tool` source and cross-checking every specific
claim (state names, budget field names, the exact 6 `BUILTIN_TOOLS`,
the CLI's `--auto` wiring) against it, and by confirming the tests named
in the chapter's "Build it yourself" section actually exist with those
names (`test_budget_enforcement`, `test_tool_errors_do_not_kill_the_loop`,
`test_confirmation_gating_deny`, the four `test_degradation_fallback_*`
tests) rather than describing tests that sound plausible.

**Next:** F1's real (non-toy) training infrastructure, agentic RL (the
last named piece of §3.6e), or continuing the book (multimodality,
memory, and packaging for humans are Chapters 4+).

## Docs Chapter 4 — multimodality, and a real silent-content-drop gap found while verifying it

Continuing the book. `docs/multimodal.md` covers the typed
`ContentBlock` vocabulary (`_MediaBlock`'s exactly-one-of-data/path/url
validator, lazy explicit byte resolution, `fetch.py`'s streamed,
byte-capped, scheme-restricted url fetching) and the three real
degraders (image/audio/video), tying back to earlier entries rather
than re-explaining them.

**A real, previously-unnamed gap found while double-checking
`degrade_message`'s "never silently drops a block" claim against
actual provider-adapter code, not assumed to hold everywhere it sounds
like it should:** `DocumentBlock` exists in the type system, and
`models.yaml` even marks `claude-opus-4-8` as supporting `document`
input — but there is no degrader for it, and none of the three provider
adapters' translation functions (`_to_anthropic_message`,
`_to_openai_messages`, `_to_gemini_content`) have a case for it. Each
is a plain `if`/`elif` chain with no `else` — an unhandled block type
is silently absent from the translated request, not raised on. Checking
further surfaced that this isn't only a `DocumentBlock` problem:
`ThinkingBlock` hits the identical silent-drop path on the second and
later turns of any real multi-turn conversation with an extended-
thinking model, since the agent loop appends the full assistant message
— thinking block included — back into `messages` for the next turn,
and Anthropic's own adapter has an existing comment naming this as
untracked-since-T2 work.

**Documented, not silently patched over — and deliberately not "fixed"
in this entry either:** making every adapter raise loudly on an
unhandled block type would change real current behavior (multi-turn
thinking-model conversations would start raising on turn two instead of
silently continuing without the thinking content) in a way that needs
its own careful pass with its own tests, not a rushed fix bundled into
a documentation entry. `degrade_message`'s own "never silently drops a
block" guarantee is real and enforced — precisely at the degradation
layer, which is one step removed from where this gap actually lives
(the lower-level wire-translation step inside each adapter, which turns
out to be a separate place the same principle doesn't currently reach).
Naming the exact boundary of a guarantee, not just that a guarantee
exists, is worth getting right in the docs even when it's not the most
flattering thing to write down.

No test or behavior changes this entry — pure documentation, plus one
real finding named honestly rather than fixed hastily.

**Next:** F1's real (non-toy) training infrastructure, agentic RL, the
real fix for the silent-block-drop gap this chapter named (needs its
own careful pass), or continuing the book (memory and packaging for
humans are Chapters 5+).

## The real fix: provider adapters no longer silently drop unhandled content blocks

Closes the gap the last entry named and deliberately didn't rush a fix
for. The careful pass it needed: distinguish a **deliberate, named**
skip from an **unknown, dangerous** one, rather than treating every
unhandled block type identically.

All three adapters' translation functions (`_to_anthropic_message`,
`_to_openai_messages`, `_to_gemini_content`) now have an explicit
`elif isinstance(b, ThinkingBlock): continue` — a real, intentional
skip, since none of the three backends currently accept a
caller-supplied reasoning trace back on a later turn anyway, so there's
nothing meaningful to round-trip yet — followed by a catch-all
`else: raise ValueError(f"... cannot translate a {type(b).__name__!r} content block ...")`.
`DocumentBlock` now hits that catch-all and raises clearly, instead of
silently vanishing from the outgoing request.

**Verified safe before writing a single test, not assumed safe from
reading the diff:** grepped the whole tree first — `DocumentBlock` is
referenced nowhere outside `content.py` itself (no test, no call site
anywhere constructs one), so making the adapters raise on it changes
zero existing behavior. `ThinkingBlock` is referenced in `mock.py` (as
an *output* type) and in `test_content.py` (pure model round-trip
tests, not provider translation), with no test anywhere asserting a
`ThinkingBlock` survives translation — the explicit skip preserves
today's real (if previously *accidental*) behavior exactly, just makes
it an intentional line of code instead of an implicit gap in an
`if`/`elif` chain. Confirmed empirically too: ran a standalone script
against all three adapters' real translation functions before touching
the test files, watching `DocumentBlock` raise and `ThinkingBlock` drop
cleanly in each.

The distinction encoded in the fix, not just described in prose:
dropping a thinking trace the model can't use anyway is harmless;
silently omitting a document the user actually attached — and letting
the model answer as though it read it — is a materially misleading
response, not a cosmetic gap. One case stays a silent skip because
that's genuinely correct; the other now fails loudly because silence
there was never correct.

6 new tests (2 per adapter — the deliberate-skip case and the
raises-loudly case), all 285 pre-existing tests still pass unchanged
(291 total). `docs/multimodal.md` updated to describe the fix instead
of the open gap.

**Next:** F1's real (non-toy) training infrastructure, agentic RL, or
continuing the book (memory and packaging for humans are Chapters 5+).

## Docs Chapter 5 — memory, and a second real stale docstring found and fixed

Continuing the book (Chapters 2-4 shipped earlier this session).
`docs/memory.md` already existed with solid content from the session
that built semantic memory and session-identity threading — promoted it
to Chapter 5 (retitled, added a "Build it yourself" section) rather
than rewriting from scratch, after re-verifying every claim in it
against current source.

**A second real stale docstring found this way, matching Chapter 3's
pattern exactly:** `sarva.memory.session`'s own module docstring still
said tool-using session persistence (`sarva run --session`) was "NOT
yet wired" — untrue since the `transcript_out` mechanism shipped
earlier this session specifically to solve that problem, and
`test_transcript_out_includes_full_tool_use_round` in
`test_agent.py` already proves it works. Fixed the docstring in the
same commit.

**A real inaccuracy caught by actually running the documented example,
not by reading the code and assuming the prose was right:** the first
draft of the "Build it yourself" section suggested running `sarva chat
"remember that..."` — wrong on two counts, caught only by running it.
First, `sarva chat` is constructed with `tools=[]` in `cli.py` — memory
tools are only reachable via `sarva run`. Second, even after switching
to `sarva run`, the offline Mock provider was assumed likely to at
least attempt something tool-shaped — running it for real showed Mock
just echoes text back and never decides to call a tool on its own,
since it isn't actually intelligent. Both docs claims were fixed to
state plainly that this walkthrough needs a real configured model,
matching the honesty bar the rest of this project holds prose to.

No test changes — two docstring fixes (one in `session.py`, matching
the earlier `loop.py` fix's pattern exactly) plus documentation
corrected against real, observed CLI behavior. 291 tests unaffected.

**Next:** F1's real (non-toy) training infrastructure, agentic RL, or
continuing the book (packaging for humans is Chapter 6+).

## Agentic RL's environment harness — sandboxed coding tasks with real verifiable rewards (§3.6e)

§3.6e's post-training line ends with agentic RL: "RL on long-horizon
tool-use tasks... Includes the RL environment harness (sandboxed coding
tasks with automatic verification)." The full RL training loop (a real
policy-gradient algorithm plus a model-in-the-loop training run) needs
compute this project doesn't have yet — but the harness that loop would
consume is genuinely buildable and testable today, and closes the last
named piece of §3.6e that was still fully unbuilt.

`sarva_foundry.rl.environment.evaluate_submission(task, submitted_code)`
runs a submission plus real test code in a genuinely separate
subprocess (not `exec()` in the caller's process — same isolation
`RunShellTool` already uses in `core/sarva/agent/tools.py`) under a
hard wall-clock timeout, returning a real binary reward: `1.0` if every
assertion in the test code held, `0.0` otherwise. A timeout is scored
as a genuine failure, not a special case — an infinite loop is not a
passing submission.

**"Sandboxed" named honestly in the module's own docstring, not
overclaimed:** subprocess isolation + timeout is real isolation, not a
full security sandbox — submitted code still has the parent process's
filesystem/network permissions. A production system needs a real
container/VM boundary; named directly as real, deferred,
infrastructure-heavy work rather than implied to already be covered.

**Every test scenario runs a real subprocess, nothing mocked** — the
whole point of this module is that the reward comes from actually
running code, so mocking the subprocess would test nothing real.
Verified end to end before writing any test: ran three real scenarios
by hand first (a correct solution scoring 1.0, a wrong one scoring 0.0
with a captured real `AssertionError`, and a genuine 2-second-timeout
infinite loop correctly caught rather than hanging the shell). One
detail worth naming: with `text=True` passed to `subprocess.run`, a
`TimeoutExpired`'s `.stdout`/`.stderr` come back already as `str` (or
`None`), not `bytes` — checked empirically rather than assumed, which
simplified an unnecessary bytes-decode branch the first draft had.

`CODING_TASKS` bundles three small, real, hand-verified tasks (add,
palindrome check, Fibonacci) — same honesty discipline as
`sarva.eval.benchmarks.ARITHMETIC`. Each task's tests are proven
*discriminating*, not just satisfiable: a dedicated test confirms a
plausible-but-wrong solution for every bundled task actually fails, not
just that the correct one passes — catching the failure mode where a
test set is too permissive to mean anything.

`examples/13_rl_coding_environment.py` runs three fixed "policies"
(standing in for what a real agentic-RL rollout would sample from a
model) against the bundled tasks, printing genuinely-earned rewards for
each — correct (1.0), plausible-but-wrong (0.0, real captured error),
and infinite-loop (0.0, caught by the timeout).

7 new tests in `test_rl_environment.py`. 291 → 298 Python tests.

**Next:** F1's real (non-toy) training infrastructure, or the actual RL
training loop (policy-gradient updates consuming this harness's
rewards) — the one piece of agentic RL still genuinely unbuilt.

## GRPO — the actual RL training loop, closing agentic RL's last unbuilt piece

The harness from the last entry computed rewards; nothing consumed
them. `sarva_foundry.train.rl` implements **Group Relative Policy
Optimization** (Shao et al. 2024, DeepSeekMath): sample a group of K
completions per prompt, score each with a real reward function, use
each completion's reward relative to its own *group's* mean —
`(reward - group_mean) / (group_std + eps)` — as the policy-gradient
weight. No separate value network/critic needed, unlike full PPO — the
lighter-weight, teaching-scale-appropriate choice.

`sample_completion` does the rollout under `torch.no_grad()` (sampling
isn't differentiable and doesn't need to be); the gradient comes
entirely from re-evaluating each sampled completion's log-probability
under the *current* model parameters afterward — reusing DPO's
`sequence_logprobs` directly rather than reimplementing it, since
REINFORCE's `E[R · grad_theta log P(action)]` estimator needs exactly
that log-probability term. `build_grpo_batch` → `Trainer.grpo_step`
follows the identical `build_*_batch` → `Trainer.*_step` shape SFT and
DPO already established.

**A real, non-obvious finding from actually running this, caught before
writing a single test:** the first manual verification run trained for
150 steps against an arbitrary target token and saw *zero* measured
reward the entire time — not a bug in the sampling code (confirmed by
inspecting the actual softmax probabilities directly), but a genuine
property of this project's tiny, weight-tied, freshly-initialized
transformers: logits for one dominant token were literally ~19.5 vs.
~5.0 for the next-highest, collapsing sampling to a single token at
>99.99% probability. Checked across ten different random seeds to rule
out a one-off fluke — every single seed showed the same pattern
(max-probability token >98.9% in every case). At `temperature=1.0` this
leaves zero exploration for GRPO to learn from — every group has zero
reward variance, correctly triggering the deliberate no-op path (see
below) every single step, which is why the first attempt showed no
progress: the machinery was working correctly on a task with no
learnable signal, not broken. Raising rollout temperature to 8.0
restored real exploration (verified: top token's probability dropped
from 99.9998% to 25% at that temperature) and is standard real-world RL
fine-tuning practice anyway, not a workaround invented to make a test
pass.

**Two properties tested directly, not just the trainability outcome:**
`test_grpo_step_is_a_deliberate_noop_when_the_group_has_zero_variance`
confirms a zero-variance group produces zero loss and *literally
unchanged model weights* (compared via `state_dict()`, not just "loss
looks like zero") while the step counter still advances — the real
guard the finding above depends on actually existing, not just being
described. `test_grpo_training_increases_the_rewarded_behaviors_probability`
is the end-to-end proof, mirroring DPO's preference-margin test
exactly: measure a target token's real sampling rate before training
(measured, not assumed: 12.5%), train for 300 real steps, measure again
(69.0%) — reproduced identically across repeated runs (confirmed
bit-for-bit deterministic given the same seed before finalizing the
test, not assumed reproducible).

`examples/14_grpo_rl_training.py` runs that exact scenario end to end
and prints the real before/after rates, then prints — labeled
explicitly as illustrative, not executed — exactly how `CODING_TASKS`/
`evaluate_submission` (example 13's harness) would plug in as the
reward function for real coding-task RL: the GRPO loop itself is
unchanged, only the reward function differs. Deliberately not run for
real against actual code tasks: a 2-layer, 16-dim toy transformer
genuinely cannot learn to write working Python from sparse
code-execution rewards in a short demo, and fabricating that result
would violate the same honesty principle this project has held to all
session.

7 new tests in `test_rl_training.py`. 298 → 305 Python tests. §3.6e's
full post-training line — SFT, DPO, and agentic RL (both the
environment harness and the GRPO training loop) — is now built, at the
scale a laptop can actually run and verify.

**Next:** F1's real (non-toy) training infrastructure (needs real
multi-GPU compute this environment doesn't have), or continuing the
book (Chapter 6: packaging for humans).

## The foundry provider adapter — a trained checkpoint becomes a real, routable model

Closes a gap named explicitly in two earlier entries, not invented fresh:
the design doc's own repo-structure diagram lists `providers/foundry.py`
alongside `anthropic.py`/`openai.py`/etc., and the eval-harness entry
said outright "the moment §3.1's planned foundry adapter exists... it
becomes gradable by this same harness with zero changes — not built
yet, named as real deferred work." Every other foundry chapter (SFT,
DPO, GRPO) trains checkpoints that had nowhere to go afterward; this is
where one comes back.

`sarva.providers.foundry_provider.FoundryProvider` implements the same
`Provider` protocol every frontier adapter implements. A checkpoint
"bundle" is a directory with `model.pt` (a real `Trainer.save_checkpoint`
output), `tokenizer.json`, and `config.json` (the flat `TransformerConfig`
fields needed to reconstruct the model's shape before loading weights).
**Honestly scoped, not silently incomplete:** MoE and long-context
RoPE-scaling configs aren't serialized yet — `save_checkpoint_bundle`
raises `NotImplementedError` rather than writing a bundle that would
silently reload as a plain dense/unscaled model mismatched from what was
actually trained.

**The dependency boundary this had to respect, not just work around:**
`core` and `sarva_foundry` have been deliberately dependency-disjoint
since the distillation glue script — `core`'s dependencies are
lightweight API clients, `sarva_foundry`'s are `torch`/`numpy`, and most
Sarva installs shouldn't need to pull in torch. `foundry_provider.py`
imports torch/`sarva_foundry` lazily, function by function, so importing
the *module* always succeeds even on a plain-core install; only actually
loading or running a checkpoint needs the new optional `sarva[foundry]`
extra (added to `core/pyproject.toml`), and does so with a clear,
actionable `ImportError` naming the install command, not a confusing
failure somewhere inside torch's own import machinery.

**Wired into `sarva.runtime` the same way Ollama already is, not as a
special case:** a new `_foundry_extra_installed()` cheap probe (mirroring
`ollama_reachable()`) gates both `build_router()` (registry availability)
and `build_providers()` (actual provider construction) from one source of
truth, so a foundry checkpoint is never marked available with no provider
able to serve it. Unlike frontier models, checkpoints aren't declared
statically in `models.yaml` — the set is entirely per-install, so they're
discovered from `SARVA_FOUNDRY_CHECKPOINTS` and added to the registry at
runtime via a new `Registry.register()` method, never a default routing
candidate, reachable only through an explicit `--model foundry/<name>`
override.

**A real, named limitation the docs state plainly rather than gloss
over:** no chat template is applied — the prompt is just the concatenated
text of system + every message, no role tags — because that's exactly how
`examples/10_sft_toy_assistant.py` and the SFT chapter's own tests train
(raw prompt text, no role tags); a checkpoint trained some other way
would need this adapter to match it. Streaming is coarse (one
`TextDeltaEvent` for the whole completion, not per-token) since there's
no wire protocol to translate for a fully local synchronous model, and
there's no batching/KV-cache reuse — precisely the gap a real foundry
inference server (§3.6f, separate deferred scope) would close.

**Verified beyond the conformance suite:** built a real toy bundle by
hand and ran it through the actual CLI, not just pytest — `sarva models`
correctly lists `foundry/toy` as `[x]` available, and `sarva eval --model
foundry/toy` runs the real arithmetic benchmark against it and scores
0%, the honest result for an untrained toy checkpoint (same
no-fabrication discipline the eval harness itself established for the
zero-config Mock provider). 10 new tests in `test_foundry_provider.py`
(save/load round-trip on real weights, the MoE-config refusal, bundle
discovery, `Registry.register`, a real end-to-end `generate()` call, an
unknown-model-id error path, and the full `runtime.py` wiring). 305 →
315 Python tests. New docs chapter: `docs/foundry/inference.md`.

**Next:** the foundry inference server (§3.6f: batched inference +
KV-cache reuse around `DecoderOnlyTransformer`), the foundry recipes
directory (§3.6h: named/costed configs starting at the 125M laptop
scale), or continuing the book (Chapter 6: packaging for humans).

## KV-cache — real incremental decoding, and a real bug it took empirical verification to catch

Closes the KV-cache half of the gap the foundry provider's own docstring
named ("no batching, no KV-cache reuse across calls — one naive forward
pass per generated token"). `sarva_foundry.model.kv_cache.KVCache`
pre-allocates a `(n_layers, batch, n_kv_heads, max_seq_len, head_dim)`
buffer per key/value; `GroupedQueryAttention.forward` and
`DecoderOnlyTransformer.forward` both gained an optional `cache`
parameter (`None` — the default, and every call site before this
parameter existed — is exactly the original unchanged behavior, pinned
by a dedicated regression test). With a cache, `token_ids` means "the
NEW tokens since the cache was last advanced," and only those tokens'
key/value get freshly projected — every previously-generated position's
key/value is read straight out of the buffer instead of being
recomputed. `sarva_foundry.inference.generate_with_cache` is the
KV-cached counterpart to `sarva_foundry.train.rl.sample_completion` —
same contract (same greedy/temperature semantics, same "returns only the
new tokens" behavior), deliberately kept a drop-in match so the two can
be compared token-for-token as a correctness proof.

**A real bug, caught by comparing against known-correct output, not by
re-reading documentation harder:** the first version leaned on
`F.scaled_dot_product_attention(..., is_causal=True)` even when the new
query length was shorter than the cached key length, on the assumption
that `is_causal` bottom-right-aligns a shorter query against a longer
key the way several other inference codebases' cache implementations do.
Wrong for this PyTorch version — confirmed empirically: cached generation
diverged from known-correct full-recompute generation starting at the
very first cached token, with differences too large (multiple full units
on raw logits) to be floating-point noise. Isolated systematically, not
guessed at: verified the cache's stored key/value buffer content matched
a manual full recomputation exactly (it did — the storage itself was
never the bug), then verified the *attention output* diverged even
though its inputs were provably correct, which narrowed the bug to the
masking call itself. Manually testing `is_causal=True` vs an explicit
mask vs `is_causal=False` against a hand-built reference for both a
single-new-token case and a multi-new-token case (`start_pos=5`, 2 new
queries) confirmed the actual fix: build the causal mask explicitly via
`torch.ones(seq_len, total_len, dtype=torch.bool).tril(diagonal=start_pos)`
— row `i` (of the new tokens, at absolute position `start_pos + i`)
attends to every key at absolute position `<= start_pos + i`. This
subsumes the no-cache case exactly (`start_pos=0`, query length equals
key length reduces to the ordinary causal mask), so it's one code path
handling both, not a cache-specific special case bolted on.

**Tests pin the property that actually matters, not just shapes:**
`test_forward_with_cache_matches_full_recompute_across_incremental_steps`
compares cached, step-by-step logits against full-recompute logits at
*every* incremental step (not just the first), and
`test_generate_with_cache_matches_naive_greedy_generation_token_for_token`
proves `generate_with_cache` and `sample_completion` produce the
identical token sequence under greedy decoding — the actual guarantee a
caller depends on. 8 new tests in `test_kv_cache.py`.

`sarva.providers.foundry_provider.FoundryProvider.generate` now calls
`generate_with_cache` instead of the naive `sample_completion`, closing
the loop for the adapter that motivated this work.
`examples/15_kv_cache_inference.py` runs both generation paths on a
128-dim, 4-layer model for 200 tokens and prints real measured
wall-clock numbers — confirmed identical token output either way, ~2.4x
faster cached on the machine this was verified on (honestly reported as
hardware-dependent, not asserted as a universal number). 8 new tests,
315 → 323 Python tests. `docs/foundry/inference.md` updated with the
full story, including the `is_causal` bug.

**Next:** batching multiple concurrent requests (the other half of
§3.6f's inference-server gap), the foundry recipes directory (§3.6h:
named/costed configs starting at the 125M laptop scale), or continuing
the book (Chapter 6: packaging for humans).

## Foundry recipes — named, costed configs, and a real OOM confirming why they're estimated, not instantiated

Closes §3.6h, named directly in the design doc's own repo-structure
diagram since T0: `foundry/recipes/  # named, costed configs: laptop-125M
-> 1B -> 7B -> 70B`. `sarva_foundry.recipes.Recipe` bundles a real
`TransformerConfig` with the training hyperparameters that go with it
(token budget, batch size, LR, warmup) plus `compute_estimate()` — a
real FLOPs-based estimate using the standard `6*N*D` dense-transformer
training-FLOPs approximation (Kaplan et al. 2020 / Hoffmann et al.
2022), not a fabricated number.

**The design decision that matters most here:** `compute_estimate`
takes hardware throughput (FLOP/s) and price ($/hour) as explicit
caller-supplied arguments rather than hardcoding a specific GPU or
price — the same no-fabrication practice that kept unverified
OpenAI/Google pricing out of `models.yaml`. "Costed" means "you can
compute a real cost from real inputs," not "we assert a fixed dollar
figure that will be stale by the time anyone reads it."

**`param_count()` doesn't instantiate the model to count its
parameters, and there's a real reason why, confirmed empirically while
building this:** constructing an actual ~70B-parameter
`DecoderOnlyTransformer` was tried directly and got killed by the OS for
memory use on this laptop — not a hypothetical concern, a real crash.
`param_count()` computes the same number analytically instead, straight
from the architecture's own weight-matrix shapes (attention's four
projections, SwiGLU's three matrices, the tied embedding table). This
formula is **verified exact, not assumed**: `tests/foundry/
test_recipes.py` instantiates real models at the two scales small
enough to build (`LAPTOP_125M`, 125,264,640 params; `SCALE_1B`,
1,057,581,056 params) and confirms `param_count()` matches
`.num_parameters()` bit-for-bit at both — since it's the same
architecture code, not a fitted approximation, that exactness holds at
the two larger, never-instantiated scales too (`SCALE_7B`, 5,802,037,248;
`SCALE_70B`, 55,628,275,712).

**Labels are honest about the field's own looseness, not inflated:**
every parameter count is the real computed number, not rounded to hit
its label exactly — the same convention published models already use
(Llama-2-7B is actually 6.7B; Mistral-7B is 7.24B). `SCALE_70B` sits
further from its label (55.6B) than the others because this project's
plain 2/3× SwiGLU hidden-dim rule differs from Llama-2-70B's own custom
FFN multiplier — reported plainly rather than hand-tuned to hit exactly
70B.

`examples/16_foundry_recipes.py` prints every recipe's real parameter
count and compute estimates under two explicitly-labeled **illustrative**
hardware profiles (never claimed as current, verified GPU pricing), then
does something the printed table alone can't prove: runs `LAPTOP_125M`'s
real architecture for a few real training steps on this machine,
measures actual wall-clock tokens/sec, converts that into a real FLOP/s
figure via the same `6*N*D` formula, and shows what `compute_estimate`
predicts from *this machine's own measured speed* — a genuine
correlation check between the formula and reality, not two disconnected
numbers. 7 new tests, 323 → 330 Python tests. New docs chapter:
`docs/foundry/recipes.md`.

**Next:** batching multiple concurrent requests (§3.6f's remaining
inference-server gap), F1's real distributed training infrastructure
(needs real multi-node compute this environment doesn't have), or
continuing the book (Chapter 6: packaging for humans).

## Packaging verification — a real `pip install`, not just the dev workspace, wired into CI

Every check in CI up to this point ran through `uv run`, which uses this
repo's own dev workspace venv directly — never actually exercising the
path a real end user takes (`pip install sarva`), and never proving the
published package *metadata* (entry points, dependency list, which files
get bundled) is even correct. The README's own "CLI works end to end"
claim had never been checked against an actual built wheel outside this
workspace. Verified by hand first, the way every CI addition this
session has been: built both wheels (`uv build --all-packages`), created
a genuinely separate venv, installed them, and ran `sarva chat` and a
`sarva[foundry]` import check — all worked, but not instantly.

**A real environmental finding along the way, not a bug:** the very
first attempt appeared to hang and hit a 2-minute timeout. Bisected
systematically rather than assumed broken: isolated it down to `import
torch` alone, in a brand-new venv, taking ~19 seconds on its own — a
SECOND `import torch` in the same venv dropped to well under a second.
This is macOS Gatekeeper verifying the code-signing of freshly-extracted
dylibs on their first load, a known real cost of installing torch fresh,
not a hang and not anything wrong with this project's code. Documented
directly in the new CI step's own comment so a future cold-runner CI run
taking a couple of minutes on this step doesn't look like a regression.

Added a new step to the existing `core` CI job (reusing its already-warm
`uv` cache rather than a separate job that would re-download torch from
scratch): build both wheels, install into a clean venv, run a real
`sarva chat` smoke test, and confirm the `sarva[foundry]` extra imports
cleanly. Also fixed the stale bits this surfaced: the README's Status
section hadn't been touched since before SFT/DPO/GRPO/the agentic RL
harness/the foundry provider adapter/KV-cache/recipes shipped — updated
to describe what's actually built now, plus a new Quickstart snippet
showing the verified `pip install`-from-wheel path directly (not just
the dev-workspace `uv sync` flow). No test count change (a CI/docs
milestone, not new library code) — the packaging step itself is the new
verification. `docs/foundry/inference.md`'s `sarva[foundry]` framing and
the wheel-based install path now agree with each other in practice, not
just on paper.

**Next:** batching multiple concurrent requests (§3.6f's remaining
inference-server gap), F1's real distributed training infrastructure
(needs real multi-node compute this environment doesn't have), or
continuing the book (Chapter 6: packaging for humans).

## Docs Chapter 6 — packaging for humans: the CLI, the server, the desktop app

Closes `docs/index.md`'s own long-standing placeholder ("Chapters 6+ —
packaging for humans — land as their own chapters get written"). Three
modules had zero book coverage until now: `sarva.cli`, `sarva.server.
app`, and `apps/desktop/src-tauri`. `docs/packaging.md` covers all
three, verified against current source before writing a word — the same
discipline that caught two real stale docstrings in earlier chapters
(`loop.py`, `session.py`).

**What got directly verified, not assumed, before publishing:** the
server's tool-confirmation handshake (`receive_json()` → `{"approved":
bool}`, matched line-for-line against `apps/desktop/src/App.tsx`'s
client-side `respondToConfirmation` — the same handshake described in
`app.py`'s own docstring is genuinely implemented on both ends, not
just documented on one); the desktop app's sidecar-kill logic (`#[cfg
(unix)]` gates only the `pgrep -P <pid>` grandchild-reaping step needed
because PyInstaller's onefile bootloader forks a real grandchild
process — confirmed Windows genuinely has no equivalent yet, a real,
still-open gap stated plainly rather than glossed over); and the
`core/sarva/server/static/` -> `apps/desktop/dist/` relationship (a
literal checked-in copy via `scripts/build-web.sh`, a manual step, not
CI-automated — CI only checks the copy hasn't gone stale).

**Session persistence gets one precise, previously-undocumented detail
right:** both `sarva chat --session` and `sarva run --session` only
save the transcript if the run actually reached `done` — a failed,
budget-exhausted, or cancelled run is never persisted, so a session
file only ever reflects turns that genuinely completed. Also documents
the real, working cross-platform release bundles (`.dmg`/`.msi`/
`.exe`/`.AppImage`/`.deb` via `release-bundle.yml`'s three-OS matrix,
draft-only GitHub Releases on tag push) alongside the honest, still-open
gap named directly in the workflow's own name — "Release bundle
(unsigned)" — no code signing or notarization yet.

No test or code changes — pure documentation, verified line-by-line
against `cli.py`, `server/app.py`, `src-tauri/src/lib.rs`,
`release-bundle.yml`, `App.tsx`, and `build-web.sh`. 330 tests
unaffected. New chapter: `docs/packaging.md`; `docs/index.md`/
`mkdocs.yml` nav updated to link it instead of describing it as
pending.

**Next:** batching multiple concurrent requests (§3.6f's remaining
inference-server gap), F1's real distributed training infrastructure
(needs real multi-node compute this environment doesn't have), or a
first pass at code-signing/notarization for the desktop release bundles
(needs a real Apple/Microsoft signing identity this environment doesn't
have — likely stays a named, deferred gap).

## `DocumentToTextDegrader` — the fourth degrader, closing the last uncovered modality

`Degrader`'s own motivating docstring example in `content.py` has said
"video -> [image frames + text transcript]" since T0, alongside the
same design intent for documents — but grepping the tree before
starting confirmed `core/sarva/multimodal/degraders/` had exactly three
converters (image/audio/video) and zero for `DocumentBlock`, which has
been typed since T0 (`models.yaml` even marks `claude-opus-4-8` as
accepting document input). A `DocumentBlock` sent toward a text-only
model had no fallback path at all — the one modality where
`UnsupportedModalityError` was the *only* possible outcome, unlike every
other modality.

Same honesty principle as the other three: real extracted text where a
real extractor exists, never a fabricated summary. `pypdf` (pure
Python, MIT) is the new dependency for real per-page PDF text
extraction — the same "commodity substrate" tier as Pillow (images) and
PyAV (video), not a black box in the sense this project's own "no black
boxes" principle actually means. Plain-text-adjacent media types
(`text/plain`, `text/markdown`, `text/csv`, `text/html`,
`application/json`) need no library at all — a UTF-8 decode of the
block's own bytes *is* the real content. Extracted text is capped at
20,000 characters, the corpus pipeline's length-filter philosophy
applied here (an attached 300-page PDF shouldn't consume a target
model's whole context window on its own), with the degraded message
stating honestly when and how much was cut.

**A scanned/image-only PDF (no text layer) degrades the same way a read
error does** — both mean "nothing could be extracted," matching the
audio degrader's own framing: an undecodable format is the *expected*
real case for a converter handling arbitrary caller-supplied bytes, not
a bug. `.docx` and other binary office formats get the same
declared-metadata-only fallback, named directly in the module's own
docstring as real, deferred scope — a second heavy dependency isn't
justified by one format the way `pypdf` is justified by PDF being
ubiquitous.

**Test fixtures are real, not fabricated, matching the video degrader's
own precedent** (which encodes a real tiny mp4 with PyAV itself rather
than shipping a binary fixture file): `_minimal_pdf_bytes()` hand-builds
a genuinely valid single-page PDF with correctly computed byte offsets
(not relying on `pypdf`'s xref-recovery leniency for a malformed one),
so the tests prove a real write-bytes-then-extract round trip. One
self-caught test bug along the way: an early truncation-test assertion
counted every literal `"x"` character across the *entire* formatted
output message (including incidental ones in words like "text/plain"),
overcounting by exactly the number of stray matches in the header —
fixed to check the actual extracted body slice directly instead of a
naive substring count.

Also fixed a genuinely stale comment this surfaced: all three provider
adapters' wire-translation `else` branches still said `DocumentBlock`
"has neither a degrader nor adapter support yet" — true when written,
false now that a degrader exists; reworded to describe precisely when
that `else` branch is still reachable (degradation skipped, or a
model's registry entry claims document support no adapter has wire-level
code for) rather than implying the gap is still total. 6 new tests
(one existing `default_degraders` coverage test updated for the new
modality), 336 Python tests. `docs/multimodal.md` updated with the new
degrader's own section and a corrected "Build it yourself" walkthrough.

**Next:** batching multiple concurrent requests (§3.6f's remaining
inference-server gap), F1's real distributed training infrastructure
(needs real multi-node compute this environment doesn't have), or a
first pass at code-signing/notarization for the desktop release bundles
(needs a real Apple/Microsoft signing identity this environment doesn't
have — likely stays a named, deferred gap).

## `sarva doctor` — the CLI command named in the repo-structure diagram since T0, never built until now

The design doc's own repo-structure diagram lists `cli.py # ... chat,
run, serve, models, doctor` — confirmed by grep that `doctor` never
existed as an actual `@app.command()` in `cli.py`, unlike every other
name in that list.

`sarva.runtime.run_diagnostics()` is the backing logic, deliberately
living in the same module as `build_router`/`build_providers` and
reading the exact same env vars, calling the exact same
`ollama_reachable`/`_has_google_key`/`_foundry_extra_installed` helpers
— so the diagnostic report can never silently drift out of sync with
what "available" actually means elsewhere in that file, the same
reason `sarva models` already lives right next to the registry it
reports on. Five checks: Anthropic/OpenAI/Google API keys, Ollama
reachability, and the foundry extra + any discovered checkpoint
bundles. `sarva doctor` (the CLI command) adds Python/platform info and
a sixth check — whether the web UI's static build exists for `sarva
serve` — printed alongside.

**A real bug caught by actually running the command, not just reading
the code:** the first version's foundry-not-configured message read
"sarva installed, but SARVA_FOUNDRY_CHECKPOINTS is unset" — the literal
substring `[foundry]` had vanished. Rich's `console.print()` treats
square brackets as markup syntax; `sarva[foundry]` looked like an
(invalid) style tag and got silently swallowed rather than erroring.
Fixed by wrapping every dynamic detail string in `rich.markup.escape()`
before printing — the same discipline this file already applies to raw
model output elsewhere in `cli.py`, just missed here on the first pass.
A dedicated regression test (`test_doctor_cli_never_swallows_bracketed_
text_as_rich_markup`) pins that `"sarva[foundry]"` actually appears in
the printed output, not just that the command exits zero.

**Framing matters, stated directly in both the code and the docs:**
`ok=False` means "not configured," not "broken" — every check here is a
genuinely optional provider, and a fresh, zero-config install is
expected to fail most of them and still work fine via the Mock
provider. 8 new tests (`test_doctor.py`, using `typer.testing.CliRunner`
for the first time in this codebase), 336 → 344 Python tests.
`docs/packaging.md` updated to describe the new command (and its own
command count corrected from seven to eight).

**Next:** batching multiple concurrent requests (§3.6f's remaining
inference-server gap), F1's real distributed training infrastructure
(needs real multi-node compute this environment doesn't have), reasoning/
thinking-token training (§3.6a names it; nothing in `foundry/` builds it
yet), or MCP's HTTP/SSE transport (the client docstring names stdio-only
as real, deferred scope).

## MCP's Streamable HTTP transport — the gap the client's own docstring named

`sarva.mcp_client`'s module docstring has said "HTTP/SSE transports are
real, deferred scope" since the day the stdio transport shipped. This
closes it with `connect_http_mcp_server`, speaking Streamable HTTP — MCP
spec revision 2025-03-26's current standard HTTP transport, superseding
the older separate SSE transport (which the `mcp` SDK still ships as
`mcp.client.sse` for servers that haven't moved off it, deliberately not
wired up here: "current standard transport," not "every historical
variant"). Nothing downstream — `list_mcp_tools`, `McpToolAdapter`, the
agent loop — knows or cares which transport a given `ClientSession` came
from; both connectors just hand one back.

**A real API-version gotcha caught immediately, not shipped:** the first
version used `mcp.client.streamable_http.streamablehttp_client(url,
headers=...)`, which worked but emitted a real `DeprecationWarning`
("Use `streamable_http_client` instead") the moment tests ran — the
installed SDK version (1.28.1) has moved to a form that takes an
explicit `httpx.AsyncClient` instead of `headers`/`timeout` kwargs
directly. Fixed by building that client via the SDK's own `create_mcp_
http_client` helper (30s timeout, redirects followed — its own
documented defaults), imported from the same `mcp.client.streamable_
http` module rather than its private `mcp.shared._httpx_utils` origin,
since the public module already re-exports it.

Wired all the way to the CLI, not left as a library-only capability
(the exact gap named in an earlier degradation-fallback entry this
session — "fully built and fully tested but completely unreachable by
any real user"): `--mcp-server` now dispatches by shape — an
`http://`/`https://` value connects over HTTP, anything else is
shell-split and run as a stdio command — so both transports mix freely
in one `sarva run` invocation. Verified against a real running server,
not just pytest: started `mcp_http_echo_server.py` by hand and ran
`sarva run "..." --mcp-server http://127.0.0.1:.../mcp --auto`, watching
the real HTTP session negotiate, list tools, and cleanly terminate
(`DELETE /mcp` → `200 OK`) in the server's own logs.

**Test fixtures mirror the stdio precedent exactly:**
`tests/fixtures/mcp_http_echo_server.py` is the same two tools
(`echo`/`fail`) as the stdio fixture, launched as a genuine subprocess
serving real MCP-over-HTTP on a real, OS-assigned free port (picked by
binding to port 0 and reading the result back — avoids CI port
collisions from a hardcoded value) rather than a mock of the protocol.
`test_mcp_client_http.py` mirrors every one of `test_mcp_client.py`'s
cases (tool listing, a real round trip, real error propagation, a real
`AgentLoop.run()`), plus a dedicated header-passthrough test and a
plain-HTTP reachability check that isolates "server never started" from
"MCP handshake failed" as distinct failure modes. 6 new tests, 344 → 350
Python tests. `docs/mcp.md` rewritten for both transports.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), reasoning/thinking-token training
(§3.6a names it; nothing in `foundry/` builds it yet), or a first pass
at code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Reasoning-token training — SFT cold start + GRPO, and two real bugs caught building it

§3.6a names "reasoning/thinking-token support... o1/R1-class" directly,
citing DeepSeek-R1-class open recipes — confirmed by grep before
starting that it was the one item on that list with zero code anywhere
in `foundry/`. `sarva_foundry.train.reasoning` closes it, reusing SFT
and GRPO completely unchanged — the only new code is a reward function
(`format_reward` + `answer_reward`, summed as `reasoning_reward`,
weighted 0.3/0.7 toward correctness, mirroring DeepSeek-R1's own reward
design). The two-stage recipe (cold-start SFT teaches the
`<think>...</think>` format, GRPO then refines answer accuracy on top)
isn't an arbitrary choice — it's the R1 paper's own published finding:
pure RL from a base model ("R1-Zero") produced real format/readability
problems in their ablation, which is exactly why R1's final recipe adds
a cold-start stage before RL.

**A real reward-hacking exploit, caught empirically while building the
example script, not shipped:** the first version of both reward
functions matched only the *first* `</think>` in a completion. GRPO
training discovered padding completions with many extra `</think>`
copies inflated the answer reward's loose "contains" check with repeated
copies of the target digit, without genuinely answering correctly — a
training run's climbing group-mean-reward curve looked like real
learning right up until the trained model's actual greedy output was
inspected directly and turned out to be degenerate spam. Both reward
functions now require **exactly one** `<think>`/`</think>` pair; the
literal degenerate completion that broke them is pinned as a permanent
regression test. Retraining with the fixed reward from scratch produced
a genuinely well-formed, prompt-differentiated result instead (31% → 56%
answer accuracy on single-digit addition, real numbers from real
generated text checked against the real digit sum).

**A second, independent real bug this surfaced, in the tokenizer
itself:** decoding a genuinely undertrained model's output crashed with
`UnicodeDecodeError` — `ByteLevelBPETokenizer.decode()` had only ever
been exercised on ids from `encode()` on real text (always valid UTF-8
by construction) in 291 prior tests, never on arbitrary model-generated
ids, which carry no such guarantee. Fixed with `errors="replace"` in the
final UTF-8 decode step (standard practice for any real tokenizer used
in inference/RL rollout), with two dedicated tests: one confirming
invalid bytes get replaced instead of raising, one confirming valid text
around the invalid bytes still round-trips exactly (the fix doesn't
collaterally damage legitimate decoding).

`examples/17_reasoning_token_training.py` runs the real two-stage recipe
on single-digit addition, with real printed before/after rates. 21 new
tests across three files (`test_reasoning.py`, `test_reasoning_
training.py`, plus 2 in `test_tokenizer.py`), 350 → 371 Python tests.
`docs/foundry/training.md` gets a new section covering the recipe, both
bugs, and the real numbers. §3.6a's architecture *and* training-recipe
lists have no remaining named gap.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), or a first pass at code-signing/
notarization for the desktop release bundles (needs a real signing
identity this environment doesn't have — likely stays deferred).

## The ablation harness — trustworthy architecture comparisons, closing §3's last named gap

Design doc §3: "architecture is composable [via `TransformerConfig`]...
+ an ablation harness so researchers can test *new* ideas at small
scale with trustworthy comparisons. This is what 'advance LLMs, not
just train them' means concretely." Confirmed by grep before starting:
two OTHER docstrings in this codebase cite published third-party
ablations (LLaVA-1.5's connector, the SwiGLU-vs-ReLU comparison), but
Sarva had none of its own — every "the architecture is composable"
claim had been asserted, never actually exercised as a real head-to-head
comparison.

`sarva_foundry.ablation.run_ablation` takes "trustworthy" — the design
doc's own word — literally rather than as filler. Two real confounds a
naive single-run comparison misses: **identical data in identical
order** across every arm (proven directly, not just claimed — two arms
given the same config and the same seed produce bit-identical final
losses, the only way that's possible if they really did see identical
data throughout), and **multiple seeds per arm** (three by default),
reporting mean and standard deviation rather than one point estimate
treated as ground truth.

**Honestly scoped, not overclaimed:** `is_difference_trustworthy`
reports one real, defensible signal — mean final losses differing by
more than their combined standard deviation — explicitly NOT a formal
p-value or hypothesis test. A genuine Welch's t-test needs a real
t-distribution CDF (an incomplete beta function this project hasn't
built), and this project doesn't approximate statistics it hasn't
actually implemented, the same discipline that kept unverified GPU
pricing and OpenAI/Google model entries out of other parts of the
codebase.

Verified with two real comparisons, not one convenient demo:
`examples/18_ablation_harness.py` runs a **positive control** (an
8-dim/1-layer model vs. a 48-dim/2-layer one — correctly flagged
trustworthy, the loss gap far exceeding cross-seed noise) alongside a
**genuine architecture question** (dense SwiGLU vs. MoE feedforward,
the two feedforward options `TransformerConfig` already composes
between) — which, at this toy scale and training budget, the harness
honestly reports as NOT trustworthy (both essentially memorize the
small corpus). Showing a real "no significant difference" result
alongside the positive control is deliberate: a harness that always
finds a winner would be much less trustworthy than one that can say so
honestly.

7 new tests (`test_ablation.py`), 371 → 378 Python tests. New docs
chapter: `docs/foundry/ablation.md`. §3's architecture-and-ablation
sentence has no remaining unbuilt half.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), or a first pass at code-signing/
notarization for the desktop release bundles (needs a real signing
identity this environment doesn't have — likely stays deferred).

## First-run guided setup — a real gap between a promise and what shipped, closed

T4's own definition of done (and the README's own quickstart text) have
said since T4: "guided first-run offers (a) 'Free & private' → pulls a
local model, or (b) 'Frontier quality' → paste an API key" — but
`apps/desktop/src/App.tsx` was a bare chat window with no such flow at
all, confirmed by grepping for any onboarding/first-run logic and
finding none. The status line has claimed "T4... desktop app done" all
session; this was a real, honest gap between that claim and what a
non-technical user double-clicking the built app actually got.

**The real missing piece wasn't UI, it was persistence.** Every
provider SDK (`anthropic.AsyncAnthropic()`, `openai.AsyncOpenAI()`,
`genai.Client()`) reads its API key from real process environment
variables internally — a key entered once anywhere had nowhere to
survive past that one process's lifetime. New `sarva.config` module:
`~/.sarva/config.json` (the same `~/.sarva/` home session storage
already uses), one deliberate, tested precedence rule (a real env var
always wins over a saved value, so an explicitly exported shell key is
never silently overridden by a stale file). `sarva.runtime.get_env()`
replaces every direct `os.environ.get(...)` call for the four
provider-key names.

**A real correctness gap caught before it shipped, not after:** simply
detecting a config-file key as "available" wasn't enough — the raw SDK
constructors only ever look at `os.environ` themselves, so a
config-file-only key would pass every availability check and then fail
to authenticate the moment a real request went out. Fixed by having
`build_providers()` construct every SDK client with an *explicit*
`api_key=...`, verified directly by checking the constructed client's
own `.api_key` attribute in a dedicated test, not just that
`build_providers()` runs without crashing.

Two new server endpoints back the UI: `GET /doctor` (the same
`run_diagnostics()` `sarva doctor` already uses, as JSON — can never
drift from what the CLI reports) and `POST /config` (persists whichever
keys the caller supplies, returns the fresh `/doctor` result in the same
round trip so the caller can confirm the key it just saved is actually
recognized). `Onboarding.tsx` polls `/doctor` on mount and completes
immediately if anything (including a reachable Ollama) is already
configured; otherwise it shows exactly the two documented choices plus
an honest "Skip for now" escape hatch remembered in `localStorage`.

**A real test-environment quirk found and handled defensively, not
just worked around:** `window.localStorage` turned out to be
unavailable in this project's own Vitest/jsdom test environment (Node's
`--localstorage-file` flag wasn't set) — would have crashed the whole
app on mount. Wrapped both the read and write sides in `try`/`catch`,
which is the right call for a real desktop webview context too (some
embedded/privacy-mode contexts restrict storage), not just a test
workaround; a missing localStorage now just means onboarding re-shows
next launch instead of crashing.

**Verified beyond the test suites:** a real `sarva serve` process, hit
with real `curl` — `POST /config` with a test key, confirmed
`~/.sarva/config.json` genuinely existed on disk with the right content
(then cleaned up), and the following `GET /doctor` call reflected it as
configured. `apps/desktop`'s full production build (`npm run build`,
`tsc -b`) ran for real. Honestly scoped: this environment has no GUI/display
access, so the onboarding screen's actual pixels were never visually
inspected in a real browser — the 20 new component tests (`vitest`,
`@testing-library/react`, real fetch mocking) are real behavioral
verification, not a substitute claimed to be equivalent to visual
inspection.

15 new Python tests (`test_config.py` + 5 new `test_server.py` cases),
20 new TypeScript tests (`Onboarding.test.tsx`), 11 existing `App.test.tsx`
cases updated to accommodate the new mount-time `/doctor` check. 389
Python tests total.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), or a first pass at code-signing/
notarization for the desktop release bundles (needs a real signing
identity this environment doesn't have — likely stays deferred).

## Quantization — closing §3.6f's third named inference-stack gap

§3.6f names "KV-cache, paged attention, quantization" together as
inference/serving scope. KV-cache shipped a few milestones back;
batching/paged attention stays deliberately deferred (real correctness
risk in code this project has been careful around); quantization was
the one piece left that's genuinely separable from both — confirmed
before starting by `grep -rn "quantiz" foundry/` returning zero hits.

New `sarva_foundry/quantization.py`: real per-output-channel int8
round-to-nearest for every `nn.Linear` — one scale (`max(|row|)/127`)
per row rather than one for the whole matrix, since different output
channels can have very different magnitudes. `quantize_model()` walks
`named_modules()` for every `nn.Linear`; `apply_quantized_weights()`
mutates a live model's weights in place with the dequantized
(round-tripped) values, for measuring quantization's real accuracy cost
on a real forward pass.

**A real open question, checked rather than assumed:** the transformer
ties `lm_head.weight` to `tok_embeddings.weight` (the literal same
`Parameter` object). Does quantizing `lm_head` in isolation and
overwriting it via `.data = ...` break that tie? Wrote a scratch script
first to check empirically before writing the real test: it doesn't —
since both names reference the identical `Parameter` object, mutating
one's `.data` necessarily mutates the other's too. Pinned directly in
`test_apply_quantized_weights_preserves_tied_lm_head_and_embedding_identity`
rather than left as an assumption about how weight tying happens to be
implemented.

**Honestly scoped, the same line the KV-cache chapter already draws:**
this reduces storage (~3.5–4x, real measured byte counts on int8 weights
plus the small per-channel float32 scale vector, not an assumed ratio)
and measures real accuracy cost. It does not speed up compute or shrink
a *running* model's memory — `dequantize()` converts back to float32
before every matmul, matching the same "commodity substrate" line this
project already draws around `torch.matmul` itself. A real int8-serving
path (weights kept compact end-to-end, dequantizing only the one layer
currently executing) is separate, deferred serving-optimization work.

9 new tests in `tests/foundry/test_quantization.py`: a *provable*
round-trip error bound (every element within `scale/2`, not just
"small"), an all-zero-output-row edge case (real, not hypothetical — a
trained weight row genuinely can end up all-zero, and the naive formula
would divide by zero), the weight-tying interaction above, a
"quantization is not a no-op" forward-pass check, and — mirroring the
ablation harness's positive-control discipline — a real toy model
trained on a real next-token objective for 200 steps, quantized, with
its loss checked to move measurably but stay bounded (proving
`apply_quantized_weights` is neither a no-op nor silently catastrophic).
`examples/19_quantization.py` runs the same story end to end and prints
real measured numbers (this run: 3.81x storage reduction, loss 0.106 →
0.110 after quantizing a genuinely trained model).

398 Python tests total (up from 389). `ruff check`/`format --check`
clean.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), Windows sidecar orphan-process reaping
(`src-tauri/src/lib.rs`'s own documented gap), or CLI conformance tests
for `chat`/`run`/`models`/`eval`/`distill`/`serve` (only `doctor` has
`CliRunner`-based tests today).

## CLI conformance tests — closing the coverage gap on every command but `doctor`

Confirmed by `grep -rln "CliRunner" tests/` before starting: exactly one
file (`test_doctor.py`). Every other `sarva` command — `chat`, `run`,
`models`, `eval`, `distill`, `sessions list`/`clear` — had only ever
been exercised through the library functions underneath it (`AgentLoop`,
`run_benchmark`, `distill()`), never through the actual `app` object a
real user's terminal invokes. `tests/conformance/test_cli.py` closes
that: 11 new tests running the real Typer `app` end to end, zero-config
(no API keys — Mock provider only), matching `cli.py`'s own "always
works" docstring guarantee.

**A real isolation problem, caught before it could quietly write to a
real home directory:** `SessionStore()` resolves `DEFAULT_SESSIONS_DIR`
(`~/.sarva/sessions`) at construction time with no CLI-level override —
tests for `chat --session`/`run --session`/`sessions list`/`sessions
clear` would otherwise read and write the real `~/.sarva/sessions` on
whatever machine runs the suite. Fixed by monkeypatching
`sarva.memory.session.DEFAULT_SESSIONS_DIR` to a `tmp_path` in every
test that touches a session, the same isolation discipline
`test_config.py` already established for `~/.sarva/config.json`.

Coverage highlights: `chat` verified to route to Mock with zero
configuration and to persist a two-message transcript under
`--session`; a wrong-media-type `--image` path verified to fail cleanly
(non-zero exit, real error text) rather than crash; `run --auto`
verified end to end through `BUILTIN_TOOLS` wiring (Mock's unscripted
turn never issues tool calls, so this proves the CLI's plumbing rather
than any specific tool); `eval --model mock` verified to report the
honest 0% score for an untrained echo provider on the arithmetic
benchmark, the same no-fabrication discipline as elsewhere in this
project; `distill` verified to write a real, parseable JSONL file with
correct prompt/completion/model fields, and to fail loudly (exit 1,
"not configured") for a model whose provider isn't set up; `sessions
list`/`clear` verified against a real saved-then-cleared session file.

11 new tests, 398 → 409 Python tests. `ruff check`/`format --check`
clean. `docs/packaging.md` updated with a new "CLI conformance tests"
section.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), or Windows sidecar orphan-process
reaping (`src-tauri/src/lib.rs`'s own documented gap).

## Windows sidecar reaping — a real bug found while investigating the documented gap

`src-tauri/src/lib.rs`'s own module doc has named "Windows has no
equivalent yet" since the sidecar-killing logic first shipped, framed
as an untested corner. Investigating it properly surfaced something
more concrete: `kill_sidecar`'s grandchild-reaping logic (the part that
actually matters — PyInstaller's onefile bootloader forks a real
grandchild process that a plain `child.kill()` never reaches) was
unconditionally `#[cfg(unix)]`-gated. That means even the ordinary
graceful window-close path — `on_window_event`'s `CloseRequested`,
which already fires identically on every platform, not just the abrupt-
signal path the doc called out — silently orphaned the real frozen
server on Windows, leaving it holding the port. Not just "untested,"
a real leak on the one shutdown path Windows already exercises.

**Fixed with Windows' own native tool, not a port of the Unix
approach:** `taskkill /F /T /PID <pid>` kills the whole process tree in
one call — simpler than Unix's `pgrep -P` + `kill -9` loop, since `/T`
already recurses through every descendant.

**The other half of the original gap — an abrupt SIGTERM/SIGINT-style
kill bypassing the window-close handler entirely — genuinely still has
no Windows equivalent, for a real, checked reason rather than a vague
TODO:** `main.rs` sets `windows_subsystem = "windows"` for release
builds (required to avoid popping a console window on launch), and
Win32's console-control-handler API (`SetConsoleCtrlHandler`, the
nearest analogue to `signal-hook`'s SIGINT/SIGTERM interception) only
delivers events to a process with an attached console — a
windows-subsystem GUI app doesn't have one. A real fix needs deeper
Win32 message-loop hooking (`WM_QUERYENDSESSION`), left open and
explained rather than silently assumed non-existent.

**Honestly scoped on verification too:** this environment has no
Windows machine, so the fix is verified the same way the rest of this
file's Windows-targeted code already is — a real `cargo check` on a
genuine `windows-latest` GitHub Actions runner (CI's existing `desktop`
matrix job), confirming the new `#[cfg(windows)]` branch compiles
correctly for the target, not that `taskkill` behaves correctly at
runtime (no way to check that without real Windows hardware). Verified
locally first on macOS: `cargo check` compiles clean with a placeholder
sidecar binary (matching CI's own setup step), and `cargo fmt --check`
confirms no formatting drift in the changed file.

No Python test count change (Rust-only milestone). `docs/packaging.md`
updated with the corrected, more precise framing of what's fixed vs.
what's genuinely still open and why.

## Image-out — the first adapter to actually fill a type the protocol always had

`ModelCapabilities.modalities_out`'s own comment has said `# v1: {TEXT};
image-out models later` since the field was written, and
`ContentEvent`'s own docstring says "e.g. images from image-out
models" — both named image generation as anticipated future work
before any adapter ever actually produced one. Confirmed by `grep -rn
"ContentEvent" core/sarva` before starting: exactly two hits, the
type's own definition and its slot in the `ProviderEvent` discriminated
union — never constructed anywhere. T2's own definition of done also
literally says "Image+PDF in, image **out**," a promise T2 never fully
delivered on.

`google_provider.py` closes it: an image-capable Gemini model returns
generated image bytes as a response part with `inline_data` populated —
the exact same `Blob` shape (`.data`/`.mime_type`) this adapter already
used to *send* images in, just on the way out instead. Translated into
`ImageBlock` + `ContentEvent`, appended to the assembled message the
same way a tool-call block already was. **Chose Gemini over OpenAI's
separate Images API deliberately, not arbitrarily:** Gemini's
image-capable models return generated images inline within the same
`generateContent` streaming call an ordinary chat turn already uses,
fitting directly into the existing per-chunk translation loop; OpenAI's
image generation lives on a wholly separate `images.generate()`
endpoint unrelated to `chat.completions`, which would need a
special-cased code path outside the `generate()` streaming contract
every other adapter follows.

**Same scoping discipline this adapter already applies to Gemini
generally:** no `models.yaml` entry claims a specific image-out-capable
Gemini model id — this session has no verified-current catalog of
which model variants actually support it or their pricing, so the
wire-level translation is real and tested, but nothing routes a real
request to it yet without an explicit model override naming a real
image-out model id. Also, honestly, not live-verified: this environment
has no API keys, so — same as the rest of `google_provider.py` since it
was first written — this is proven with hermetic `SimpleNamespace`-based
tests (the established "unit-test pure translation, verify the rest
live" pattern `test_openai_provider_streaming.py`/
`test_google_provider_streaming.py` already use), not a live call. No
new live test added either: doing so responsibly would need a real,
verified image-out model id, which this session doesn't have — adding
one would mean guessing, which this project doesn't do.

3 new tests in `tests/conformance/test_google_provider_image_out.py`:
an inline-data part becoming a `ContentEvent` with the right
media type/bytes, the generated image surviving into the final
assistant message, and text+image coexisting in one response without
either clobbering the other. 409 → 412 Python tests. `ruff check`/
`format --check` clean. `docs/providers.md` updated with a new
"Image-out" section.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a real local Whisper/TTS audio in/out
pipeline (T2's other still-unmet promise), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Local speech — closing T2's other still-unmet promise (Whisper/TTS)

T2's own definition of done has said "audio in/out (local Whisper/TTS)"
since T2; `AudioToTextDegrader` existed but never actually transcribed
anything (always "could not be transcribed," confirmed by reading its
own body before starting), and there was no TTS anywhere at all —
confirmed by `grep -rln "whisper\|Whisper\|TTS" core/sarva` returning
nothing.

New `sarva.audio` closes both directions, with two deliberately
different substrate choices, each picked to avoid a heavy dependency
where a real, lighter option already existed:

- **TTS shells out to the OS's own bundled engine** (macOS `say`,
  Linux `espeak`/`espeak-ng`) instead of a Python TTS library.
  `pyttsx3` (the standard cross-platform wrapper) was tried first and
  rejected once installed: it pulled in the ENTIRE `pyobjc` framework
  suite on macOS — 100+ separate packages — just to reach the exact
  same `say` command this module now calls directly via `subprocess`,
  for a fraction of the footprint.
- **STT uses `faster-whisper`**, a new, genuinely optional
  `sarva[audio]` extra (there's no OS-native local speech recognizer to
  shell out to the way TTS has one). Checked before adding it via
  `importlib.metadata.requires("faster-whisper")`: its hard
  dependencies (`ctranslate2`, `huggingface-hub`, `tokenizers`,
  `onnxruntime`, `av`, `tqdm`) pull in no `torch` — `av` is already a
  `core` hard dependency (the video degrader) — so this stays a
  lightweight extra alongside `sarva[foundry]`, not a second heavy ML
  stack.

**A real bug found empirically, not a hypothetical:** the first version
called `say -o file.wav "text"` with no explicit voice and it appeared
to work (exit code 0, a file got written) — but `afinfo` on the result
showed 0.005 seconds of real audio for a six-word sentence, real
near-silence, not a parsing artifact. Explicitly naming a bundled voice
(`-v Samantha`) on the identical text produced correct, full-length
audio. `say`'s own DEFAULT voice resolution was silently producing
broken output in this environment — the kind of bug that would have
looked like a working feature (a file gets written, no error) until
someone actually listened to one. `synthesize()` always passes an
explicit voice now, and a dedicated regression test
(`test_synthesize_with_default_macos_voice_produces_full_length_audio`)
pins that longer text produces proportionally more audio bytes.

**A second real, honest finding, caught while writing the tests, not
swept under a passing assertion:** the "tiny" Whisper model
occasionally mishears the word "Sarva" itself as "Serve a" — a
plausible phonetic near-miss for an uncommon proper noun a small model
was never going to be perfectly tuned for. Rather than picking a lucky
seed or retrying until a test happened to pass, every test was written
against common, unambiguous words instead — an honest choice about what
a real round-trip test can reliably prove, documented directly in the
test's own comment.

`AudioToTextDegrader` now attempts real transcription first when the
extra is installed, falling back to the original honest metadata-only
message only when it's missing or transcription genuinely fails on
that specific audio — proven with a real, no-mocking round trip:
`sarva.audio.synthesize()` generates genuine speech locally, the
resulting WAV goes straight through the degrader, and the words that
come back are checked against the words that were spoken. New `sarva
speak TEXT [--out PATH] [--voice NAME]` CLI command is the reachable
surface for TTS — the same "fully built but unreachable by any real
user" gap this project has named and fixed before (the MCP HTTP
transport milestone). `sarva doctor`/`GET /doctor` gained two checks
("Speech-to-text (local Whisper)", "Text-to-speech (local)") sourced
from the same `sarva.audio` functions everything else uses, so they can
never drift from real availability.

**CI now actually exercises the `sarva[audio]` extra, not just
declares it:** `uv sync --all-packages --group dev` never included
optional extras (only workspace members like `sarva-foundry`, which is
why `foundry` was always present without asking) — added
`--all-extras` so `faster-whisper` installs for real in CI, plus a new
clean-wheel-install smoke check (`from sarva.audio import transcribe`)
mirroring the existing foundry-extra check. `core`'s CI job runs on
`macos-latest`, so the macOS `say` path (the primary, verified
implementation) genuinely executes there — not just locally.

12 new tests across `test_audio.py`, `test_cli.py` (`sarva speak`),
`test_degraders.py` (real end-to-end transcription), and `test_doctor.py`
(2 new diagnostic checks) — plus a `test_server.py` fixture update for
the same 2 new checks. 412 → 421 Python tests (1 environment-gated
skip). `ruff check`/`format --check` clean. `docs/packaging.md` gained
a new "Local speech" section, `docs/multimodal.md`'s audio-degrader
description updated, `README.md` updated.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Linux `espeak`-path real-runtime
verification (written against documented CLI behavior but only macOS
confirmed by direct execution here), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Native video-in — closing T5's other still-open roadmap line

The design doc's own T5 roadmap line names "MCP client, video input" as
a still-open deliverable; MCP shipped a few milestones back, video
input never had a native path — every `VideoBlock` had exactly one
route to any model, `VideoToTextDegrader` sampling up to 4 frames into
`ImageBlock`s first, confirmed by `grep -n "VideoBlock" core/sarva/
providers/*.py` returning nothing before starting. Real and useful, but
lossy: a model that genuinely understands video motion/audio/temporal
structure never got the chance to.

`google_provider.py` now translates `VideoBlock` directly too, via the
identical `inline_data`/`Blob` shape already used for `ImageBlock` —
the same wire mechanism, just a different media type, since Gemini's
own API treats inline video and inline images identically at the
`Part` level. Chose Gemini for the same reason image-out did: it's the
one provider among the three with genuine native video understanding
built into the same chat-completion call every other content type
already goes through — Anthropic and OpenAI's chat APIs have no video
input support to translate to at all.

**Additive, not a replacement:** `VideoToTextDegrader` stays exactly as
useful as before for every other provider, or for a caller who
explicitly wants the frame-sampled fallback (e.g. a text-only model, or
before this adapter existed). Nothing about the degradation opt-in
mechanism (`AgentLoop(degraders=...)`) changed — a video sent to
Gemini through the normal flow now just has a real native path
available where previously it had none.

**Honestly scoped on size, not silently assumed unlimited:** inline
`Blob` data is base64-encoded directly into the request body, which
Gemini's own documented limits cap around 20MB total request size —
fine for the short clips this project's tests and examples use, but a
real caller with a longer video needs Gemini's separate Files API
(upload once, reference by URI), named directly as real, deferred
follow-up work rather than something this change silently mishandles.

1 new test (`test_video_block_translation_round_trips_raw_bytes`,
mirroring the existing image-block translation test exactly). 421 → 422
Python tests. `ruff check`/`format --check` clean. `docs/providers.md`
gained a new "Video-in" section; `docs/multimodal.md`'s video-degrader
description cross-references it.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Linux `espeak`-path real-runtime
verification (written against documented CLI behavior but only macOS
confirmed by direct execution here), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Extended-thinking round trip — a comment that was more pessimistic than the code

`anthropic_provider.py`'s own `_to_anthropic_message` had a comment
saying `ThinkingBlock` round-tripping "lands when the agent loop starts
threading provider_data back through GenerateRequest, tracked as real
deferred work, not yet built." Reading it closely before treating it as
settled: `ThinkingBlock.provider_data` already existed as a real field,
`generate()` already populated it with the SDK's own signature the
moment a thinking block was produced, and `agent/loop.py` already
threads that exact `Message` object into the next turn's history
unmodified (`messages.append(done.message)`) — no stripping, no
reconstruction, nothing lossy in between. The "agent loop plumbing"
the comment worried about had already landed by the time this was
looked at; the only genuinely missing piece was this one function
actively throwing the block away instead of reconstructing it. A stale
comment describing a bigger gap than the code actually had — this
project's own recurring theme of checking a claim against current
source rather than trusting an old note, just found in a comment this
time instead of a docstring or a README line.

**Why this matters, not just tidiness:** Anthropic requires the
*original* signature back on a reused thinking block — an
anti-tampering check — when continuing a conversation after a tool
call made during extended thinking. Dropping it unconditionally, as
the code did, meant every multi-turn tool-using conversation with
thinking enabled lost its reasoning continuity on the very next turn,
silently, no error, just a strictly-worse continuation. Fixed:
`_to_anthropic_message` now reconstructs `{"type": "thinking",
"thinking": ..., "signature": ...}` whenever `provider_data` actually
carries a signature, and still drops the block exactly as before when
it doesn't (a hand-built session, or one from before this field
existed) — no fabricated signature ever sent, since Anthropic would
reject one anyway.

**Verified beyond the single-block translation unit test:** a new
hermetic, end-to-end test drives `AnthropicProvider.generate()` twice
against a fake SDK client — turn one returns a thinking+tool_use
response, the test builds history the *exact* way `AgentLoop` itself
does (not a shortcut), and turn two's actual captured request payload
is inspected to confirm the reconstructed thinking block matches
byte-for-byte, including ordering (thinking block before the tool_use
block, matching the order Anthropic originally returned them). Proves
the real pipeline works, not just that the translation function
returns the right dict in isolation.

3 test changes (1 updated to reflect the new signature-present
behavior, 1 new no-signature-drop test, 1 new end-to-end round-trip
test), 422 → 424 Python tests. `ruff check`/`format --check` clean.
`docs/providers.md` gained a new bullet in the "every backend
disagrees" section.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Linux `espeak`-path real-runtime
verification (written against documented CLI behavior but only macOS
confirmed by direct execution here), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Foundry checkpoint bundles now serialize MoE and RoPE-scaling configs

`foundry_provider.py`'s own module docstring had named this as a real,
open gap since the adapter first shipped: MoE and long-context
RoPE-scaling are real, trainable foundry architecture features, but
`save_checkpoint_bundle` refused to save a checkpoint trained with
either — `NotImplementedError`, rather than silently writing a bundle
that would reload as a plain dense/unscaled model not matching what was
actually trained. Closed by checking what the two config types
actually needed: `MoEConfig` (`n_experts`/`n_experts_per_tok`/
`n_shared_experts`) and `RopeScalingConfig` (`method`/`factor`) are both
flat dataclasses with only JSON-safe fields — nothing about them
resisted serialization, the refusal was really just "nobody had wired
the two extra dict conversions through yet."

`save_checkpoint_bundle` now writes `"moe"`/`"rope_scaling"` as nested
`null`-or-object fields in `config.json`; `load_checkpoint_bundle`
reconstructs real `MoEConfig`/`RopeScalingConfig` instances from them
before building `TransformerConfig`. **Real backward compatibility,
verified rather than assumed:** a dedicated test hand-writes a
`config.json` in exactly the shape the OLD code would have produced
(no `"moe"`/`"rope_scaling"` keys at all, not even `null` ones) and
confirms `load_checkpoint_bundle` still reconstructs it correctly —
proving the old-format path works, not just that the new format
round-trips.

3 test changes (the old "refuses MoE" test replaced with two real
save-then-load round-trip tests, one per config type, plus the
backward-compatibility test), 424 → 426 Python tests. `ruff check`/
`format --check` clean. `docs/foundry/inference.md`'s "Checkpoint
bundles" section updated to drop the now-resolved caveat.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Linux `espeak`-path real-runtime
verification (written against documented CLI behavior but only macOS
confirmed by direct execution here), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Foundry is gradable through the eval harness — a stale docstring found in three places, and a real automated-coverage gap it was hiding

`eval/harness.py`'s own module docstring said the foundry adapter
"doesn't exist yet — named as real, deferred work, not implied to
already be done." `docs/eval.md` said the same thing almost verbatim.
Both were wrong: `FoundryProvider` has existed since an earlier
milestone and is fully wired into the registry. Checking a claim
against current source before trusting it, the same recurring theme
this session keeps running into in different files.

**The stale docstring was hiding something more concrete than itself
being outdated:** `test_eval_harness.py`'s own docstring grouped
foundry grading under "live-only, exercised by whoever runs `sarva
eval` with a configured API key" — but foundry needs no API key and no
network at all, unlike Anthropic/OpenAI/Google. It had been genuinely
hand-verified exactly once, in an earlier session, via a manual `sarva
eval --model foundry/toy` CLI run — and never pinned as a real,
permanent regression test. There was no actual reason it had to stay
in the "can't test here" bucket.

New test in `test_foundry_provider.py`: trains a real tiny checkpoint,
wraps it in `FoundryProvider`, runs it through the real
`run_benchmark()` against the real bundled `ARITHMETIC` benchmark, and
asserts the honest result — 0% accuracy for an untrained toy model,
same no-fabrication discipline this project already applies to the
zero-config Mock provider's own eval score. Three docstrings corrected
to describe what's actually true now: `eval/harness.py`,
`test_eval_harness.py`, and `docs/eval.md`.

1 new test, 426 → 427 Python tests. `ruff check`/`format --check`
clean.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Linux `espeak`-path real-runtime
verification (written against documented CLI behavior but only macOS
confirmed by direct execution here), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Ollama verified live — and a real, latent test-isolation bug it surfaced

Every provider adapter's module docstring has carried some version of
"written to the documented shape, not yet exercised against a live
[thing]" since it was written — true for Anthropic/OpenAI/Google
because this environment has no credentials for any of them. Ollama is
categorically different: it needs no API key, only a locally running
server, and this environment can actually run one. `brew install
ollama` (already available via Homebrew), `ollama serve`, `ollama pull
qwen2.5:0.5b` (~400MB — `models.yaml`'s real registered default,
`qwen3:8b`, is ~5GB, too large to be a reasonable one-off verification
download).

With a real server running: `tests/live/test_live_providers.py::
test_ollama_terminal_event_law` passed for real (added an
`OLLAMA_TEST_MODEL` env var override first, matching the
`OPENAI_TEST_MODEL`/`GOOGLE_TEST_MODEL` precedent, so the pulled model
didn't require editing the test itself). Went further than the one
pinned test: a direct script confirmed real incremental streaming
(`TextDeltaEvent`s accumulating to the exact final message text) and
real tool-calling (a `get_weather` tool call correctly parsed into a
`ToolCallEvent` + `DoneEvent` with `StopReason.TOOL_USE`) against the
live server — the parts of `ollama_provider.py` that had never been
exercised against real wire data before, not just the happy-path
terminal event.

**A real, latent bug this surfaced, not caused by anything this session
did wrong:** running the full test suite with the real Ollama server up
broke 7 tests across `test_cli.py`/`test_server.py`. Every one asserted
"zero-config routes to Mock" without ever mocking away
`ollama_reachable()` — silently true in CI and in this sandbox purely
because no real Ollama server had ever happened to be running before
now, not because anything actually forced Mock. Once a real server
existed, the real router legitimately preferred `ollama/qwen3:8b`
(reachable, but not the specific small model actually pulled) over
falling back to Mock, and those tests broke on a real, if incidental,
routing decision. **Any contributor who runs this suite on a machine
with their own local Ollama already running — a very plausible setup
for exactly the kind of person this project's free/local/private tier
targets — would have hit the identical failures**, independent of
anything this session did; the real Ollama install just made it visible
here first.

Fixed at the root, not by shutting the server back down to make the
symptom disappear: `test_cli.py`'s `_clear_provider_env` now also
mocks `ollama_reachable` to `False`, matching the pattern `test_doctor.
py` already used correctly; `test_server.py` gained a `_force_mock_only`
helper (same patch target as the existing `_use_scripted_mock`) applied
to every test that assumed default zero-config routing without
explicitly forcing it. All 427 tests pass with the real Ollama server
left running in the background — the suite is now actually isolated
from incidental local machine state, not just working by coincidence.

`ollama_provider.py`'s own module docstring, `docs/providers.md`, and
this entry all record the real verification. No test *count* change
(existing tests fixed, not new ones added) — 427 stays 427.
`ruff check`/`format --check` clean.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Linux `espeak`-path real-runtime
verification (written against documented CLI behavior but only macOS
confirmed by direct execution here), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## The espeak TTS branch verified for real too, closing the last open item from the local-speech milestone

The local-speech milestone's own "Next" note named this directly: the
Linux `espeak`/`espeak-ng` branch of `sarva.audio.synthesize()` was
written against documented CLI behavior but only ever verified via the
macOS `say` branch, since real macOS always takes that branch first.
Closed by installing `espeak-ng` via Homebrew (available on macOS too,
not Linux-exclusive) and writing a test that hides `say` specifically
(`shutil.which` monkeypatched to return `None` only for `"say"`, every
other command resolving normally) so the actual espeak subprocess call
runs for real, not mocked — the same bar the `say` branch already
cleared.

Verified two levels deep, mirroring the `say` branch's own verification
story: a raw `espeak-ng -w file.wav "..."` call confirmed the CLI shape
this module's code uses actually produces a valid WAV, and a full
`synthesize()` → `transcribe()` round trip (real espeak-ng speech, fed
through real `faster-whisper`) confirmed the words come back correctly
— "the quick brown fox jumps over the lazy dog" synthesized and
transcribed cleanly. The only piece of `sarva.audio`'s TTS surface still
genuinely unverified is the Windows branch, which has no engine
implemented at all yet (a real, named, open gap, not glossed over).

2 new tests, 427 → 429 Python tests. `ruff check`/`format --check`
clean. `sarva.audio`'s own module docstring and `docs/packaging.md`
updated with the real verification record.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## MCP verified against a real, independent third-party server, not just this project's own fixtures

Every MCP conformance test this project had — stdio and HTTP alike —
talked to a fixture server this same project wrote
(`tests/fixtures/mcp_echo_server.py`/`mcp_http_echo_server.py`). Real
subprocesses, genuine protocol traffic, but a fixture written by the
same codebase as the client testing against it can in principle share
the client's own misreading of the spec — a gap worth closing directly,
the same instinct behind verifying Ollama against a real server instead
of only this project's own mock.

Installed and ran `@modelcontextprotocol/server-filesystem` — Anthropic's
own official reference filesystem MCP server — via `npx`, and drove it
through Sarva's real `connect_stdio_mcp_server`/`list_mcp_tools`/
`McpToolAdapter` path, not a special test-only code path. It listed all
14 real tools the server actually implements (not assumed from
documentation), and a full read-then-write round trip worked: `read_
text_file` returned the exact file contents, `write_file` reported
success, and — the real proof for the write direction, not just
trusting the tool's own claim — the written file was read back directly
from disk afterward and matched exactly. Also confirmed through the
actual CLI: `sarva run --mcp-server "npx -y
@modelcontextprotocol/server-filesystem ..."` connected and printed the
real tool list, proving the wiring works at the command-line boundary
too, not just the library call.

New `tests/live/test_live_mcp.py`, gated the same way every other
live-external-service test in this project is (`pytest.mark.live`,
skipped by default; additionally skipped if `npx` isn't on `PATH`) —
deliberately not made a default CI dependency, since depending on npm
registry availability on every push isn't a tradeoff this project makes
for any other live verification either.

1 new test, always excluded from the default 429 (live-gated), so no
count change to the numbers that matter for the default run. `ruff
check`/`format --check` clean. `docs/mcp.md`'s "Verification" section
updated with the real third-party interop record.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## The docs site had never actually been built — found while sweeping docs/index.md for staleness

Started this as a routine staleness sweep of `docs/index.md` (the
book's own front page): "Chapter 4... the three real degraders
(image/audio/video)" was stale (document makes four, and audio now
does real transcription, both from earlier milestones this session);
"Chapter 6... the CLI's seven commands... no Windows sidecar-kill
signal handling yet" was stale on both counts (nine commands now,
including `speak`; the Windows sidecar leak on graceful close is
fixed, only the abrupt-kill signal path remains genuinely open). Fixed
both.

**Verifying those fixes actually rendered led to a much bigger, real
finding:** `mkdocs build` had never once been run against this
project's `docs/mkdocs.yml` — no CI job builds it, and every chapter
across this whole session was written and reviewed as plain Markdown,
never through the actual site generator. Installed `mkdocs`/
`mkdocs-material` for real and ran a build: it failed immediately,
before rendering a single page — `mkdocs.yml` lived inside `docs/`
itself with the default `docs_dir` (which looks for a `docs/`
subdirectory *relative to the config file*, i.e. a nonexistent nested
`docs/docs/`), not because any chapter's content was wrong. The book
this project's own positioning is built around had a config bug that
made it entirely unbuildable, the whole session, undetected.

**Fixed at the root, matching mkdocs' own idiomatic layout, not
special-cased:** moved `mkdocs.yml` to the repo root (`docs_dir`
defaults to `docs/`, a real sibling directory from there — no `nav:`
path needed to change, since every path was already relative to
`docs_dir`). Verified for real: a full `mkdocs build --strict` now
succeeds, producing real rendered HTML for all 13 chapters — spot-
checked `site/index.html` directly for the corrected "Nine commands"/
"four real degraders" text landing correctly.

**Closed the loop so this can't silently regress again:** new `docs`
CI job (`ubuntu-latest`, installs `mkdocs`/`mkdocs-material`, runs
`mkdocs build --strict` — `--strict` turns a broken `nav:` entry or
dead internal link into a real CI failure, not a silently missing
page). `README.md` gained a one-line "build/preview the book locally"
note.

No Python test count change (docs/config-only milestone) — 429 stays
429. `ruff check`/`format --check` clean (unaffected, no Python
touched).

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## The built wheels had no license metadata at all — found the same way the docs config bug was

Same instinct as the mkdocs config fix: check a real built artifact
directly instead of assuming the repo's own honesty (a real `LICENSE`
file, README saying "MIT") automatically propagates into what actually
ships. It doesn't, automatically — neither `core/pyproject.toml` nor
`foundry/pyproject.toml` declared a `license` field at all, confirmed
by building a real wheel and grepping its actual `METADATA` file for
`License`/`Classifier` lines: nothing. A real `pip install sarva` (or
a PyPI listing, if this project ever publishes there) would have shown
no license information whatsoever, despite this genuinely being an
MIT-licensed project.

Fixed with the modern PEP 639 form — `license = "MIT"` (an SPDX
expression string, not the older `{text = "..."}` table) — added to
both packages' `[project]` tables. Verified directly, not assumed:
`uv build --all-packages` then `unzip -p *.whl "*.dist-info/METADATA"`
now shows `License-Expression: MIT` for both `sarva` and
`sarva-foundry`, with `Metadata-Version: 2.4` confirming the installed
hatchling actually supports the modern field rather than silently
ignoring it.

**A real, honest limit found and not glossed over:** also tried
`license-files = ["../LICENSE"]` to physically bundle the actual
license text into the wheel's `dist-info/licenses/` directory (the
other half of PEP 639) — the build didn't error, but the file never
actually appeared in the wheel, confirmed by listing its full contents.
Rather than leave a declaration that silently does nothing, removed it
— the honest, verified fix is the SPDX expression alone; bundling the
literal license text is real, separate, unimplemented follow-up work,
not something this change gets to claim.

Closed the loop with a real CI check, not just a one-time manual
verification: the existing "Verify installable wheels" step now greps
each real built wheel's `METADATA` for `License-Expression: MIT` and
fails loudly if it's missing, run against the exact command sequence
verified by hand first.

**Update, same session:** the "real, separate, unimplemented follow-up
work" above turned out to be closeable immediately — the actual reason
`../LICENSE` bundled nothing wasn't some deeper limitation, it was
specifically the `../` traversal: hatchling's `license-files` globs are
sandboxed to the project directory, confirmed by testing the identical
config with an in-tree copy instead (`license-files = ["LICENSE"]`,
`core/LICENSE` physically present) — it worked immediately. Copied
`LICENSE` into both `core/` and `foundry/` (each a real,
independently-installable package needs its own in-tree copy; a
comment in each `pyproject.toml` explains why this duplication is
deliberate, not accidental, and to keep both in sync with the root
`LICENSE` if it's ever amended). Verified thoroughly: both wheels now
bundle a real `dist-info/licenses/LICENSE`, confirmed **byte-identical**
to the repo-root `LICENSE` via `diff`, not just "a file exists." The CI
check was strengthened to match — it now verifies the bundled license
text is byte-identical to the repo root's, not just that the METADATA
expression string is present.

No Python test count change (packaging-metadata-only milestone) — 429
stays 429. `ruff check`/`format --check` clean (unaffected).
`docs/packaging.md`'s "Verified, not assumed" section updated.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Ollama availability is per-model now, not per-server — a second, deeper bug the live setup surfaced

The earlier Ollama-live-verification milestone's test-isolation fix
addressed the symptom: several tests broke once a real Ollama server
started running, because they assumed "zero-config routes to Mock"
without controlling for a reachable-but-model-not-pulled server. Fixed
by mocking `ollama_reachable` in those tests. That left the actual
production code path untouched, and it has the same real bug: running
`sarva run "list files" --mcp-server "..." --auto` in this exact
environment — Ollama genuinely reachable, only a small model
(`qwen2.5:0.5b`) actually pulled, not the registry's registered
`qwen3:8b` — ended with a real `run ended: failed`. `build_router()`
marked `ollama/qwen3:8b` "available" the instant the server merely
answered, with zero regard for which model tag was actually present.
The zero-config Mock fallback never got a chance, because the router
genuinely believed an unpulled model was a working one.

**Real fix, not another test-only patch:** new `ollama_pulled_models()`
queries the exact same `/api/tags` endpoint `ollama_reachable()`
already hits (confirmed its real response shape against the running
server: `{"models": [{"name": "qwen2.5:0.5b", ...}]}`) and returns the
real set of locally-pulled tags. `build_router()` now marks
`ollama/<tag>` available only when that exact tag is in the real pulled
set, not merely when the server answers. `run_diagnostics()`'s Ollama
check gained the same real data in its own detail message (`pulled:
qwen2.5:0.5b`, or an explicit "no models pulled yet" for a reachable
but empty server) — real information a user can act on, not just a
green checkmark that doesn't mean what it implies.

**Verified against the real, still-running local server, not just
unit tests:** re-ran the identical `sarva run ... --auto` command that
had failed and it now correctly falls back to Mock; `sarva doctor`
correctly shows `pulled: qwen2.5:0.5b`; `sarva models` correctly shows
`ollama/qwen3:8b` as `[ ]` unavailable rather than `[x]`.

5 new tests (`test_runtime.py`, new — the real bug reproduced directly
with a registered-but-unpulled model, the fix verified with an
actually-pulled matching tag, the unreachable-server short-circuit
path, and the real `/api/tags` response-shape parsing; plus 2 more in
`test_doctor.py` for the enhanced detail message). 429 → 434 Python
tests. `ruff check`/`format --check` clean. `docs/providers.md`'s
Ollama section gained a follow-up paragraph on this second, deeper fix.

## FoundryProvider now raises instead of silently dropping unsupported content

A real inconsistency, found by checking `foundry_provider.py` against
the discipline the three frontier adapters already hold themselves to:
`anthropic_provider.py`/`openai_provider.py`/`google_provider.py` all
raise a loud `ValueError` for a content-block type they have no wire
mapping for, with the same reasoning stated directly in each — silently
dropping it would answer as if the content was never sent, a
materially misleading response, not a cosmetic gap.
`foundry_provider.py`'s own `_flatten_prompt` didn't follow that
discipline at all: it built the prompt via `Message.text()`, which
silently drops every non-`TextBlock` by design (the right behavior for
its own stated job, "just give me the words," but wrong here). A
foundry checkpoint's own registry entry declares
`modalities_in={TEXT}`, `tool_use=False` — genuinely text-only, so an
`ImageBlock`/`ToolCallBlock`/anything else reaching this adapter meant
a caller's real content was being thrown away with no signal at all.

Fixed by checking every block in `_flatten_prompt` and raising the
same clear `ValueError` the other three adapters already would, naming
the real block type that couldn't be translated. Reachable only via an
explicit model override — the router's own modality check would never
route an image-bearing request to a text-only-registered model on its
own — the identical reachability note the other adapters' own guards
already carry, so this isn't a new caveat, just consistency.

1 new test (`test_foundry_provider_raises_instead_of_silently_dropping_an_image`,
mirroring the frontier adapters' own untranslatable-block-type tests).
434 → 435 Python tests. `ruff check`/`format --check` clean.
`docs/foundry/inference.md`'s "What the adapter honestly does and
doesn't do" section gained a new bullet.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Ollama adapter: the same silent-drop bug just fixed in Foundry, plus its first-ever unit tests

Immediately after fixing `foundry_provider.py`'s silent content drop,
checked the one other adapter without a loud-failure guard for
untranslatable content — `ollama_provider.py`. Same bug, same shape:
`_to_ollama_message`'s translation loop handled `TextBlock`/
`ToolCallBlock`/`ToolResultBlock` explicitly but had no `else` branch
at all, so an `ImageBlock` (or anything else) reaching it was silently
skipped, the model answering as if it had never received it. Fixed
with the same `ValueError` guard the Anthropic/OpenAI/Google/Foundry
adapters already carry. **Named directly, not silently assumed
unreachable:** real vision-capable Ollama models (llava, qwen2-vl, ...)
do accept images via a separate `images: [base64, ...]` field in
Ollama's own chat API this adapter has never built — real, deferred
follow-up, distinct from the "this adapter genuinely can't" case
`foundry_provider.py`'s equivalent guard describes.

**A second, real gap found while fixing the first:** `_to_ollama_
message` had zero unit-test coverage anywhere in the conformance
suite — the only thing exercising it at all was
`tests/live/test_live_providers.py`, skipped without a real running
server, unlike every other adapter's own dedicated translation-test
file. New `tests/conformance/test_ollama_provider.py`: text/tool-call/
tool-result translation, the new unsupported-block-type guard (two
block types, not just one, to prove it's genuinely general), and
`_strip_local_prefix`'s namespace-stripping — all hermetic, no
network. Re-ran the real live Ollama test afterward against the
still-running local server to confirm the fix didn't disturb the real
working path.

8 new tests, 435 → 443 Python tests. `ruff check`/`format --check`
clean. `docs/providers.md`'s "every backend disagrees" section gained
a new bullet (Ollama's own tool-result shape), and its Ollama section
gained a paragraph on this third fix.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## The eval harness's own core grading logic was wrong — Mock scored 30%, not the honest 0% every prior claim assumed

This project's docs and journal have repeated "Mock scores 0%, the
honest result" many times this session — always asserted, never
re-verified against the actual number. Re-checked it directly: `sarva
eval --model mock` genuinely reported **30%** accuracy, not 0%. Two
independent, real bugs compounded into that number:

1. `contains_match`'s naive `expected in output` substring check graded
   a genuinely wrong numeric answer as correct whenever the right
   digits happened to appear inside a longer wrong number — e.g.
   `"9" in "The answer is 89"` is `True`, even though 89 is not 9.
2. `ARITHMETIC`'s own `div-1`/`div-2` cases used a perfect square as
   the dividend with its own square root as the divisor (`144 / 12`,
   `81 / 9`) — so the correct answer (`12`, `9`) was already sitting in
   the prompt text verbatim, and Mock's own prompt echo passed grading
   on those two cases without computing anything at all.

Fixed both: `contains_match` now matches on a real word boundary
(`re.search(r"\bexpected\b", ...)`, not a raw substring); `div-1`/
`div-2` replaced with `84 / 7` and `45 / 5`, chosen so the quotient
never appears anywhere in the prompt. `sarva eval --model mock` now
genuinely reports `0% (0/10)`, verified directly through the real CLI,
not just a unit test.

**A related test bug this surfaced, explaining why nothing caught it
sooner:** the CLI conformance test meant to guard exactly this claim
asserted `"0%" in result.stdout` — and `"30%"` also contains `"0%"` as
its own trailing substring, the identical class of bug as the grader
itself, just one layer up. That assertion would have silently passed
at 10%, 20%, 30%, or 100% just as easily as at the honest 0%. Fixed to
check the precise `"0/10"` correct/total marker the CLI already prints,
which no wrong accuracy percentage could satisfy by coincidence.

**Also pinned as a structural invariant, not just a one-time fix:** a
new test walks every `ARITHMETIC` case and asserts its expected answer
never appears (word-boundary-matched) in its own prompt text, so a
future case can't silently reintroduce the same flaw.

3 new tests (`contains_match`'s substring-false-positive regression,
the prompt-doesn't-leak-the-answer invariant, and a real integration
level check that Mock scores exactly 0.0 against the real bundled
benchmark), 1 existing test corrected (`test_eval_grades_the_mock_
provider_against_the_arithmetic_benchmark`, `"0%"` → `"0/10"`). 443 →
446 Python tests. `ruff check`/`format --check` clean. `docs/eval.md`
gained a new section on this bug; `sarva.eval.benchmarks`'s own module
docstring documents the `div-1`/`div-2` fix directly.

## A third reward-hacking exploit found in GRPO's own reward function — the same bug just fixed in the eval harness

`answer_reward`'s own docstring said it followed "the same
`contains_match` philosophy" `sarva.eval`'s default grader uses —
which turned out to mean it inherited `contains_match`'s exact bug
too, just fixed a few commits earlier. A raw substring check
(`expected_answer in answer_segment`) rewards a genuinely WRONG answer
whenever the right digit happens to appear inside a longer wrong
number. For `examples/17_reasoning_token_training.py`'s own task
(single-digit addition), this is a real, not hypothetical, risk:
roughly half of all real sums are two-digit (10-18), so a model
answering `"17"` when the expected answer was `"7"` was scored as
fully correct by the actual reward function real GRPO training uses.
Confirmed directly: `answer_reward("<think>...</think>The answer is
17", "7")` returned `1.0` before this fix.

This is the module's THIRD documented reward-hacking exploit —
`format_reward`/`answer_reward` already had two others closed earlier
this session (the `</think>`-padding exploits, both from a real GRPO
training run discovering them). Fixed the same way
`contains_match` was: matched on a real word boundary (`\bexpected\b`),
not a raw substring.

**The already-published 31% → 56% GRPO numbers were re-checked against
the fix, not left standing on faith:** re-ran the example (fixed seed,
fully deterministic) after the fix and got the identical 31% → 56%
result. The exploit was real, proven by the standalone reproduction —
it just didn't happen to change this specific already-published run's
numbers. Confirmed by actually re-running it, not assumed because the
fix "should" leave a healthy run unaffected.

1 new test (`test_answer_reward_does_not_reward_a_wrong_answer_
containing_the_right_digit`, mirroring `contains_match`'s own new
regression test). 446 → 447 Python tests. `ruff check`/`format
--check` clean. `docs/foundry/training.md` gained two new paragraphs:
the exploit itself, and the honest re-verification of the published
numbers.

## `sarva --version` — a small, real, genuinely missing convenience

Noticed while poking at `sarva --help`'s own output for something
unrelated: no `--version` flag existed at all, not even as a stub —
`typer`'s default top-level options were just `--install-completion`/
`--show-completion`/`--help`. A real gap for a real CLI tool; anyone
reporting an issue or checking what they have installed would
reasonably reach for it first. Added a top-level `@app.callback()`
with an eager `--version` option that prints
`importlib.metadata.version("sarva")` (the real installed package
version, not a hardcoded string that could drift from `pyproject.toml`)
and exits before any subcommand logic runs. Verified for real:
`sarva --version` prints `sarva 0.1.0.dev0`, exit code 0; `sarva --help`
and every existing command still work unchanged.

1 new test, 447 → 448 Python tests. `ruff check`/`format --check`
clean. `docs/packaging.md` updated.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## A real SSRF gap in WebFetchTool, found by security-auditing the built-in tools

After several rounds of finding real bugs by re-checking specific
claims, stepped back to security-audit the built-in tools directly
(`ReadFileTool`/`WriteFileTool`/`RunShellTool`/`WebFetchTool`).
`_within_workdir`'s path-traversal guard checked out clean — verified
directly with a real symlink escape attempt and a real `../../../../`
traversal attempt, both correctly blocked. `WebFetchTool` didn't:
marked `destructive=False` (so it runs with **zero confirmation**, even
in the CLI's default non-`--auto` mode), it would fetch any http(s)
URL with no restriction at all. Confirmed directly, not hypothetical:
`web_fetch` on `http://127.0.0.1:11434/api/tags` — this environment's
own real running Ollama server — succeeded and returned the response
straight into the model's context. The identical request shape reaches
a cloud metadata endpoint (`http://169.254.169.254/...`, a well-known
SSRF target for exfiltrating cloud credentials) or any other internal
service with the same ease — a real OWASP-listed vulnerability class
(SSRF), reachable by a tool the model can call with no human in the
loop at all.

**Fixed with the standard mitigation, not a partial one:** before every
fetch, the target hostname is resolved and every returned IP checked
against `ipaddress`'s `is_global` property — covers RFC 1918 private
ranges, loopback, link-local (which includes the metadata address),
and other reserved ranges, for both IPv4 and IPv6, in one check.
`follow_redirects=True` was replaced with a bounded (5-hop) manual
redirect loop that re-validates the target host on **every** hop, not
just the caller-supplied URL — a validate-once-up-front check would
miss a legitimate public site's own server redirecting straight to an
internal address, a real bypass this project didn't want to leave
open. Relative/protocol-relative/absolute `Location` headers resolved
via `urllib.parse.urljoin` against the current URL, the standard
RFC 3986 resolution.

**Verified against real addresses, not just unit-level assertions:**
the real local Ollama server, the real cloud-metadata IP, a real
private-range IP, and a simulated redirect-to-internal-address (no
real attacker-controlled public redirector was available to test
against, so this one case used a monkeypatched `httpx.AsyncClient.get`
returning a real `httpx.Response` object). Real public traffic
verified unaffected too: a real `https://example.com` fetch, and a
real `http://github.com` → `https://github.com/` redirect chain, both
still work exactly as before.

5 new tests: 4 hermetic (loopback block, cloud-metadata block,
private-range block, simulated redirect-to-internal-address), 448 → 452
in the default run, plus 1 live-only test (a real redirect to a real
public site) not counted in that default total. `ruff check`/`format
--check` clean. `docs/agent-loop.md` gained a new section on this fix.

## The SSRF fix's other half: sarva.multimodal.fetch shared the identical gap

`WebFetchTool` wasn't the only real url-fetching code path in this
codebase — `sarva.multimodal.fetch.fetch_bytes()` (what
`resolve_media_bytes()` calls for any `url`-sourced `ImageBlock`/
`DocumentBlock`/etc.) had the exact same unrestricted-fetch shape, and
it runs even deeper in the pipeline than a tool call: inside provider
adapters' own message translation, with no tool-confirmation boundary
at all. Checked directly whether this is reachable through any current
external/model-controlled input path (no server endpoint or MCP tool
result constructs a `url`-sourced block from external input today) —
it isn't, right now. But the type exists specifically to support
url-sourced media, and leaving this second path unguarded while
`WebFetchTool` got fixed would be real, avoidable inconsistency the
moment anything does wire a url-sourced block up to external input.

**Refactored, not duplicated:** moved the SSRF guard
(`ensure_public_host`, same `ipaddress.is_global` check) into
`sarva.multimodal.fetch` — the more foundational of the two modules
(`agent/tools.py` already imports from `multimodal/`, not the other
way around) — and had `WebFetchTool` import it from there instead of
keeping its own copy. `fetch_bytes()` gained the same bounded
per-hop-revalidated redirect loop `WebFetchTool` already has, replacing
its own `follow_redirects=True`.

**A real test-hygiene issue caught while wiring this in, not shipped:**
`test_fetch.py`'s own docstring promises "no real network calls, fully
deterministic," but its existing tests all used `https://example.com/
...` as a stand-in hostname — and the new SSRF guard does a REAL DNS
lookup before every fetch, which would have made those tests silently
dependent on real network/DNS access despite the file's own promise
otherwise (the same class of "test depends on unstated environment
assumption" bug this session already found and fixed once, for
Ollama). Fixed by explicitly monkeypatching `ensure_public_host` to a
no-op in the handful of pre-existing tests that are about response
handling, not the guard itself — and writing the new SSRF-guard tests
against real IP literals instead of hostnames, which need no DNS
lookup and stay genuinely hermetic. Confirmed hermetic, not just
patched and hoped: the full 12-test file runs in 0.11 seconds, far too
fast to include a real DNS round trip.

5 new tests. 452 → 457 Python tests. `ruff check`/`format --check`
clean. `docs/multimodal.md` and `docs/agent-loop.md` both updated —
the latter now points at the shared guard's real location instead of
describing it as `WebFetchTool`-only.

**Next:** batching multiple concurrent inference requests (§3.6f), F1's
real distributed training infrastructure (needs real multi-node compute
this environment doesn't have), a Windows TTS engine (genuinely
unimplemented, not just unverified), or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).

## Windows TTS — closing the one named gap this environment couldn't verify locally, so CI does

`sarva.audio`'s own module docstring named it outright: "the Windows
branch genuinely has no engine at all yet." macOS shells out to `say`,
Linux to `espeak`/`espeak-ng` — both real, already-installed,
OS-native engines. Windows has an equivalent: `System.Speech.Synthesis`
(SAPI), part of every desktop Windows .NET Framework install, reached
via PowerShell — no third-party dependency, the identical "already on
the machine" bar the other two branches were picked against.

**The real risk this branch had to be built around, not an
afterthought:** `synthesize()`'s whole reason to exist is letting an
agent speak arbitrary text, including model-produced text a caller
never fully controls. A naive implementation would interpolate that
text into a `powershell -Command "..."` string — a real command-
injection surface (a string like `"; Remove-Item -Recurse -Force C:\;
"` breaking out of the intended command). Instead, the text is written
to a temp file and read back *inside* a fixed PowerShell script via
`Get-Content` — the text never becomes part of any command string or
argv element PowerShell parses as syntax, only file content. A
dedicated hermetic test (`test_windows_branch_never_puts_raw_text_
on_the_command_line`) proves this directly: it monkeypatches
`platform.system`/`shutil.which`/`subprocess.run`, feeds in a
deliberately hostile string, and asserts that string never appears in
the captured argv or script content — only in the temp text file's own
content, read back exactly.

**Honesty about what "verified" means here:** this dev environment has
no Windows machine, the same limitation that left this gap open in the
first place. Rather than ship an unverified implementation and call it
done, or claim confidence this project's own discipline doesn't allow,
a new `windows-audio` CI job runs on a genuine `windows-latest` GitHub
Actions runner and executes `tests/conformance/test_audio.py` for
real — including the pre-existing generic `_needs_tts`-marked tests
(`test_synthesize_produces_real_nonempty_wav_bytes`, and the renamed
`test_synthesize_with_default_voice_produces_full_length_audio`, which
already existed to catch macOS `say`'s own real near-silent-default-
voice bug and is equally applicable to SAPI's default voice) — rather
than writing Windows-specific mocked tests that would only prove the
code *looks* right. This mirrors exactly how the desktop `cargo check`
job's own `windows-latest` matrix leg already verifies Rust-side
Windows compilation without a local Windows machine.

`tts_engine_available()` gained a Windows branch (`shutil.which
("powershell")`/`"pwsh"`) so `sarva doctor` can never claim
availability `synthesize()` would then fail to honor — the same
single-source-of-truth pattern `ollama_reachable`/
`_foundry_extra_installed` already established. `docs/packaging.md`'s
"Local speech" section and `sarva.audio`'s own module docstring both
updated to describe the real implementation instead of naming the gap.
1 new test, 457 → 458 Python tests. `ruff check`/`format --check`
clean.

**Next:** whether CI's real `windows-latest` run actually validates
SAPI's default voice the same way it validates `say`'s (the test is
generic and will surface it either way); batching multiple concurrent
inference requests (§3.6f, still a deliberate deferral — real
correctness risk); F1's real distributed training infrastructure
(needs real multi-node compute this environment doesn't have); or a
first pass at code-signing/notarization for the desktop release
bundles (needs a real signing identity this environment doesn't have —
likely stays deferred).

## The `windows-audio` job's first real run, and a `uv sync --package` bug it caught immediately

The very first push of the `windows-audio` job (previous entry) failed
before a single Python test ran: `uv sync --package sarva --group dev`
errored with `Group `dev` is not defined in the project's
dependency-groups table`. Reproduced locally first, not assumed to be
CI-only flakiness — the identical command fails the same way on this
macOS dev machine. Root cause: constraining `uv sync` to one workspace
member (`--package sarva`) stops it from resolving a dependency group
declared in the *workspace root* `pyproject.toml` rather than that
member's own — `dev` lives only at the root. Fixed by building the
venv manually (`uv venv` + `uv pip install -e core "pytest>=8.0"
"pytest-asyncio>=0.24"`) instead of `uv sync`, which both sidesteps the
group-resolution gap and avoids pulling in `torch`/foundry this job
never needed. Caught a second, unrelated real issue while debugging:
the committed `uv.lock`'s `provides-extras` for `sarva` was missing
`"audio"` even though `core/pyproject.toml` has declared it as a real
extra for a while — a plain `uv sync --all-extras` regenerates the
lockfile correctly; fixed in the same commit.

**The re-run is the actual point of this job, and it delivered a real
answer:** on genuine Windows (a `windows-latest` GitHub Actions
runner), `test_synthesize_produces_real_nonempty_wav_bytes` and the
generic default-voice regression test both passed on the first try —
unlike macOS `say`, SAPI's own default voice did *not* reproduce the
near-silent-output bug that made `synthesize()` pass an explicit voice
for the `say` branch. Genuinely determined by running the real code on
real Windows, not inferred from documentation or assumed safe by
analogy — this project's "verify, don't assume" discipline applied to
its own prior finding, which could easily have turned out differently.
The injection-safety test
(`test_windows_branch_never_puts_raw_text_on_the_command_line`) passed
on real Windows too, not just the hermetic macOS run that first proved
the property. Full CI run genuinely green: `web`, `desktop` (all three
OSes), `docs`, `core`, and `windows-audio`, commit `53e3b1e`.

## Ollama vision — closing the last adapter's own named follow-up, verified against a real local vision model

Two milestones ago, fixing Ollama's silent-content-drop bug named the
real remaining gap directly in its own commit message: "real
vision-capable Ollama models do accept images via a separate
`images: [base64, ...]` wire field this adapter doesn't build yet."
Closed now, the same way every other Ollama claim this session has
been verified — against a real running server, not just the documented
API shape.

`_to_ollama_message` is now `async`, matching the Anthropic/Google
translators' own shape, so it can call `resolve_media_bytes` (the
shared SSRF-guarded fetch path) instead of only handling `data`/`path`
sources. An `ImageBlock` becomes one raw base64 string in Ollama's
`images` array — no `data:` URI prefix, no `media_type` field, the
opposite shape from Anthropic's `source: {type: "base64", media_type,
data}` object — confirmed against the real wire response before
writing any adapter code, not assumed to match a sibling adapter's
convention.

**Verified against a real, small, vision-capable local model:** `ollama
pull moondream` (~1.7GB; its own `/api/tags` entry reports
`"capabilities":["completion","vision"]`, checked directly rather than
assumed from the model's size class). A genuine solid-red PNG (built
with Pillow) sent through the real `OllamaProvider.generate()` path —
not a raw curl shortcut standing in for the actual adapter code — came
back `"!!!RED!!!"`: the model genuinely read the pixels and identified
the color, not an echo, not a hallucinated guess. A wrong answer here
would have been just as informative a result as a right one; this was
checked, not assumed to pass.

`moondream:latest` is now a second registered `ollama/*` entry in
`models.yaml` (`modalities_in: [text, image]`), gated by the exact same
`ollama_pulled_models()` per-tag availability check the text model
already goes through — the router only offers it once that exact tag
is genuinely pulled, not the instant any Ollama server answers (the
same real bug fixed for the text-only entry a few milestones back).
`tool_use: false` on the registry entry, matching what moondream's own
capabilities list actually declares rather than assumed from
comparison to larger vision models.

6 new/changed tests in `test_ollama_provider.py` (every pre-existing
test needed `await` once the translator went async; new coverage for
single and multiple images, and the images-key omitted when none are
present) — 458 → 460 Python tests. `ruff check`/`format --check`
clean. `docs/providers.md` updated with the real verification record.

**Next:** batching multiple concurrent inference requests (§3.6f,
still a deliberate deferral — real correctness risk); F1's real
distributed training infrastructure (needs real multi-node compute
this environment doesn't have); Gemini's Files API for long-video
input (named as real, deferred follow-up when native video-in
shipped — no API key in this environment to verify live against, same
limitation as the rest of the Google adapter); or a first pass at
code-signing/notarization for the desktop release bundles (needs a
real signing identity this environment doesn't have — likely stays
deferred).
