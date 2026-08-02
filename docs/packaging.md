# Packaging for humans: the CLI, the server, and the desktop app

Every chapter so far has been about the engine — providers, the agent
loop, multimodality, memory. This one is about the three skins that
actually put that engine in front of a person: the `sarva` command-line
tool, the FastAPI server a browser or the desktop app talks to, and the
Tauri-wrapped native app that bundles both into a double-clickable
`.dmg`/`.msi`/`.AppImage`. Three different surfaces, one shared engine —
`sarva.runtime.build_router`/`build_providers` back both the CLI and the
server, so neither skin can drift out of sync on what "zero-config"
means or how local providers get detected.

## The CLI: `sarva.cli`

`sarva chat "hello"` works with no configuration at all — the module's
own docstring states the design goal directly: "Zero-config by default:
with no `ANTHROPIC_API_KEY` set, everything routes to the offline
`MockProvider` so `sarva chat "hello"` always works." `sarva --version`
prints the real installed version (`importlib.metadata.version("sarva")`)
and exits — a genuinely common convenience that had no code path here
at all until it was noticed missing while poking at the CLI's own
`--help` output. Eleven commands, each doing one thing:

- **`chat MESSAGE [--image PATH] [--model ID] [--session NAME]`** —
  one-shot, tool-free, single-turn (`AgentLoop(tools=[],
  confirm=always_allow)`). The simplest possible entry point,
  deliberately with no tool access.
- **`run TASK [--workdir .] [--image PATH] [--model ID] [--auto] [--session NAME] [--mcp-server CMD]... [--mcp-header "Name: Value"]... [--mcp-env "NAME=VALUE"]...`**
  — the full agent loop with `BUILTIN_TOOLS` (files, shell) plus any MCP
  servers. `--mcp-server` is repeatable; each value is shell-split and
  connected via `connect_stdio_mcp_server` inside an `AsyncExitStack`
  (see the MCP chapter — `--mcp-env` is `--mcp-header`'s stdio
  counterpart, threading real environment variables like an auth token
  through to a local server subprocess), its tools appended to the
  built-in list. Without
  `--auto`, every destructive tool call stops for a real
  `typer.confirm(f"Allow {call.name}({call.arguments})?")` prompt;
  `--auto` swaps that for `always_allow`. `--image` landed later than
  `chat`'s own copy of the same flag — a real gap found by checking
  `/ws/chat` (`run`'s own server-side mirror) against what it actually
  read from the frame, not what `chat`/`/chat` already supported; see
  BUILD-JOURNAL.md.

  **`--model` closes a gap that predates every one of the above:**
  `AgentLoop.run(model_override=...)` — bypassing the router's default
  candidate selection entirely for a caller-named model — has been a
  real, spec-frozen parameter since T1, but neither `chat` nor `run`
  ever exposed a way to set it; there was no way to actually pick a
  model from the CLI at all. Fixed the same way `Router.pick()` was
  hardened alongside it: an unknown `--model` id is a hard, visible
  failure (`unknown model 'x' -- see 'sarva models' for the full list`,
  `run ended: failed — ...`, and — new — a nonzero exit code, so a
  scripted `sarva chat ... || handle_it` can actually detect it),
  never a silent substitution for some other model — see
  `UnknownModelError`'s own docstring in `sarva.providers.registry` for
  why that distinction needed its own exception type, not just a plain
  `LookupError`.
- **`models`** — lists every registry entry with `[x]`/`[ ]` marking
  whether it's currently available (API key present, Ollama reachable,
  a foundry checkpoint discovered — see the providers and foundry
  chapters).
- **`doctor`** — diagnoses the local setup: which provider API keys are
  set, Ollama reachability, whether the `sarva[foundry]` extra is
  installed and any checkpoints it discovers, and whether the web UI is
  built in for `sarva serve`. Backed by `sarva.runtime.run_diagnostics`,
  which reads the exact same env vars and calls the exact same helpers
  `build_router`/`build_providers` use — the report can never silently
  drift out of sync with what "available" actually means elsewhere.
  Every unchecked item is optional, not broken: a fresh, zero-config
  install is expected to fail most of these and still work fine via the
  Mock provider.
- **`eval [--model ID]`** / **`distill PROMPTS --model ID --out PATH`**
  — the eval harness and distillation pipeline, covered in their own
  chapters. Both resolve `--model` via `Registry.get()` directly (they
  need no modality/availability routing, unlike `chat`/`run`'s
  `Router.pick()`) — a real bug found by actually running `sarva eval
  --model bogus-id`: neither caught the resulting `KeyError` at all,
  crashing with a raw Python traceback instead of `chat`/`run`'s own
  clean `unknown model '...' -- see 'sarva models' for the full list`
  message. Fixed with a shared `_require_known_model` helper giving
  both commands the identical message and a clean nonzero exit,
  checked fast — before `eval` prints its own benchmark header, before
  `distill` writes anything — rather than failing mid-run.

  **A second, later bug found on the *write* side, not the read side
  every prior file-path fix in this project had covered:**
  `sarva.distill.save_jsonl()`'s plain `path.open("w")` raised a raw
  `FileNotFoundError`/`PermissionError` straight through Typer for a
  bad `--out` path — confirmed live with both a missing parent
  directory and a read-only one. Worse than every other file-path bug
  fixed so far: this one only surfaces *after* every real (potentially
  rate-limited, non-free) API call to the teacher model has already
  completed, throwing away the one artifact distillation exists to
  produce, with no way to recover it short of re-running the whole
  distillation from scratch. Fixed by wrapping the `save_jsonl(records,
  out)` call in `except OSError`, reusing the same `_print_file_error`
  helper `chat --image`/`speak --out`/`transcribe` already share.
- **`sessions list`** / **`sessions clear NAME`** — inspect or delete
  persisted chat sessions. **A real bug found by actually corrupting one
  session file among several good ones:** `SessionStore.load()` raises
  a pydantic `ValidationError` (a `ValueError` subclass, the same base
  `_sanitize()` already raises for a malformed on-disk filename) for a
  file that isn't valid JSON or doesn't match the expected shape, and
  `sessions list` called it with no error handling at all — one bad
  file crashed the whole command with a raw traceback, hiding every
  other, perfectly good session's listing too. Fixed by catching
  `ValueError` per entry and reporting `NAME  (corrupt or unreadable)`
  instead of aborting, so one bad file can't take the rest of the
  listing down with it.
- **`config set [--anthropic-api-key ...] [--openai-api-key ...] [--gemini-api-key ...]`**
  / **`config show`** / **`config unset [--anthropic-api-key] [--openai-api-key] [--gemini-api-key]`**
  — manage provider API keys in `~/.sarva/config.json` from the command
  line. `sarva.config.save_config`/`get_env` have backed the desktop
  app's first-run screen (`POST /config`) since it shipped, but a
  CLI-only user with no desktop app had no way to reach the same
  persistence at all — another instance of the "fully built,
  unreachable by a real user" shape this project keeps finding.
  `show` never prints an actual key value, only whether one is set and
  which source won (a real environment variable always beats a saved
  file). `set` reuses `save_config` directly, so it inherits the exact
  same owner-only file permissions the credential-exposure fix already
  established — verified as a real, second caller of that fix, not
  just the server's own path. `unset` (new `sarva.config.unset_config`)
  is `set`'s own missing counterpart, added in the same milestone
  rather than left as a fresh gap: a key saved by mistake, or a switch
  back to relying on a real env var, had no way to actually be removed
  short of hand-editing or deleting the whole file (losing every other
  saved key too) — a name that was never saved is a silent no-op, and
  a real environment variable is never touched, only the saved file.
- **`speak TEXT [--out speech.wav] [--voice NAME]`** — local
  text-to-speech, no API key, no network. See "Local speech" below.
- **`transcribe AUDIO_FILE [--model-size tiny]`** — local speech-to-text
  via `faster-whisper`, `speak`'s reverse. `sarva.audio.transcribe()`
  has backed `AudioToTextDegrader` since local speech shipped, but
  nothing ever exposed it as a command a real user could just run —
  the same "fully built, unreachable by any real user" shape this
  project keeps finding. Printed with `markup=False` (the transcript is
  real speech, externally-derived text, not this project's own strings
  — same discipline `chat`/`run` already apply to model output).
- **`serve [--host 127.0.0.1] [--port 8000]`** — starts the same server
  described below; its own docstring calls it "the surface a web UI or
  desktop app uses."

**A corrupted `~/.sarva/config.json` crashed nearly every command and
server endpoint — the broadest blast radius of any "unhandled
exception where a clean error belongs" bug found in this project so
far.** `get_env()` backs almost every provider-availability check
`build_router()`/`build_providers()`/`run_diagnostics()` make, and
`load_config()` parsed the file with a bare `json.loads()` — disk
corruption, an interrupted write, or a bad hand-edit crashed `doctor`,
`chat`, `run`, `models`, `eval`, `distill`, and all three `config`
subcommands with a raw `json.JSONDecodeError` traceback, confirmed live
on every one. The server side was worse in the usual two ways: `GET
/models`/`GET /doctor`/`POST /config` returned genuine unhandled 500s,
and `/ws/chat` crashed the whole ASGI call with no frame sent at all.
**Fixed with one new exception type, `sarva.config.ConfigError`**
(deliberately not a `ValueError` — reusing that base risked being
silently caught by an unrelated `except ValueError` scoped to a
different failure, like an invalid session name, with a misleading
message), raised once inside `load_config()` and handled at three
different points depending on what each skin can express: the CLI
wraps `_build_router`/`_build_providers` (closing every one of their
many call sites at once) plus `doctor`/`config show`/`set`/`unset`
individually; the server registers one `@app.exception_handler(ConfigError)`
covering every plain HTTP route for free (`/models`, `/doctor`, `POST
/config`), while `/chat` and `/ws/chat` catch it explicitly to keep
their own established failure shapes (`ChatResponse(state=failed,
detail=...)` and a `state_changed`/`run_done` frame pair) instead of
falling through to a generic error a WebSocket route can't even
receive. Verified the new tests are real: reverted the fix and watched
all eleven fail with the raw `JSONDecodeError`/`ImportError` before
re-applying.

**A bad file path crashed four commands with a raw traceback — `chat
--image`, `run --image`, `speak --out`, `transcribe`, and `distill` —
the same "unhandled exception where a clean error belongs" bug class
already fixed for `--model`/`--session`, just on file-path arguments
instead.** `Path.read_bytes()`/`read_text()`/`write_bytes()` all raise
a plain `OSError` subclass (`FileNotFoundError`, `PermissionError`,
`IsADirectoryError`) for a nonexistent path, an unreadable file, or a
missing parent directory, and none of these five call sites caught it.
Confirmed live on all four commands (`chat --image`/`run --image`
share `_load_image`, so fixing it once covers both): each printed a
full Python traceback instead of a clean message. Fixed with three
small shared helpers (`_read_bytes_or_exit`, `_read_text_or_exit`,
`_write_bytes_or_exit`) catching `OSError` in one place and printing
`cannot {read,write} {description} '{path}': {reason}` before exiting
1 — the same shape `--model`'s clean failure already established, just
generalized to any file argument rather than one `except` clause per
command. **Verified the new tests are real:** reverted the fix and
watched all four new tests fail with the raw, uncaught `OSError`
propagating (an empty `stdout`, the traceback going to `stderr`
instead) before re-applying.

**`_write_bytes_or_exit` (`speak --out`'s own write) had the identical
interrupted-write bug already found and fixed at five other real call
sites** (`WriteFileTool`, foundry checkpoint/tokenizer saving,
`distill.save_jsonl` — see the memory and foundry-training chapters).
It used a plain `path.write_bytes(data)`, which truncates the target to
0 bytes the instant it's opened, before a single byte of new content is
written — confirmed live, a crash mid-write destroys a previously-good
audio file with no error. Found by a sweep specifically re-checking
whether the earlier propagation milestone had actually found *every*
real call site (it hadn't — this was the sixth). Fixed via the same
shared `sarva.atomic_write` helper.

**Session persistence works identically for `chat` and `run`:** both
load prior history via `SessionStore().load(name)` before the turn, and
save the full transcript afterward — but only if the run actually
reached `done`. A run that errors, gets budget-exhausted, or is
cancelled mid-way is never persisted; a session file only ever reflects
turns that genuinely completed, not partial or failed state.

## The server: `sarva.server.app`

Two different endpoints for two different needs, and the module's own
docstring is explicit about why there are two rather than one:

- **`POST /chat`** mirrors `sarva chat` exactly — single-turn,
  non-streaming, tool-free. A plain REST request can't naturally pause
  mid-response for a confirmation round-trip, so this endpoint never
  needs to.
- **`WS /ws/chat`** mirrors `sarva run` — the tool-using surface. It
  streams the same `AgentEvent`s the CLI renders over the socket as
  JSON frames, and when a destructive tool call needs a decision, it
  sends a `needs_confirmation` frame and genuinely *waits* — the
  server's confirm callback is `reply = await
  websocket.receive_json(); return bool(reply.get("approved", False))`
  — for the client to send `{"approved": bool}` back before continuing.
  In `"auto": true` mode the same frame still gets sent (purely
  informational there — a client in auto mode must not reply to it).

This isn't just described in a docstring — the desktop app's own
`apps/desktop/src/App.tsx` implements exactly this handshake: it
branches on `event.type === "needs_confirmation"` and its
`respondToConfirmation` sends `{ approved }` back over the same socket,
matching the server side precisely. `GET /health` and `GET /models`
round out the REST surface for basic liveness/registry checks.

### `/ws/chat` had zero Origin validation — a real cross-site WebSocket hijacking (CSWSH) gap, not a hypothetical one

**The most severe finding across every sweep of this project so far,
by a different measure than the GRPO NaN bug or the duplicate-
`tool_call_id` confirm-gate bypass: this one is externally reachable by
an ordinary webpage, not just a malformed model response.** WebSocket
connections are **not** subject to the Same-Origin Policy or CORS the
way `fetch()` is — a browser will happily open `new
WebSocket("ws://127.0.0.1:8000/ws/chat")` from any page's own script,
regardless of which site that page came from, and send it whether the
tab is in the foreground or a background one the user forgot about.
Confirmed live: a `TestClient` WebSocket handshake carrying `Origin:
https://evil.example.com` was fully accepted, with a complete run
executing over it. Since `ws_chat` builds a real `AgentLoop` with
`BUILTIN_TOOLS` (file read/write, shell execution, memory), and honors
a client-supplied `"auto": true` by setting `confirm=always_allow`, an
ordinary webpage — no user interaction beyond having it open — could
silently drive real shell/file-tool access with every destructive-tool
confirmation gate pre-approved, purely because `sarva serve`/the
desktop app happened to be running locally. A classic cross-site
WebSocket hijacking (CSWSH) vector, distinct from every payload-shape
or exception-handling bug already fixed on this same endpoint — those
all assumed the caller was legitimate and just sent malformed data;
this one never checked whether the caller was legitimate at all.

**Fixed with a standard same-origin check before `websocket.accept()`
is ever called:** if the handshake's `Origin` header is present, it
must name the same `host:port` the request's own `Host` header does;
otherwise the connection is closed with code `1008` (Policy Violation)
before any frame is read or an `AgentLoop` is even constructed. `Origin`
being *absent* is deliberately not rejected — real browsers always send
it on a WebSocket handshake (mandated by RFC 6455 specifically for
browser clients), so its absence means a non-browser caller (a script,
a future first-party client) that the actual threat this closes — an
ordinary webpage silently driving the connection — structurally cannot
produce; rejecting those too would be a real regression with no
security benefit. Verified this doesn't break the legitimate case
either: the desktop app's own webview always loads its UI from the
exact same `http://127.0.0.1:8000` origin its server serves from
(`tauri.conf.json`'s `frontendDist`/`devUrl`), so its own WebSocket
connections are always genuinely same-origin and pass unaffected.
Verified the new tests are real: reverted the fix and watched the
cross-origin test fail with `DID NOT RAISE WebSocketDisconnect` (the
connection was still fully accepted) before re-applying.

### The confirmation-reply read itself had no timeout and no shape validation — a crash and a hang, one layer deeper than every previous `/ws/chat` fix

`ws_confirm`'s own `reply = await websocket.receive_json(); return
bool(reply.get("approved", False))` had two real gaps, distinct from
every other raw-JSON fix on this endpoint: those all guard the
*initial* payload frame; this is `AgentLoop`'s own confirm callback,
invoked from a completely different point deep inside the run, only
when a destructive tool call is actually pending.

**Confirmed live, two failure modes:** a malformed reply (a JSON array
instead of `{"approved": bool}`) made `reply.get(...)` raise an
uncaught `AttributeError`, crashing the whole ASGI call with no
`run_done` frame sent — the identical bare-disconnect shape every prior
raw-JSON fix on this endpoint targets, just reached through the
confirmation round-trip instead of the initial frame. Separately,
`receive_json()` had no timeout at all: a client that simply never
replies — deliberately, or by closing its tab in a way the browser
doesn't surface as a clean disconnect — hung the connection forever,
with the run stuck in `AWAITING_CONFIRMATION` and no recovery path.
This is the confirmation-layer counterpart to the already-fixed
hung-tool-call bug, but a genuinely separate call site: that fix
bounds `run_one`'s own `tool.run(...)` await; this one is `AgentLoop`'s
confirm callback itself, invoked before any tool runs at all.

Fixed by wrapping the read in `asyncio.wait_for` (a 300-second timeout
— deliberately much longer than the 90-second per-tool-execution
backstop, since a real person has to read a prompt and click a button,
not an automated process) and validating the reply is actually a dict
before reading `"approved"` from it. Both failure modes — timeout and
malformed shape — resolve to the same outcome: **a plain decline**, not
a crash and not a hang. This matches the "reject, don't guess"
discipline already applied elsewhere in this file (malformed
`--mcp-header`/`--mcp-env` entries): a destructive action given an
ambiguous or absent confirmation signal must never default to running.
Verified the new tests are real: reverted the fix and watched both fail
— the malformed-reply test with the raw, uncaught `AttributeError`; the
never-replies test with `AttributeError: <module 'sarva.server.app'>
has no attribute '_CONFIRM_TIMEOUT_SECONDS'` (the constant this fix
introduces) — before re-applying. All 35 pre-existing server tests pass
unchanged.

**A real gap found by checking what the desktop app actually calls, not
what the server merely supports:** `/chat` has taken optional
`image_base64`/`image_media_type` fields since images were wired into
the CLI, but `/ws/chat` — the *only* endpoint `App.tsx` ever calls —
never read them from the incoming frame at all. Since the desktop app
never calls `/chat`, this meant there was genuinely no way to attach an
image through the web UI, despite the CLI and the REST endpoint both
already supporting it. Closed by sharing one helper
(`_extra_content_blocks`) between both endpoints so they can't drift
apart again, wiring the same optional fields into `/ws/chat`'s frame,
and giving `App.tsx` an actual attach-image control: a hidden file
input behind a 📎 button, a removable chip showing the attached file's
name, and the image sent base64-encoded (via `File.arrayBuffer()`, not
`FileReader.readAsDataURL` — one fewer prefix-parsing step, and the
same path both real browsers and this project's own jsdom test
environment support identically) alongside the next message. **Verified
beyond the test suites:** a real `sarva serve` process hit with a real
`websockets` client sending a real image over `/ws/chat` completed
cleanly end to end, confirmed against the server's own request log.

**A malformed `image_base64` field crashed both endpoints, the same bug
shape already fixed here for an invalid `session` name.**
`base64.b64decode()` raises `binascii.Error` (a `ValueError` subclass)
on malformed input, and neither endpoint caught it: `/chat` returned a
genuine unhandled 500, and `/ws/chat` — worse — crashed the whole ASGI
call with no error frame sent at all, leaving the client to see a bare
`ClosedResourceError` on its next receive. Fixed by folding the
`_extra_content_blocks()` call into the same `try`/`except ValueError`
block that already handles the session-name case, so both failure
modes now get the identical clean treatment: a real
`ChatResponse(state="failed", ...)` on `/chat`, and a real
`state_changed` + `run_done` frame pair on `/ws/chat`. Verified the new
tests are real: reverted the fix and watched both fail with the raw
`binascii.Error` before re-applying.

**Three more ways to crash `/ws/chat` before it even reached the code
that validates `session`/`image_base64`, found by a fresh sweep of the
raw-JSON parsing `/ws/chat` alone does.** `/chat` never sees these,
because Pydantic validates `ChatRequest`'s field types before the
handler runs at all — `/ws/chat` parses a schema-less raw frame, so
nothing validated any of this until now. All three confirmed live with
a real `TestClient` WebSocket session before fixing, all three crashed
the whole ASGI call with the client seeing a bare `ClosedResourceError`
(the same failure mode already fixed for an invalid session name and a
malformed `image_base64`), and all three are now reported as the same
clean `state_changed` + `run_done` frame pair: **(1)** a non-JSON first
frame — Starlette's `receive_json()` does a bare `json.loads()` with no
error handling of its own, so a malformed frame raised an uncaught
`json.JSONDecodeError`. **(2)** valid JSON that isn't an object (a bare
list or string) — every `payload.get(...)` call downstream assumes a
dict, so this raised an uncaught `AttributeError`. **(3)** a non-string
`session` or `model` field (e.g. `{"session": 123}`, `{"model": ["a",
"b"]}`) — `SessionStore._sanitize()`'s regex match and
`AgentLoop.run()`'s `router.pick(override=model_override)` (a dict
lookup, several frames deep in the async generator) both raise a plain
`TypeError`, a sibling the existing `except ValueError` blocks never
covered. Fixed by wrapping the initial `receive_json()` call and
validating the payload shape before any field is read, explicitly
type-checking `model`, and widening the existing session/image_base64
`except ValueError` to `except (ValueError, TypeError)` — plus a small
shared `_send_failure()` helper (the four failure paths in this handler
had grown identical enough to be worth naming once). **Verified the new
tests are real:** reverted the fix and watched all four fail — three
with the raw, uncaught exception each case names above, and the fourth
(`model`) with the real `TypeError: unhashable type: 'list'` raised
several frames deep inside `AgentLoop.run()` itself — before
re-applying. All 29 pre-existing server tests pass unchanged. 4 new
tests, 566 → 570 Python tests.

**Both endpoints also gained an optional `model` field**, the REST/WS
counterpart to the CLI's own `--model` (see the agent-loop/providers
chapters for the `UnknownModelError` safety fix that motivated
building this properly rather than a bare pass-through): `ChatRequest`
and the WS frame both accept `"model": "<id>"`, threaded straight into
`AgentLoop.run(model_override=...)`. An unknown id is never a 500 or a
silent fallback to a different model — `/chat` now returns a new
`detail` field on `ChatResponse` (`null` on success) naming exactly
what went wrong, and `/ws/chat` clients already see the same message
in the `state_changed` frame's own `detail` field, since the full
event stream reaches the client regardless. Verified live against a
real running `sarva serve` process with real `curl` requests, not just
the test suite — a valid `"model": "mock"` and a bogus id both behave
exactly as documented.

**`App.tsx` now has a real model picker, closing that follow-up.** A
`<select>` next to the composer, populated from a new mount-time `GET
/models` fetch (best-effort — an unreachable server just leaves it
empty, the same graceful-degradation instinct as the `/doctor` fetch),
defaulting to "Auto" (`""`, sent as no `model` field at all — the exact
meaning omitting `--model` has on the CLI, not a separate sentinel the
server would need to know about). Unavailable models are listed too,
suffixed `(unavailable)`, rather than hidden — selecting one and
getting a real provider error back is more honest than a picker that
silently disagrees with what `sarva doctor`/`GET /doctor` would say.
**Wiring this in surfaced a small, separate, real gap in the same
file:** `App.tsx`'s `run_done` handler on a non-`DONE` state only ever
showed the generic `"run ended: <state>"` — `state_changed`'s own
`detail` field (the actual reason, e.g. an unknown model id) reaches
the client in the WS event stream already, it was just never read.
Fixed alongside the picker, the same fix the CLI and `/chat`'s
response already got. `.attached-image` also had no CSS at all until
now (a real gap from the image-attach milestone, noticed and closed
while touching this same UI area) — both it and the new `.model-picker`
are styled consistently with the rest of the app, dark-mode included.

**A hung "Thinking…" state with no recovery, found the same way — by
sweeping for one bug class already fixed elsewhere in this codebase
applied to code that hadn't been checked yet.** `App.tsx`'s WebSocket
handling set `onopen`/`onmessage`/`onerror`, but never `onclose` —
`streaming` only ever flipped back to `false` inside the `run_done` or
`onerror` branches. A socket that closes for any reason that doesn't
reliably fire `onerror` first (the server process killed mid-stream, a
reverse-proxy idle timeout, a raw TCP reset) left `streaming` stuck
`true` forever, and every composer control — the text input,
attach-image button, model picker, and send button — is gated on it,
so the *entire* UI locked up with no recovery except a full page
reload. Fixed with a real `onclose` handler, guarded by a `settled`
flag set by whichever of `run_done`/`onerror` fires first so a normal
completion or a real error is never overwritten by a redundant, less
specific close message — real WebSockets fire `onclose` after *any*
close, including a clean one the client itself initiated via its own
`ws.close()` call. **Verified the new regression test is real, not
just green:** reverted the fix and watched the exact new test fail for
the right reason (`streaming` still `true`, composer still disabled)
before re-applying, the same discipline already applied to the MCP
tool-name-escaping fix. 3 new tests (the raw-close recovery itself,
that `onerror`'s own message isn't overwritten by a later close, and
that a clean `run_done` never shows a spurious close error), 28 → 31
TypeScript tests.

**The same hung-UI symptom, reached through a different vector `onclose`
doesn't cover.** A fresh sweep, checking whether `onmessage` itself
could still hang the composer even with `onclose` in place, found it
could: `ws.onmessage`'s `JSON.parse(raw.data)` had no `try`/`catch`. A
`JSON.parse` throw inside a WebSocket's `onmessage` handler does
**not** trigger `onerror` or `onclose` in a real browser (or in jsdom)
— the exception is just logged to the console and the event silently
dropped, confirmed directly by sending a malformed, non-JSON frame and
watching `streaming` stay `true` with the composer still disabled, no
`onerror`/`onclose` handler ever firing. Fixed by wrapping the parse in
its own `try`/`catch`, treating a malformed frame the same way `onerror`
treats a real connection error: `settled = true`, a clear message
("received a malformed message from the server"), `streaming` reset,
and the socket explicitly closed rather than left open to receive more
frames after the protocol is already out of sync. **Verified the new
test is real:** reverted the fix and watched it fail with the raw,
uncaught `SyntaxError` from `JSON.parse` (not a normal assertion
failure) before re-applying. All 31 pre-existing tests pass unchanged.
1 new test, 31 → 32 TypeScript tests.

`GET /doctor` and `POST /config` are the two endpoints the first-run
onboarding screen (below) depends on — `/doctor` returns exactly what
`sarva doctor` prints, as JSON (reusing `run_diagnostics()` directly, so
the two can never drift out of sync), and `/config` persists whichever
provider key the caller supplies via `sarva.config.save_config`.

**Serving the web UI is genuinely optional, not a hard dependency of
the API:** if `core/sarva/server/static/` exists, it's mounted at `/`
via `StaticFiles(..., html=True)` so `sarva serve` alone gives a
complete browser experience; if it doesn't exist, the server is simply
API-only, with nothing breaking either way.

## First-run guided setup — a real gap between what was promised and what shipped

T4's own definition of done, and the README's own quickstart text, have
both promised "guided first-run offers (a) 'Free & private' → pulls a
local model, or (b) 'Frontier quality' → paste an API key" since T4 —
but until now, `App.tsx` was a bare chat window with no such flow at
all. A non-technical user double-clicking the built app got a chat box
with no path to configure anything, the exact opposite of the mission's
own "non-developer completes install→first answer in <3 minutes, no
terminal" promise.

**The real missing piece wasn't the UI — it was persistence.** Every
provider's SDK client (`anthropic.AsyncAnthropic()`, `openai.AsyncOpenAI()`,
`genai.Client()`) reads its API key from real process environment
variables internally; a key entered once in any UI had nowhere to
survive past that single process's lifetime. `sarva.config` adds a real
file, `~/.sarva/config.json` (the same `~/.sarva/` home session storage
already uses), with one deliberate precedence rule: a real environment
variable always wins over a saved config value, so an explicitly
exported shell key is never silently overridden by a stale file.
`sarva.runtime`'s `get_env()` — used everywhere `os.environ.get(...)`
used to appear for the four provider-key names — checks both.

**A config-file-only key had to actually authenticate, not just "look
configured":** `build_providers()` now constructs every SDK client with
an *explicit* `api_key=...` sourced via `get_env()`, rather than the raw
SDK constructors' own (config-file-blind) `os.environ` auto-detection —
verified directly by checking the constructed client's own `.api_key`
attribute, not just that `build_providers()` doesn't crash.

**A real gap found later, by checking an actual saved file's mode bits
rather than assuming `write_text` was fine:** `~/.sarva/config.json`
holds plaintext provider API keys, but was being written with whatever
the platform default happened to be — `0644` on this machine's real
umask, world-readable. `save_config` now creates the file via
`os.open(..., 0o600)` directly (no create-then-chmod window where it's
briefly exposed) and `chmod`s it explicitly afterward too, so a file an
older version already wrote insecurely gets tightened on the very next
save. Real and meaningful on POSIX (macOS/Linux); on Windows,
`os.chmod` only toggles the read-only attribute, not genuine per-user
ACL isolation — named honestly as a real, separate, deferred gap rather
than assumed equivalent.

**A real, separate race found and closed later: two concurrent callers
could silently lose each other's saved key.** `save_config`/
`unset_config`'s read-modify-write (`existing = load_config(path);
existing.update(values); _write_config(path, existing)`) was never
locked — two callers racing this sequence (the CLI and the desktop
app, or two CLI invocations) can both read the same "before" state,
each add a different key, and each write their own merged dict back;
the second write silently overwrites the first's key with no error.
Confirmed live with two real threads: a genuine `OPENAI_API_KEY` saved
by one caller vanished entirely after a second, concurrent caller's own
unrelated save completed. A different bug class from the already-fixed
atomic-write-on-save gap — that fix makes each *individual* write
crash-safe; it does nothing to serialize *two separate* read-modify-
write cycles against each other. Fixed with a real, cross-process
exclusive lock (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows)
held for the entire read-modify-write critical section, on a dedicated
sibling `.lock` file — never the config file itself, so it never
interferes with `_write_config`'s own atomic-rename mechanism, and a
plain reader (`load_config`/`get_env`) never needs to acquire it at all
(the atomic write already guarantees a reader only ever sees a complete
old or new version). **Honestly scoped the same way the permissions fix
above is:** the Windows branch uses a real, standard Windows locking
API, but — unlike this project's TTS Windows branch, which a genuine
`windows-latest` CI job exercises end to end — no CI job actually runs
`config.py`'s tests on real Windows (`windows-audio` only runs
`test_audio.py` there), so it's implemented and reasoned through
correctly but not live-verified the same way. Verified the new tests
are real: reverted the fix and watched both fail — one with
`ImportError: cannot import name '_exclusive_lock'`, the other with the
real lost-update (`KeyError: 'OPENAI_API_KEY'`) — before re-applying.

**The lock itself had a real Windows-specific bug, found by a much
later sweep re-examining this exact mechanism after mirroring it into
`SessionStore`.** `_exclusive_lock` re-opened its dedicated `.lock`
file with truncating `"wb"` mode and rewrote the one-byte marker on
*every* acquisition, not just the first. Harmless on POSIX — `flock()`
is purely advisory and doesn't interact with ordinary reads/writes at
all — but `msvcrt.locking()` on Windows is a *mandatory* byte-range
lock the OS enforces against any I/O to that region from other
processes, including a plain `write()` that never itself calls
`locking()`. A second, contending caller's truncating write targets
exactly the byte a first caller already holds locked: that write fails
with a sharing-violation `OSError` *before* the second caller ever
reaches its own `msvcrt.locking()` call — contention crashing the
second caller instead of making it wait, the opposite of what this
function exists to do. Not live-verified here either (the same honest
Windows gap named above), reasoned from documented Win32/CRT mandatory-
locking semantics rather than assumed correct by analogy to the POSIX
branch. Fixed with `os.open(..., O_CREAT)` (no `O_TRUNC`): the marker
byte is written once, only if the file turns out to be empty, and
every later acquisition just opens and locks the already-populated
file without writing to it at all. The same fix landed in
`SessionStore.locked` (see the memory chapter), which had mirrored the
original, buggy version faithfully. **Verified via mtime, not inode or
content:** both the buggy and fixed versions write the identical single
byte value, so a file's content and inode stay the same either way —
the property that actually distinguishes "rewrote the same byte" from
"never touched it again" is whether a write syscall happened at all,
observable as whether `mtime` advances across repeated acquisitions.
Reverted and watched the new test fail with a real, later `mtime` on
every one of five repeated acquisitions before re-applying.

`Onboarding.tsx` is the screen this makes possible: on mount it polls
`GET /doctor`; if any provider (including a reachable Ollama) is already
configured, it completes immediately and the user never sees it. If
not, it offers exactly the two documented choices — Ollama instructions
with a live "Check again" re-poll, or a key-paste form that `POST
/config`s and shows the fresh `/doctor` result — plus an honest "Skip
for now" escape hatch (remembered in `localStorage`) for anyone who just
wants the always-available Mock provider.

## The web UI and the desktop app

`apps/desktop/` is a React 18 + TypeScript + Vite project (`npm run
build` = `tsc -b && vite build`) that plays two roles: it's the source
for `core/sarva/server/static/` (a **literal, checked-in copy** of its
build output — `scripts/build-web.sh` runs the build, then wipes and
recopies `dist/` into the static directory, so `sarva serve` needs no
Node toolchain at runtime at all), and it's wrapped by `apps/desktop/
src-tauri/` into the native desktop app. Rebuilding that copy is a
manual step (`./scripts/build-web.sh`), not CI-automated — CI only
*checks* the copy is fresh, not that anyone remembered to run the
script.

**The desktop app's whole job is spawning and reliably killing one
sidecar process.** `src-tauri/src/lib.rs` starts the frozen Python
server as a Tauri sidecar (`sidecar("sarva-server").args(["serve"])`) on
launch. Killing it cleanly turned out to need more than the obvious
`child.kill()`: PyInstaller's onefile bootloader forks a real grandchild
process that call alone can't reach, so `kill_sidecar` reaps it on every
platform now — `pgrep -P <pid>` + `kill -9` on macOS/Linux,
`taskkill /F /T /PID <pid>` (Windows' native process-tree kill) on
Windows. **A real bug this closed, not just a documented gap:** until
now the grandchild-reaping logic was unconditionally `#[cfg(unix)]`-
gated, so even the ordinary graceful window-close path — which already
fires identically on every platform via Tauri's `CloseRequested` —
silently orphaned the frozen server on Windows, still holding the port.
**One piece genuinely still doesn't have a Windows equivalent, for a
real, checked reason:** catching an abrupt SIGTERM/SIGINT-equivalent
(the app killed directly rather than closed gracefully) needs Win32's
console-control-handler API, which only delivers events to a process
with an attached console — this app is `windows_subsystem = "windows"`
in release builds specifically to avoid popping one. A real fix would
need deeper Win32 message-loop hooking (`WM_QUERYENDSESSION`), left
open and explained rather than silently assumed away — this environment
also has no Windows machine to verify runtime behavior on, only CI's
`windows-latest` `cargo check` job, which confirms the code compiles
correctly for the target, not that it behaves correctly at runtime.

**A real bug found by actually checking what a `log::info!`/`log::warn!`
call does with no logger registered** (a standalone `log`-crate repro,
not just reading the plugin's source): it's a genuine silent no-op —
`log::max_level()` reads back as `Off`, confirmed directly. This
mattered here because `tauri_plugin_log`'s registration was gated
behind `cfg!(debug_assertions)`, so a real release build never
registered a logger at all — every sidecar stdout/stderr line, and the
one place a sidecar crash is ever noticed at all
(`CommandEvent::Terminated`), vanished with zero record: no console
line, no log file, nothing a user or support engineer could look at.
The module's own doc comment claims "the failure surfaces as a log line
from the sidecar process" — true only in debug builds until this fix.
Fixed by registering the plugin unconditionally; `Builder::default()`'s
own default targets already write to both stdout *and* a real log file
in the platform's app-log directory, so releases get real, persisted
sidecar diagnostics with no extra target configuration needed. Verified
with `cargo check --locked` (this project's own established
verification depth for `src-tauri/`, since there's no Windows machine
or GUI runtime available here — see the grandchild-reaping fix above
for the same caveat) plus the standalone `log`-crate repro proving the
no-logger behavior this bug depended on.

Real, working cross-platform installers do exist:
`.github/workflows/release-bundle.yml` ("Release bundle (unsigned)")
builds `.dmg` (macOS), `.msi`/`.exe` (Windows), and `.AppImage`/`.deb`
(Linux) on a `[macos-latest, ubuntu-latest, windows-latest]` matrix,
triggered manually or by pushing a `v*` tag — the same mechanism behind
this project's own `v0.1.0` release. A tag push additionally creates a
**draft** GitHub Release, deliberately never auto-published (a maintainer
has to explicitly publish it — pushing a tag alone was never meant to be
enough to make something publicly visible on its own). The name in
parentheses is the honest part: no code signing or notarization yet, so
an unsigned build will trigger Gatekeeper (macOS) or SmartScreen
(Windows) warnings — a known, documented gap, not glossed over.

## Verified, not assumed

Every specific claim above — the exact confirmation-frame handshake,
the `static/` copy relationship, the Unix-only kill logic and the
Windows gap, the release workflow's real artifact types — was checked
against current source (`cli.py`, `server/app.py`,
`src-tauri/src/lib.rs`, `release-bundle.yml`, `App.tsx`,
`build-web.sh`) rather than written from memory of having built it,
the same discipline that caught two real stale docstrings while writing
earlier chapters — and, separately, a real gap in the built artifact
itself: neither `pyproject.toml` declared a `license` field, so a real
built wheel's METADATA had no license information at all despite this
being a genuinely MIT-licensed repo with a real `LICENSE` file, found
by inspecting the actual wheel rather than assuming the metadata
matched the repo. Both now declare `license = "MIT"` (verified in the
built `METADATA`: `License-Expression: MIT`) and `license-files =
["LICENSE"]`, each package keeping its own in-tree copy of the
repo-root `LICENSE` — hatchling's `license-files` globs can't reach
outside the project directory, confirmed empirically (`../LICENSE`
built without error but silently bundled nothing) before landing on
the working fix. The bundled text is verified byte-identical to the
repo root's, and a CI check pins both the metadata and the file on
every push.

The onboarding flow specifically was verified beyond its own test
suite: a real `sarva serve` process, hit with real `curl` requests —
`POST /config` with a test key, confirming `~/.sarva/config.json`
genuinely existed on disk with the right content afterward (then
cleaned up), and the following `GET /doctor` call reflecting it as
configured. `apps/desktop`'s full production build (`npm run build`,
`tsc -b`) was run for real, not assumed to still pass.

## Local speech: `sarva.audio`

T2's own definition of done has promised "audio in/out (local
Whisper/TTS)" since T2 — `AudioToTextDegrader` (the multimodal chapter)
always reported "could not be transcribed" regardless of input until
now, and there was no TTS anywhere. `sarva.audio` closes both
directions, with two deliberately different substrate choices:

- **TTS shells out to the OS's own bundled engine** (macOS `say`,
  Linux `espeak`/`espeak-ng`) rather than a Python library. `pyttsx3`,
  the common cross-platform wrapper, was tried and rejected: it pulled
  in the entire `pyobjc` framework suite (100+ packages) on macOS just
  to reach the same `say` command this module now calls directly.
- **STT uses `faster-whisper`** (a new, genuinely optional
  `sarva[audio]` extra) — no OS-native local speech recognizer exists
  to shell out to the way TTS has one. Its own hard dependencies pull
  in no `torch`, so this stays a lightweight extra alongside
  `sarva[foundry]`, not a second heavy ML stack.

**A real bug found empirically while building this:** macOS `say`'s own
DEFAULT voice (invoked with no `-v`) produced near-silent,
sub-10-millisecond output for real text in this environment — confirmed
directly with `afinfo`, not assumed — while an explicitly named,
always-bundled voice (`Samantha`) produced correct, full-length audio
for identical text. `synthesize()` always passes an explicit voice for
exactly this reason.

`AudioToTextDegrader` now attempts real transcription first when
`sarva[audio]` is installed, falling back to the original honest
metadata-only message only when the extra is missing or transcription
genuinely fails on that specific audio — never a fabricated transcript.
`sarva doctor`/`GET /doctor` gained two checks ("Speech-to-text (local
Whisper)", "Text-to-speech (local)") from the same `sarva.audio`
functions this module uses, so they can never drift from what's
actually available. `sarva speak` is the CLI's own reachable surface
for TTS — closing the same "fully built but unreachable by any real
user" gap this project has named and fixed before.

**STT had the identical gap for a while longer:** `sarva.audio.
transcribe()` backed `AudioToTextDegrader` from the start, but nothing
exposed it as a command — `sarva transcribe AUDIO_FILE` closes it,
`speak`'s direct reverse. **A real bug caught by this command's own
test, not shipped:** the first version's `except ImportError` handler
printed the real "faster-whisper is not installed -- pip install
sarva[audio]..." message unescaped — Rich's markup parser silently
swallowed the literal `[audio]`, the identical class of bug `doctor`'s
dynamic detail text was fixed for earlier in this same file. Fixed
with `escape()`, and `speak`'s equivalent (bracket-free today, but
one edit away from the same bug) got the same defensive fix alongside
it. Verified live end to end, not just the test suite: a real
`sarva speak "..."` output piped straight into a real
`sarva transcribe`, real words back.

**Both non-Windows TTS branches verified against real installed
binaries, not just documented CLI shapes:** the `say` branch runs
unconditionally on real macOS; the `espeak-ng` branch was verified too
— installed via `brew install espeak-ng`, then exercised for real by a
test that hides `say` specifically (macOS's own Darwin branch would
otherwise always win) so the actual espeak subprocess call runs, not a
mock. A full `espeak-ng` → `faster-whisper` round trip (real
synthesized speech, transcribed back, words checked) passed the same
way the `say` round trip already had.

**A corrupted audio attachment could crash the whole process with a
native SIGBUS — the identical bug class already found and fixed for
the video degrader.** `faster_whisper.audio.decode_audio()` uses the
exact same PyAV/libavcodec `av.open`/`container.decode` call that
crashed the video decoder; confirmed directly, not assumed from that
earlier fix alone, by fuzzing a real WAV (random byte flips
concentrated in the header, across several seeds): multiple fuzzed
variants killed the process outright with a real SIGBUS (signal 10),
not a Python exception, when fed through `sarva.audio.transcribe()`.
No `try`/`except`, however broad, can catch a native memory fault.
Fixed the same way as the video degrader: the actual decode step now
runs in an isolated subprocess (`sarva._audio_decode_worker`, spawned
via `python -m`), with `subprocess.run`'s own `timeout=` handling
kill-and-reap automatically (no manual `proc.kill()`/`proc.wait()`
needed the way the `asyncio`-based fixes elsewhere in this project
require). Only the risky decode step moved to a subprocess — the
expensive-to-reload whisper model itself stays cached in-process via
`_whisper_model`'s existing `lru_cache`, since `WhisperModel.transcribe()`
accepts an already-decoded numpy array directly and skips its own
internal decode step when given one. `transcribe()`'s and
`AudioToTextDegrader`'s existing behavior for an ordinary (non-crashing)
bad-audio failure is unchanged — both still degrade cleanly, just now
via a clean `RuntimeError` from the isolated decode step instead of
whatever exception the in-process call used to raise.

**`synthesize()` itself could crash with a raw subprocess error, found
by actually running it against the real `espeak-ng` binary with a bad
`--voice`.** `espeak-ng` genuinely exits 1 for an unrecognized voice
name (confirmed directly: `espeak-ng -v bogus-name "hello"` → `Error:
The specified espeak-ng voice does not exist.`), and the raw
`subprocess.CalledProcessError` propagated uncaught all the way
through `sarva speak --voice bogus-name` — a bare Python traceback,
since `speak`'s own CLI command only ever caught `RuntimeError` (the
"no engine at all" case). Fixed in `synthesize()` itself, not with a
second `except` clause in the CLI: any `CalledProcessError` from
whichever engine actually ran becomes a `RuntimeError` carrying the
engine's own real captured stderr (`"text-to-speech engine failed:
Error: The specified espeak-ng voice does not exist."`) — the one
piece of information that actually explains what broke, not dropped in
translation. Verified against the real installed binary both at the
library level and through the actual CLI, and confirmed the new tests
are real by reverting the fix and watching both fail for the right
reason before re-applying.

**`synthesize()`'s three `subprocess.run` calls had no timeout at
all — a real bug found by actually running it against a hung TTS
binary, not a theoretical concern.** The sibling STT decode path
(`_decode_audio_isolated`) explicitly mirrors `RunShellTool`'s own
timeout fix, but that reasoning was never applied to TTS. Confirmed
live: a fake `say` binary that never returns hung `synthesize()`
indefinitely, with no way to recover — this module's own docstring
already names the real threat model this matters for ("an agent
speaking its own output"), so `text` can be arbitrary, potentially
adversarial model-generated content, not just a short human-typed
phrase, and a wedged or resource-exhausted OS speech engine has no
guard at all. Fixed by adding `timeout=60` (mirroring `RunShellTool`'s
own 60-second timeout) to all three `subprocess.run` calls (`say`,
PowerShell/SAPI, `espeak`/`espeak-ng`), and a new `except
subprocess.TimeoutExpired` clause in `synthesize()` turning it into the
same clean `RuntimeError` shape the "engine itself fails" case above
already produces. `subprocess.run`'s own `timeout=` already kills and
reaps the child on expiry — no manual cleanup needed, the same
"`TimeoutExpired` handles it" discipline the STT decode path already
established. Verified the new test is real: reverted the fix and
watched it fail with a raw `TypeError` (the fake `subprocess.run` in
the test expects a `timeout` keyword argument the reverted code never
passes) before re-applying.

**Windows had no engine at all until now** — this module's own
docstring named it as genuinely unimplemented, not just unverified.
It's closed the same way the other two branches are: shell out to an
OS-bundled engine, `System.Speech.Synthesis` (SAPI) via PowerShell,
rather than pulling in a third-party TTS library. The text to speak is
written to a temp file and read back inside the PowerShell script via
`Get-Content` — deliberately never interpolated into the command
string, so arbitrary (e.g. model-produced) text passed to `sarva
speak` can never be interpreted as PowerShell syntax; a dedicated
hermetic test pins exactly that property by inspecting the real argv
and file contents a monkeypatched `subprocess.run` receives. This
project has no Windows machine to develop against directly, so what
actually verifies the branch runs and produces real audio is a new
`windows-audio` CI job — a genuine `windows-latest` GitHub Actions
runner running `sarva.audio.synthesize()` for real, the same
discipline `release-bundle.yml`'s own Windows matrix leg already
established for the desktop bundle.

## CLI conformance tests

Until now, only `doctor` had `typer.testing.CliRunner` coverage
(confirmed by `grep -rln "CliRunner" tests/` returning exactly one
file) — every other command was only ever exercised indirectly, through
the library functions it wraps, never through `app` itself the way a
real user actually invokes it. `tests/conformance/test_cli.py` runs
`chat`, `run`, `models`, `eval`, `distill`, and `sessions list`/`clear`
through the real Typer `app`, zero-config (Mock provider only) — the
same "always works with no API keys" guarantee the module's own
docstring makes, now actually exercised at the command-line boundary
rather than only at the function-call boundary underneath it. Sessions
are isolated per test by monkeypatching `sarva.memory.session.
DEFAULT_SESSIONS_DIR` to a `tmp_path`, so no test ever touches a real
`~/.sarva/sessions` on the machine running them.
