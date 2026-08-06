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
- **`config set [--anthropic-api-key ...] [--openai-api-key ...] [--gemini-api-key ...] [--google-api-key ...]`**
  / **`config show`** / **`config unset [--anthropic-api-key] [--openai-api-key] [--gemini-api-key] [--google-api-key]`**
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

### `GOOGLE_API_KEY` was a fully first-class credential name everywhere except the three places a user would actually save it

A much later fresh-eyes sweep found the "fully built, unreachable by a
real user" shape recurring one layer deeper, inside `config set`/
`config unset`/`POST /config` themselves rather than around them.
`sarva.config.KNOWN_KEYS` has listed `GOOGLE_API_KEY` (the exact
env-var name Google's own SDK docs have historically used) alongside
`GEMINI_API_KEY` since it was added — `get_env`'s own Gemini fallback
already tries both, and `config show` already correctly reports
whichever one is set. But `server/schemas.py`'s `SaveConfigRequest`
only ever declared three fields, `config set`/`config unset` only ever
exposed three flags, and `POST /config`'s own route handler only ever
built a three-entry `values` dict — `SaveConfigRequest`'s own docstring
had said "four provider-key names" since it was written, while
defining three.

Because Pydantic ignores unknown request fields by default, `POST
/config` with `google_api_key` set didn't error — it silently
vanished with a `200 OK`, the exact "user thought they sent it, it got
silently dropped" failure mode this project explicitly guards against
elsewhere (`--mcp-header` parsing rejects a malformed entry rather
than dropping it). Confirmed live: a real `POST /config` call with
`google_api_key` set left the saved config file empty, and `sarva
config set --help` showed no `--google-api-key` flag to reach the CLI
path at all. Reachable by a completely ordinary user — anyone who
already exported `GOOGLE_API_KEY` and tries to persist it via the
desktop app's onboarding screen or `sarva config set`, no adversarial
input needed.

Fixed by adding the missing field/flag to all three write paths
together — `SaveConfigRequest.google_api_key`, `POST /config`'s
`values` dict, and `--google-api-key` on both `config set` and `config
unset` — closing the sibling-propagation gap in one pass rather than
one write path at a time. Verified live: the same repro now correctly
saves, shows, and removes a `GOOGLE_API_KEY`. Verified by reverting and
watching both new tests fail with the literal old bug's own shape: an
empty saved config after a real `POST /config`, and a rejected,
unrecognized `--google-api-key` CLI flag. 2 new tests, 792 → 794
Python tests.
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

**`sarva models` had a fourth, not-yet-covered instance of the
"unescaped external text" bug class this project has already fixed
three separate times** (`doctor`'s own `check.detail`, an MCP server
command repr, MCP tool names — see the MCP chapter). A local foundry
checkpoint's `id`/`display_name` come straight from the checkpoint
bundle's own DIRECTORY NAME (`model_info_for_bundle()`) — fully
user-controlled (a user's own `--output-dir` choice, or a shared/
downloaded checkpoint folder) — and `models_cmd` printed them with no
`escape()` call at all, unlike `doctor`, which already escapes the
identical checkpoint-name data reaching it through `DiagnosticCheck.
detail`. Confirmed live: a real checkpoint bundle directory named
`chatbot-v2 [draft]` — an ordinary, non-adversarial naming choice, not
a contrived attack — had `[draft]` silently swallowed from both the
printed model id and display name, since Rich interpreted it as an
unknown style tag. Fixed by escaping both fields, matching the
already-established pattern. Verified live the identical checkpoint
name now renders literally. Verified by reverting and watching the new
test fail with the exact swallowed value reproducing itself. 1 new
test, 715 → 716 Python tests.

**A fifth and sixth instance, found one round later by giving
`eval`/`distill` the identical sweep `sarva models` just got:** four
call sites echoed a model id back with no `escape()` call —
`_require_known_model`'s own "unknown model" error (shared by both
commands), `eval`'s "skip (provider not configured)" line, `eval`'s
per-model accuracy line, and `distill`'s "provider not configured"/
"Distilling N prompts from..." lines — the same untrusted foundry-
checkpoint-directory-name source as the `sarva models` fix directly
above. Confirmed live: running `sarva eval` with no `--model` filter
(the documented default: grade every available model) against a
foundry checkpoint named `chatbot-v2 [draft]` silently dropped
`[draft]` from the printed "skip" line — reachable through completely
ordinary, default use, not a flag a user has to opt into. A further
check while verifying this fix's own completeness found a sixth,
differently-sourced instance in the same command: `distill`'s
`DistillationError` handler embeds the underlying `ProviderError`'s
own text verbatim — genuinely external text (whatever a real provider
SDK/API actually said) this project doesn't control the shape of, not
a checkpoint name this time. Confirmed live the identical way: a
provider error containing `[bold red]...[/bold red]` was silently
swallowed. Fixed by escaping all six sites. Verified live each one now
renders its bracket-laden text literally. Verified by reverting and
watching all four new tests fail with the exact swallowed values
reproducing themselves. 4 new tests, 716 → 720 Python tests.

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

### That same fix's own `bool(...)` cast silently approved a destructive action given a non-boolean truthy value — the exact ambiguity it was written to reject

A much later fresh-eyes sweep found a real gap inside the fix directly
above: `return bool(reply.get("approved", False))` is a Python-
truthiness cast, not the strict boolean check the surrounding code's
own established contract (`{"approved": bool}`, documented in this
module's own docstring) actually promises — `bool("false")` is `True`,
since any non-empty string is truthy in Python. Confirmed live: a
client sending `{"approved": "false"}` — a **JSON string**, not a JSON
boolean, exactly what a form/select's `.value` produces, or any client
that serializes booleans as strings with no explicit `=== "true"`
conversion — got a destructive `write_file` call **approved**, the
opposite of the sender's intent, with no error and no signal anything
was wrong. This directly contradicts the timeout/non-dict fix's own
stated philosophy just above: "a destructive action given an ambiguous
or absent confirmation signal must never default to running" — a
non-boolean truthy value isn't ambiguous by accident here, it was
silently *accepted* as approval.

Reachable, not hypothetical: `/ws/chat`'s own docstring documents
`{"approved": bool}` as the wire protocol for *any* client to
implement, not a private internal API. The first-party desktop app
avoids this by accident — TypeScript enforces the boolean type at its
one `send(JSON.stringify({ approved }))` call site — but nothing on
the server enforces or even checks it; any other client (a test
script, a future mobile client, a third-party integration) is one
string-typed variable away from silently auto-approving every
destructive action a user believes they just declined. Fixed by
replacing the truthiness cast with a strict identity check,
`reply.get("approved") is True`: only the real JSON boolean `true`
approves, and every other value — a string, a number, a missing key —
fails closed the same way the timeout and non-dict cases already do.
Verified live the identical `{"approved": "false"}` reply now correctly
declines. Verified the new test is real: reverted the fix and watched
it fail with the literal old bug's own shape (`is_error: False`, the
file written) before re-applying. 1 new test, 764 → 765 Python tests.

### That same `bool(...)` gap existed two lines above it too — in `auto`, materially more severe than the `approved` fix it sits right next to

The very next round's fresh-eyes sweep found the identical truthiness-
cast bug in the same handler, on the two other caller-supplied JSON
values sitting just above `ws_confirm`: `auto = bool(payload.get
("auto", False))` and `verify = bool(payload.get("verify", False))`.
`auto` selects `always_allow` (zero user confirmation, ever) over the
real `ws_confirm` gate for *every* destructive tool call in the whole
session — a client sending `"auto": "false"` (a JSON string, the exact
same realistic shape the `approved` fix's own comment already names)
got every destructive tool call auto-approved with no confirmation
prompt at all, the opposite of what sending a falsy-looking value would
reasonably suggest. Confirmed live: a scripted `write_file` turn with
`"auto": "false"` and the client deliberately never replying to any
prompt completed in milliseconds with the file already written; the
same turn with the real boolean `False` correctly blocked on
`ws_confirm` and timed out as a decline. Materially more severe than
the `approved` bug it sits two lines above: that one could misapprove
only one already-surfaced confirmation prompt; this one disables the
confirmation system for the entire session, silently, for a client that
was actively trying to ask for it. Fixed the identical way: `auto =
payload.get("auto") is True`, `verify = payload.get("verify") is True`.
Verified live the identical `"auto": "false"` request now correctly
blocks on confirmation. Verified the new test is real: reverted the fix
and watched it fail with the tool having already run
(`is_error: False`, no confirmation reply ever sent) before
re-applying. 1 new test, 765 → 766 Python tests.

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

**A much later fresh-eyes sweep found `session=""` still fell through
every one of the fixes above** — a genuinely different failure than
any of the three JSON-shape gaps just closed, since `""` is a
perfectly well-typed string. `store.locked(session)` (`SessionStore.
locked`) special-cases `name is None` specifically; `""` is not
`None`, so it reached `SessionStore._sanitize()`'s regex and raised
`ValueError: invalid session name: ''` — even though every OTHER use
of the same value in both handlers (`store.load(session) if session
else []`, `if session and state == DONE: store.save(...)`) already
treats `""` identically to no session at all. Confirmed live: an
otherwise-identical request succeeded with no `session` field and with
a real named session, but failed outright with `session=""` — a
completely ordinary client pattern (a form/state field initialized to
`""` and always serialized, rather than omitted or sent as `null`)
hits this on literally every chat request until a client happens to
special-case it away. Fixed by normalizing once, in each handler
(`session = req.session or None` for `/chat`, `session = payload.get
("session") or None` for `/ws/chat`), so every downstream use — `locked()`
included — agrees on what "no session" means, instead of a truthy
check in some places and an identity check in another. Verified live
both endpoints now succeed identically whether `session` is omitted,
`null`, or `""`. Verified by reverting and watching both new tests
fail with the literal old bug's own message, `invalid session name:
''`. 2 new tests, 762 → 764 Python tests.

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

### The standalone TypeScript SDK had the identical shape of gap the desktop app's own frame-parsing fix (above) had already closed — a raw parse error instead of the documented error type

`sdks/typescript/src/client.ts`'s `requestJson()` — the shared helper
every REST method (`health`, `models`, `doctor`, `saveConfig`, `chat`)
routes through — called `response.json()` unconditionally, *before*
`response.ok` was ever checked. `sarva.server.app` registers a global
exception handler for exactly one exception type, `ConfigError`; any
other unhandled exception falls through to Starlette's own default
handler, which returns a **plain-text** 500 body, not JSON — a real,
reachable case, not a hypothetical one: `POST /config` → `save_config_
route` has no exception handling of its own around `save_config()`,
and `save_config()` → `config.py`'s `_exclusive_lock` → `os.open()`
raises a plain `PermissionError` (an `OSError`, not `ConfigError`)
whenever `~/.sarva` isn't writable — a read-only home directory, a
container mount, an ordinary permissions issue, no adversarial intent
needed. Calling `.json()` on that plain-text body raised a raw
`SyntaxError` instead of constructing the documented `SarvaApiError`
class every caller is meant to catch to get the real `status`/`body` —
discarding both entirely.

Confirmed live end to end, not just reasoned about: a scratch
`~/.sarva` directory `chmod 500`'d (read-only), a real `sarva serve`
process started against it, `curl -i -X POST /config` returning the
real raw traceback and a `text/plain` 500, then the actual *compiled*
SDK (`dist/index.js`, not just the source) calling `client.saveConfig
(...)` against that same real server and throwing `SyntaxError`
instead of `SarvaApiError`. `saveConfig()` is a documented, first-class
`SarvaClient` method (the design doc's own desktop first-run "paste an
API key" flow), and the SDK's own pre-existing test suite only ever
exercised the JSON-error-body case, so this gap in the non-JSON case
was never caught.

Fixed by reading the response body as text exactly once, then trying
to parse it as JSON and falling back to the raw text on failure,
*before* checking `response.ok` — an error response with a JSON body
(the common case) still gets its parsed object as `SarvaApiError.
body`; a non-JSON body (this bug's case) gets the raw string instead of
an opaque parse error, and either way `SarvaApiError` is always the
type actually thrown. Fixing this also surfaced a real gap in the
SDK's own test fixtures: the existing `fakeFetch` helper hand-built a
partial object exposing only `.json()`, which would have silently
masked this exact regression returning in the future — replaced with a
real `Response` instance (matching what real `fetch` actually returns),
built via `mockImplementation` rather than a shared `mockResolvedValue`
since a real response body can only be read once. Verified the new
test is real: reverted the fix and watched it fail with the literal
old bug reproducing itself, a raw `SyntaxError` instead of
`SarvaApiError`. 1 new test, 18 → 19 TypeScript SDK tests.

### `POST /config` itself froze the whole server under lock contention — the one blocking call in this file that never got wrapped in `asyncio.to_thread`

A much later fresh-eyes sweep, applying the same "does this blocking
call actually run off the event loop thread" lens the slow-Ollama-probe
fix (`docs/memory.md`'s memory chapter has the `note`/`search_notes`/
`remember`/`recall_memory` chain of this exact bug class) already
established for `build_router`/`build_providers`/`run_diagnostics`:
`save_config_route`'s own `save_config(non_empty)` call was the one
blocking call in this handler still called directly, with no
`asyncio.to_thread` — even though the very next line's comment
(`# asyncio.to_thread: see /models' own comment.`) sits two lines
below it, describing a discipline this one call never actually got.

`sarva.config.save_config()` acquires a real, cross-process
`fcntl.flock()`/`msvcrt.locking()` on `config.json.lock` and blocks
synchronously until it gets it — the exact "blocking cross-process lock
called directly from an `async def`" shape already found and fixed
for `SessionStore`'s own locking above. Confirmed live: a real second
OS process holding that lock for 3 seconds froze the event loop
solid — 0 of ~60 expected heartbeat ticks landed on a concurrent
coroutine during the whole window, meaning every other in-flight
request `sarva serve` was handling (another user's `/chat` or
`/ws/chat` stream) would have frozen too, not just the `/config`
request itself. Reachable with no adversarial input: the desktop
app's onboarding screen and `sarva config set` both write through
this exact lock, meant to be used interchangeably against the same
running server — any overlap between them (a user running `sarva
config set` in a terminal while the desktop app or a browser tab is
also mid-save) contends on it.

Fixed by wrapping the call as `await asyncio.to_thread(save_config,
non_empty)`, matching every sibling blocking call already in this same
handler and file. Verified live with the identical repro: the
concurrent heartbeat coroutine now ticks normally throughout the
3-second contended wait. Verified by reverting `server/app.py` alone
and racing a real `/health` request against a slow `POST /config` over
a genuine `httpx.AsyncClient`/ASGI transport (the same concurrency
methodology the slow-Ollama-probe test already uses): `/health` took
0.211s instead of its expected few milliseconds, reproducing the exact
old freeze. 1 new test, 752 → 753 Python tests.

### `/chat` and `/ws/chat` themselves never wrapped `SessionStore.load`/`.save` — the sibling gap the `POST /config` fix above never checked for

A much later fresh-eyes sweep, applying the round that had just found
`ReadFileTool`/`WriteFileTool`/`EditFileTool` blocking the event loop
in `agent/tools.py` (see the agent-loop chapter) as a lens one file
over: `SessionStore.load()`/`.save()` (a `read_text()` call, an
`atomic_write_bytes()` open/write/fsync/rename — see the memory
chapter) are real, synchronous filesystem I/O, called directly from
both `/chat` and `/ws/chat`'s own `async def` handlers with no
`asyncio.to_thread` — even though every *other* blocking call in these
same two handlers (`build_router`, `build_providers`,
`run_diagnostics`, `save_config`, even the session lock's own
`flock`/`msvcrt.locking` acquire inside `store.locked()` itself) was
already wrapped, including the `POST /config` fix directly above this
one in the same file.

Confirmed live with the same heartbeat-coroutine methodology used
throughout this project for this exact bug class: a simulated slow
disk froze the whole event loop for the duration of a single `store.
load()` call — near-zero ticks recorded instead of the expected count.
A concurrent-request race (the technique the `POST /config` fix's own
regression test uses) doesn't reliably catch this specific call site:
`store.locked(session)` does its own real `asyncio.to_thread` dispatch
for the session flock *before* `store.load()` ever runs, and that
genuine yield point lets an unrelated fast request sneak in and finish
regardless of whether the later `load()` call blocks — the heartbeat
technique, measured across the one request that actually exercises the
slow call, isn't fooled by that. Reachable with no adversarial input:
`store.load()` runs on every `/chat`/`/ws/chat` turn with a session
set, and `store.save()` on every one that completes successfully — on
a slow or network-mounted filesystem (this project's own `sarva.
config` docstring already names "shared dev servers, lab machines, CI
runners with persistent home directories" as real, not hypothetical),
this stalls every *other* concurrent user's in-flight turn too, not
just the one doing the session I/O.

Fixed identically to every sibling instance of this bug class: both
calls, in both handlers, now go through `asyncio.to_thread`. Verified
live: the same heartbeat repro now ticks throughout the call instead
of freezing solid. Verified by reverting `server/app.py` alone and
watching both new tests fail with the literal old bug's own shape —
near-zero heartbeat ticks for both the load and save call sites. 2 new
tests, 799 → 801 Python tests.

### `ws_chat`'s own cleanup could crash with a raw `RuntimeError` — a `finally` block assuming a precondition the exception it was cleaning up after had already invalidated

A much later fresh-eyes sweep found `ws_chat`'s `finally: await
websocket.close()` — the very last line of the handler, meant to be an
unconditional safety net — could itself raise. A **send-side**
disconnect (the client's TCP connection dropping while the handler is
mid-stream inside `websocket.send_text()`, well inside the `async for
event in loop.run(...)` loop — a phone losing signal, a laptop
sleeping, a proxy timeout; most of a streaming turn's wall-clock time
is spent sending, not waiting on a client reply, so this is the more
common disconnect shape, not a rare one) already transitions
Starlette's own `application_state` to `DISCONNECTED` and raises
`WebSocketDisconnect`, caught cleanly by the `except WebSocketDisconnect:
pass` a few lines up. But the unconditional `close()` in `finally` then
sends another `websocket.close` ASGI message, and Starlette's own
`send()` refuses that once `application_state` is already
`DISCONNECTED` — raising a raw `RuntimeError('Cannot call "send" once a
close message has been sent.')` that propagated straight out of the
whole ASGI call, unlike every other failure path on this endpoint,
which is deliberately routed through a clean frame or muted.

A **receive-side** disconnect (e.g. the client closing while
`ws_confirm` awaits a reply) doesn't hit this: `application_state` is
still `CONNECTED` when `WebSocketDisconnect` is raised from `receive()`,
so `close()` there succeeds normally — which is exactly why this
survived this file's own otherwise thorough, many-rounds-deep
error-handling hardening: every prior fix on this endpoint happened to
exercise the receive-side shape, never the send-side one. Confirmed
live, driving the real ASGI app directly (scope/receive/send, not
FastAPI's `TestClient` WebSocket session — see the new test's own
comment for why: a genuinely unhandled exception in the app leaves
`TestClient`'s background-thread reader blocked forever waiting for a
message that will never arrive, rather than failing fast) with a
`send_text` simulating exactly what Starlette's real `send()` does
internally on an `OSError`: mark the socket `DISCONNECTED`, then raise
`WebSocketDisconnect`. The subsequent `finally`-block `close()` then
raised the raw `RuntimeError` shown above.

Fixed by guarding the `finally` block: `if websocket.application_state
!= WebSocketState.DISCONNECTED: await websocket.close()` — a socket
that's already disconnected needs no further closing; one that's
genuinely still open still gets its own explicit close, unchanged.
Verified live the identical repro now completes with no exception.
Verified by reverting and watching the new test fail with the literal
old bug's own shape: the raw `RuntimeError` propagating out of the ASGI
call. 1 new test, 848 → 849 Python tests.

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

**A much later fresh-eyes sweep found the permissions fix above had
only ever covered the *file*, never the *directory* it lives in —**
the identical gap `SessionStore`/`LongTermMemoryStore`/
`VectorMemoryStore` (see the memory chapter) had already closed for
their own `~/.sarva/...` subdirectories, `os.chmod`ing them to `0o700`
right after `mkdir`, but this module — whose own docstring most
directly promises exactly this hardening, for the most sensitive data
in the whole `~/.sarva/` tree — never picked up the same fix for
`~/.sarva` itself. Confirmed live on a fresh install (the directory
doesn't exist yet): `save_config`'s own `mkdir` left `~/.sarva` at
`0o755` under the same common `022` umask this module's own docstring
already tested against for the file — world-readable, letting any
other local user on a shared machine list `~/.sarva`'s contents and
confirm a key is configured — and at `0o777` (genuinely
world-**writable**) under a more permissive real `000` umask, letting
another local user unlink or rename `config.json` out from under the
app regardless of the file's own `0600` bits. Fixed in `_exclusive_
lock`, the one place both `save_config` and `unset_config` already
route every write through, rather than duplicated at each call site —
self-healing the same way the file permission fix already is, tightening
a directory an older version of this module left insecure on the very
next save. Verified live a fresh, never-before-existing `~/.sarva`
directory now lands at `0o700`. Verified by reverting and watching both
new tests fail with the literal old bug's own value, `0o755` where
`0o700` was expected. 2 new tests, 771 → 773 Python tests.

`Onboarding.tsx` is the screen this makes possible: on mount it polls
`GET /doctor`; if any provider (including a reachable Ollama) is already
configured, it completes immediately and the user never sees it. If
not, it offers exactly the two documented choices — Ollama instructions
with a live "Check again" re-poll, or a key-paste form that `POST
/config`s and shows the fresh `/doctor` result — plus an honest "Skip
for now" escape hatch (remembered in `localStorage`) for anyone who just
wants the always-available Mock provider.

**`get_env()`'s own precedence rule — "a real environment variable
always wins" — silently broke for the ordinary case of clearing one.**
`get_env()`'s docstring states the precedence explicitly: a real
process environment variable if set, else whatever's saved in
`config.json`, else `None`. But the check was `if env_value:`, plain
truthiness, not `is not None` — the same "legitimate empty/zero value
collapsed into absent" shape already found and fixed twice in this
project's audio subsystem (`_decode_audio_isolated`'s stdout check,
`AudioToTextDegrader`'s duration check), reappearing here on a third
kind of value (an env var string) in a third, unrelated subsystem.
`os.environ.get(name)` already distinguishes "not set" (`None`) from
"set to empty string" (`""`) correctly; the truthiness check collapsed
both into the same branch. The ordinary shell idiom `ANTHROPIC_API_KEY=
sarva ...` — explicitly clearing an inherited or previously-saved key
for one invocation, not a contrived input — was indistinguishable here
from the variable never having been exported at all, so it silently
fell through to the stale saved config value instead, directly
contradicting the function's own stated precedence. **Confirmed live**:
a key saved earlier via `sarva config set` resurrected itself and got
used to construct a real, authenticated provider client even after the
env var was explicitly cleared for that run. **Fixed** with `if
env_value is not None:` — matching `os.environ.get`'s own real
`None`/`""` distinction instead of collapsing it. Verified every
`get_env()` call site in `sarva.runtime` (`build_providers`,
`run_diagnostics`) already treats an `""` return as falsy/not-configured,
so nothing downstream needed to change. Verified by reverting and
watching the new test fail with the literal old bug's own shape — the
stale `"sk-from-config"` value where `""` was expected. 1 new test,
851 → 852 Python tests.

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

**A completely ordinary, non-adversarial audio attachment could exhaust
host memory — a genuine DoS with no malicious or corrupted input
needed, distinct from the SIGBUS bug above.** `WhisperModel.transcribe()`
hands the fully-decoded audio array to `faster-whisper`'s own
`FeatureExtractor`, which computes the log-mel spectrogram for the
*entire* array in one call — a full-length STFT, materializing
overlapping-frame, FFT, magnitude, and mel arrays all at once — before
its own 30-second-window inference loop even begins. Confirmed live
with real measurements, not an estimate: a plain sine-tone WAV,
nothing crafted, drove peak RSS roughly linearly with duration — a
30-minute file (14.4 MB compressed as mp3) reached ~3.0 GB. A
completely ordinary 2-hour recording (a normal meeting/podcast/lecture
attachment) would extrapolate toward ~12 GB, easily OOM-killing a
typical 8–16 GB host, with nothing upstream bounding it — locally
path/data-sourced audio blocks bypass `fetch.py`'s own 20 MB URL-fetch
cap entirely, since that only applies to `url`-sourced blocks, and the
existing decode-timeout fix bounds wall-clock time, not memory, so it
does nothing to help here either. Since the actual blow-up happens
inside a vendored dependency this project doesn't own the internals
of, the fix is a duration cap at the `sarva.audio.transcribe()` call
site: `_MAX_TRANSCRIBE_SECONDS = 600` (10 minutes, chosen from the
measured ~100 MB/minute rate to keep worst-case memory around 1 GB),
checked right after decode (where duration is cheaply knowable from
the decoded sample count) and before the expensive feature-extraction
step ever runs. Raises the same `RuntimeError` shape every other
`transcribe()` failure already does, so `AudioToTextDegrader` needed no
changes at all — its existing broad `except Exception: pass` already
treats this identically to a decode failure, falling back to the clean
metadata-only report. **A real, adjacent gap found while making this
fix, not a new one introduced by it:** `sarva transcribe` (the CLI
command, the one place a real person runs this directly rather than
through the degrader) never caught `RuntimeError` at all — meaning a
decode failure, a decode timeout, *or* this new duration-cap error all
crashed with a raw traceback through that one specific command, a
pre-existing gap this fix closed at the same time by widening the
command's own `except` clause. Verified with a real synthetic WAV at
601 seconds (601s, wired via `wave` module directly, no ffmpeg
dependency) and a real `resource`-measured memory comparison: reverting
the fix made the same test spend over a minute genuinely transcribing
the whole file rather than rejecting it early, confirming the check
fires for the real reason, not a contrived one.

**A third exception type from the identical call, missed by the same
command, found by a much later fresh-eyes sweep.** `--model-size` is a
free-text string option with no validation at all — a plausible real
mistake (a typo like `large-v4`, misremembering "xlarge", a stale
copy-pasted model name), not a contrived attack. `WhisperModel
(model_size, ...)` (inside `_whisper_model`) raises a plain
`ValueError` for anything that isn't a recognized size shorthand or a
Hugging Face repo id, confirmed live with a real WAV file and a bad
size string. `AudioToTextDegrader` never hits this at all — it always
calls `transcribe()` with the function's own default `model_size`,
never a caller-supplied one — so this was reachable only through
`sarva transcribe` itself, the one command whose `except` clause the
`RuntimeError` fix right above had already widened once for this exact
same underlying call, and still didn't cover every exception type it
can raise. Fixed with one more `except ValueError` clause alongside the
existing two, same clean-message treatment. Verified by reverting and
watching the new test fail exactly as the original bug would have —
empty output where the clean error message belongs.

**The duration cap itself, above, was checked too late — a fresh-eyes
sweep found it ran *after* the exact expensive, unbounded step it was
supposed to prevent, not before.** `transcribe()` called `_decode_
audio_isolated(audio_bytes)` unconditionally first and only computed
`duration_s`/checked it against the cap afterward — meaning
`capture_output=True` had already read the *entire* decoded array back
from the worker subprocess into the long-lived parent process before
the "protective" cap ever got a chance to fire. Confirmed live: a
7.2 MB, 2-hour real audio file (ordinary — a long lecture/meeting/
voicemail recording with quiet stretches, not crafted) decoded to a
~460 MB float32 array that fully materialized in the parent, driving
its RSS from ~27 MB to ~1.2 GB before rejection; the relationship has
no ceiling — a similarly-encoded ~24-hour file, still well under
100 MB on disk, would have materialized roughly 5.5 GB, easily enough
to OOM-kill a `sarva serve` host serving multiple concurrent users.
Worse still, this wasn't even isolated to the throwaway subprocess the
way the SIGBUS crash fix above claims process isolation for — the
memory blow-up landed squarely in the process every other user's
in-flight turn shares. Fixed by moving the cap check *into* the
isolated worker itself (`sarva._audio_decode_worker`, passed the cap
as an argument): the worker now checks duration right after its own
decode and, if over the cap, writes a small, distinct stderr marker
(`AUDIO_TOO_LONG:<duration>`) and exits nonzero **without ever writing
the decoded array to stdout at all** — the parent's `_decode_audio_
isolated` recognizes that marker and raises the identical `RuntimeError`
message the caller already expected, but never has to receive the huge
payload to do it. Verified live with the same 2-hour repro: parent RSS
growth dropped from ~1.2 GB to a small, roughly constant overhead
unrelated to audio length at all (the same order of magnitude whether
the input is 5 seconds or 2 hours), and the worker's own real stdout is
now provably empty for a too-long input — proven directly at the
worker's own subprocess boundary in the new test, not just inferred
from `transcribe()`'s eventual exception. Verified by reverting and
watching the new test fail with the literal old bug's own shape: a
real, populated stdout instead of an empty one. 1 new test, 775 → 776
Python tests.

**A much later fresh-eyes sweep found `_decode_audio_isolated`'s own
success check had the identical "empty is silently treated as failed"
truthiness bug already fixed once elsewhere in this same audio
subsystem.** `if result.returncode != 0 or not result.stdout:` treats
a genuinely valid, zero-frame WAV — a plausible real artifact, not
contrived (a voice-message client where the user tapped-and-
immediately-released record) — as a decode FAILURE, even though the
worker exits `0` and writes `0` bytes to stdout exactly per its own
documented contract (zero samples is still a success, per this
module's own docstring above). The identical shape as
`AudioToTextDegrader`'s own already-fixed `duration_s or ...` bug — a
legitimately-zero real value that Python truthiness silently confuses
with "absent" or "failed" — just reintroduced independently, one file
over, on different data (stdout bytes instead of a duration float).
Confirmed live: a real, valid zero-frame WAV (a genuinely constructed
44-byte WAV header, zero frames) decoded successfully by the worker
(`returncode=0`, empty stderr) still raised `"could not decode audio"`
in the parent, misreporting a completely successful decode as a
failure. Fixed by dropping `or not result.stdout` entirely: the
worker's own `returncode` is already the correct, sufficient
success/failure signal per its own documented contract — stdout
emptiness on a `returncode == 0` success is meaningful data (zero
samples), never a second failure indicator to `or` in. Verified live
the same zero-frame WAV now transcribes successfully to an empty
string; a genuinely undecodable input (garbage bytes) still correctly
raises. Verified by reverting and watching the new test fail with the
literal old bug's own shape: `RuntimeError: could not decode audio`
for audio that decoded perfectly fine. 1 new test, 850 → 851 Python
tests.

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

**A much later fresh-eyes sweep found a third, structurally different
way for `synthesize()` to crash — before the child process is even
spawned, on ordinarily long text with nothing crafted about it.** The
macOS (`say`) and Linux (`espeak`/`espeak-ng`) branches both pass
`text` as a literal `argv` element to `subprocess.run`, with no size
cap anywhere — unlike the Windows branch (below), which already writes
`text` to a temp file and reads it back via PowerShell's `Get-Content`
specifically to avoid interpolating arbitrary text into a command at
all. If `text` is large enough to exceed the OS's real `execve()`
argument-list limit (`ARG_MAX` — confirmed on this machine via `getconf
ARG_MAX`, ~1MB; Linux additionally caps any *single* `argv` string at
128KB via `MAX_ARG_STRLEN`, independent of the total), `Popen.__init__`
raises `OSError: [Errno 7] Argument list too long` — neither a
`TimeoutExpired` nor a `CalledProcessError`, the only two types this
function already caught, so it propagated raw straight past `sarva
speak`'s own `except RuntimeError` and crashed with a full Python
traceback. Confirmed live: `sarva speak "$(cat transcript.txt)"` on an
ordinarily long piece of text — a book chapter, an article, a meeting
transcript, `speak`'s own advertised use case — hits this once the
text approaches the real OS limit, no crafted or adversarial input
needed. This module's own docstring already names "an agent speaking
its own output" as the real threat model `text` can be arbitrary,
potentially large content for — the same reasoning that already
motivated the timeout fix directly above. Fixed with one more `except
OSError` clause in `synthesize()`, translating it into the same clean
`RuntimeError` shape every other engine failure already gets. Verified
live the same over-long-text scenario now fails cleanly instead of
crashing. Verified the new tests are real: reverted the fix and
watched both the library-level and CLI-level tests fail with the raw,
uncaught `OSError` propagating before re-applying. 2 new tests,
776 → 778 Python tests.

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

**`_whisper_model`'s own `lru_cache`, named above as the fix for
redundant reloads, turned out to have an unsynchronized check-then-act
race — the identical shape already found and fixed twice elsewhere in
this project (`_probe_ollama`, round 128; `FoundryProvider`'s
`load_checkpoint_bundle`, round 130).** A much later fresh-eyes sweep
found that `functools.lru_cache` only holds its own internal lock
around the cache dict's read/insert bookkeeping — it releases that lock
*before* calling the wrapped function itself, so two threads racing a
cold cache both see a miss and both construct a real, expensive
`WhisperModel` (a weight download on first use, then real CTranslate2
model initialization). `AudioToTextDegrader.degrade()` calls
`transcribe()` via `asyncio.to_thread`, so two concurrent voice-message
transcriptions — two users, or two tabs of the same user, against a
freshly-started `sarva serve` process, both defaulting to
`model_size="tiny"` — genuinely race this on real OS threads; nothing
adversarial required. Confirmed live: 8 threads synchronized via a
`threading.Barrier` to hit a cold cache at the same instant produced 8
real `WhisperModel` constructions, not the 1 this cache exists to
guarantee — defeating the exact "reloading it every call would be a
real performance regression" guarantee this module's own docstring
names. Fixed with a manual `dict` + `threading.Lock` around the whole
check-construct-store span (the same shape as the round 128/130
fixes), preserving the original `maxsize=4` LRU-eviction behavior via
an `OrderedDict` rather than switching to an unbounded cache. Verified
live the identical 8-concurrent-callers repro now measures exactly 1
real construction. Verified by reverting and confirming the original
`lru_cache`-based code reproduces the same race shape (8 real
constructions) once the fake model's own init is given a realistic
amount of latency to widen the race window. 1 new test, 825 → 826
Python tests.

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
