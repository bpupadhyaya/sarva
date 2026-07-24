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
- **`run TASK [--workdir .] [--image PATH] [--model ID] [--auto] [--session NAME] [--mcp-server CMD]... [--mcp-header "Name: Value"]...`**
  — the full agent loop with `BUILTIN_TOOLS` (files, shell) plus any MCP
  servers. `--mcp-server` is repeatable; each value is shell-split and
  connected via `connect_stdio_mcp_server` inside an `AsyncExitStack`
  (see the MCP chapter), its tools appended to the built-in list. Without
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
- **`sessions list`** / **`sessions clear NAME`** — inspect or delete
  persisted chat sessions.
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
