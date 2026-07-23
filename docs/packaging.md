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
`MockProvider` so `sarva chat "hello"` always works." Seven commands,
each doing one thing:

- **`chat MESSAGE [--image PATH] [--session NAME]`** — one-shot,
  tool-free, single-turn (`AgentLoop(tools=[], confirm=always_allow)`).
  The simplest possible entry point, deliberately with no tool access.
- **`run TASK [--workdir .] [--auto] [--session NAME] [--mcp-server CMD]...`**
  — the full agent loop with `BUILTIN_TOOLS` (files, shell) plus any MCP
  servers. `--mcp-server` is repeatable; each value is shell-split and
  connected via `connect_stdio_mcp_server` inside an `AsyncExitStack`
  (see the MCP chapter), its tools appended to the built-in list. Without
  `--auto`, every destructive tool call stops for a real
  `typer.confirm(f"Allow {call.name}({call.arguments})?")` prompt;
  `--auto` swaps that for `always_allow`.
- **`models`** — lists every registry entry with `[x]`/`[ ]` marking
  whether it's currently available (API key present, Ollama reachable,
  a foundry checkpoint discovered — see the providers and foundry
  chapters).
- **`eval [--model ID]`** / **`distill PROMPTS --model ID --out PATH`**
  — the eval harness and distillation pipeline, covered in their own
  chapters.
- **`sessions list`** / **`sessions clear NAME`** — inspect or delete
  persisted chat sessions.
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

**Serving the web UI is genuinely optional, not a hard dependency of
the API:** if `core/sarva/server/static/` exists, it's mounted at `/`
via `StaticFiles(..., html=True)` so `sarva serve` alone gives a
complete browser experience; if it doesn't exist, the server is simply
API-only, with nothing breaking either way.

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
process that call alone can't reach, so a `#[cfg(unix)]`-gated
`pgrep -P <pid>` + `kill -9` step reaps it specifically on macOS/Linux.
**Windows genuinely doesn't have this yet** — a plain window-close still
cleans up via Tauri's `CloseRequested` handler, but there's no
equivalent to the Unix SIGINT/SIGTERM handler that also triggers the
grandchild reap, a real, open, documented gap rather than a silently
assumed non-issue.

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
earlier chapters.
