"""sarva.server.app — the FastAPI server: REST + WebSocket over the agent loop.

Zero-config by default, same as the CLI: with no ANTHROPIC_API_KEY set,
everything routes to the offline MockProvider. `/chat` is single-turn,
non-streaming, and tool-free (mirrors `sarva chat` exactly — a REST
request can't naturally pause mid-request for a confirmation round-trip).
`/ws/chat` is the tool-using surface (mirrors `sarva run`): it streams the
same AgentEvents the CLI renders, and when a destructive tool needs
confirmation, it sends a `needs_confirmation` frame and *waits* for the
client's `{"approved": bool}` reply before continuing — the bidirectional
protocol REST can't offer is exactly why tools live here, not on `/chat`.

If a built web UI is present at `sarva/server/static/` (the React app in
`apps/desktop/`, built via `npm run build` and copied in — see
BUILD-JOURNAL.md for why this is currently a manual step rather than a CI
one), it's mounted at `/` so `sarva serve` alone gives a complete browser
experience. Without it, this is API-only — nothing breaks either way.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sarva.agent.budget import Spend
from sarva.agent.events import AgentState, RunDoneEvent, StateChangedEvent
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import BUILTIN_TOOLS, always_allow
from sarva.config import ConfigError, save_config
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ContentBlock, ImageBlock, Message, ToolCallBlock
from sarva.multimodal.degraders import default_degraders
from sarva.runtime import build_providers, build_router, run_diagnostics
from sarva.server.schemas import (
    ChatRequest,
    ChatResponse,
    DoctorCheckOut,
    ModelInfoOut,
    SaveConfigRequest,
)

_STATIC_DIR = Path(__file__).parent / "static"

# A real, human-facing prompt -- someone actually has to read it and
# click a button -- so this is deliberately much longer than
# AgentLoop's own 90-second per-tool-execution backstop
# (core/sarva/agent/loop.py's _TOOL_TIMEOUT_SECONDS), which bounds an
# automated tool call, not a person.
_CONFIRM_TIMEOUT_SECONDS = 300

# A real bug found by actually racing two concurrent turns on the same
# session with asyncio.gather (mirroring /chat and /ws/chat's own
# concurrency model exactly -- no threads needed, since a single-
# process asyncio event loop already interleaves two in-flight requests
# across the real, multi-second `await`s an agent turn makes for model
# calls): SessionStore.load() at the start of a turn and .save() at the
# end are a classic unlocked read-modify-write pair, the identical bug
# class already fixed for config.json's concurrent-write race
# (`_exclusive_lock` in sarva.config) -- just never applied to session
# storage. Confirmed live: two ordinary turns on the SAME session name
# (e.g. two browser tabs both chatting into the default session)
# produced a session file holding only ONE of the two turns' new
# messages, the other silently discarded with no error to either
# client. Both handlers wrap their whole load-through-save span in
# `store.locked(session_id)` -- a real, cross-process exclusive lock
# (see SessionStore.locked's own docstring), not just an in-process one:
# a prior version of this fix used a plain in-process `asyncio.Lock`
# here, which closed the same-process case (two connections to the same
# running `sarva serve`) but left an identically-shaped cross-process
# race open (the CLI and a running server, or two separate CLI
# invocations, writing the same session file) -- confirmed live with
# two genuine OS processes before that gap was closed too. A single
# cross-process lock now covers both cases at once, since POSIX
# `flock`/Windows `msvcrt.locking` correctly serialize same-process
# callers against each other exactly as well as different-process ones.


def _is_same_origin(origin: str | None, host: str | None) -> bool:
    """True iff a browser-sent `Origin` header names the same host:port
    this request's own `Host` header does -- the standard "reject
    cross-site WebSocket hijacking" check every browser-facing WS
    endpoint needs, since (unlike `fetch()`) the WebSocket handshake is
    NOT subject to the Same-Origin Policy or CORS at all: any webpage a
    user's browser has open can open `new WebSocket("ws://127.0.0.1:
    8000/ws/chat")` and the browser will send it regardless of which
    site's tab it came from. A real bug found by actually connecting a
    TestClient WebSocket session with `Origin: https://evil.example.com`
    set: the handshake was accepted and a full run completed -- and
    `ws_chat` builds an `AgentLoop` with `BUILTIN_TOOLS` and, if the
    client also sends `"auto": true`, `confirm=always_allow`, meaning a
    malicious webpage could drive real shell/file-tool access with
    every destructive-tool confirmation gate auto-approved, no user
    interaction required, purely by being open in a background tab
    while `sarva serve`/the desktop app is running locally -- a classic
    cross-site WebSocket hijacking (CSWSH) vector.

    `origin` being absent is NOT rejected: real browsers always send
    `Origin` on a WebSocket handshake (mandated by RFC 6455 for browser
    clients specifically), so its absence means a non-browser caller
    (a script, a future first-party CLI client) that this specific
    threat -- an ordinary webpage silently driving the connection --
    structurally can't produce; blocking those callers too would be a
    real regression with no security benefit, since the drive-by risk
    this closes is inherent to *browser* WebSocket behavior."""
    if origin is None:
        return True
    return urlsplit(origin).netloc == (host or "")


def _extra_content_blocks(
    image_base64: str | None, image_media_type: str | None
) -> list[ContentBlock]:
    """Shared by /chat (a validated ChatRequest) and /ws/chat (a raw JSON
    frame with no schema of its own) so the two request paths can't drift
    apart on what "an attached image" means."""
    if image_base64 and image_media_type:
        return [ImageBlock(media_type=image_media_type, data=base64.b64decode(image_base64))]
    return []


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sarva",
        description="An open, all-in-one multimodal AGI tool.",
    )

    @app.exception_handler(ConfigError)
    async def config_error_handler(request: Request, exc: ConfigError) -> JSONResponse:
        # A real bug found by actually corrupting ~/.sarva/config.json and
        # hitting /models, /doctor, or POST /config: get_env() backs
        # nearly every provider-availability check, so a bad file crashed
        # every one of them with a raw json.JSONDecodeError 500 -- not
        # even the clean-but-still-500 shape a normal unhandled exception
        # gets in production, a genuine unformatted traceback. One handler
        # here covers every plain HTTP route in this app; /chat and
        # /ws/chat get their own explicit handling below since their
        # response shapes (ChatResponse, WS event frames) aren't generic
        # JSON errors and a WebSocket route isn't covered by this handler
        # at all.
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", response_model=list[ModelInfoOut])
    async def models() -> list[ModelInfoOut]:
        # asyncio.to_thread: build_router() ultimately makes a real,
        # synchronous httpx.get() network call to probe Ollama
        # (runtime._probe_ollama) -- see that function's own comment on
        # the bug this closes. Called inline, that blocks this whole
        # process's event loop for the full network round-trip (up to
        # its 0.3s timeout, or longer against a black-holed/slow-to-
        # refuse address), stalling every other concurrent request this
        # server is handling, not just this one.
        router = await asyncio.to_thread(build_router)
        return [
            ModelInfoOut(id=m.id, display_name=m.display_name, available=m.id in router.available)
            for m in router.registry.all()
        ]

    @app.get("/doctor", response_model=list[DoctorCheckOut])
    async def doctor() -> list[DoctorCheckOut]:
        """The same diagnostics `sarva doctor` prints, as JSON — what the
        desktop app's first-run screen polls to decide whether anything is
        configured yet, reusing `run_diagnostics()` exactly so this can
        never drift out of sync with what the CLI reports."""
        # asyncio.to_thread: see /models' own comment -- run_diagnostics()
        # calls build_router()/build_providers() internally, so it makes
        # the identical blocking Ollama-probe network call.
        checks = await asyncio.to_thread(run_diagnostics)
        return [DoctorCheckOut(name=c.name, ok=c.ok, detail=c.detail) for c in checks]

    @app.post("/config", response_model=list[DoctorCheckOut])
    async def save_config_route(req: SaveConfigRequest) -> list[DoctorCheckOut]:
        """Persists whichever provider keys the caller supplied to
        `sarva.config`'s file — the desktop first-run screen's "paste an
        API key" path writes here. Returns the fresh `/doctor` result
        (not just {"ok": true}) so the caller can confirm the key it just
        saved is actually recognized, in one round trip."""
        values = {
            "ANTHROPIC_API_KEY": req.anthropic_api_key,
            "OPENAI_API_KEY": req.openai_api_key,
            "GEMINI_API_KEY": req.gemini_api_key,
            "GOOGLE_API_KEY": req.google_api_key,
        }
        non_empty = {k: v for k, v in values.items() if v}
        if non_empty:
            # asyncio.to_thread: see /models' own comment -- a real,
            # sibling-propagation gap found by a fresh-eyes sweep.
            # save_config() acquires a real, cross-process fcntl.flock()/
            # msvcrt.locking() on config.json.lock and blocks
            # synchronously until it gets it, exactly the "blocking
            # cross-process lock called directly from an async def" shape
            # every OTHER blocking call in this file is already wrapped
            # against -- this one call site never got the same fix.
            # Confirmed live: a second process holding that lock for 3s
            # froze the whole event loop solid (0 of ~60 expected
            # heartbeat ticks landed) -- every other concurrent request
            # (another user's in-flight /chat or /ws/chat stream) would
            # have frozen too. Reachable with no adversarial input: the
            # desktop app's onboarding screen and `sarva config set`
            # both write through this exact lock, and both are meant to
            # be used interchangeably against the same running server.
            await asyncio.to_thread(save_config, non_empty)
        # asyncio.to_thread: see /models' own comment.
        checks = await asyncio.to_thread(run_diagnostics)
        return [DoctorCheckOut(name=c.name, ok=c.ok, detail=c.detail) for c in checks]

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        store = SessionStore()
        # A real bug found by a fresh-eyes sweep: `req.session` is a
        # caller-controlled `str | None` with no non-empty constraint
        # (ChatRequest's own schema), and every OTHER use of it below
        # already treats "" the same as no session at all (`store.load
        # (req.session) if req.session else []`, `if req.session and
        # state == DONE`) -- but `store.locked(req.session)` checks `name
        # is None` specifically, so "" reached `SessionStore._sanitize()`
        # and raised `ValueError: invalid session name: ''`, before the
        # agent ever ran. Confirmed live: an otherwise-identical request
        # succeeded with no `session` field, succeeded with
        # `session="default"`, and failed outright with `session=""` --
        # a very ordinary client pattern (a form/state field initialized
        # to "" and always serialized, rather than omitted or sent as
        # `null`) hits this on literally every chat request. Normalized
        # once, here, so every use below -- `locked()` included -- agrees
        # on what "no session" means.
        session = req.session or None
        try:
            # The whole load-through-save span of the turn is inside the
            # session lock -- see SessionStore.locked's own docstring for
            # why locking only the final save() wouldn't actually close
            # the race. store.locked() itself validates the session name
            # (the same SessionStore._sanitize() call `store.load` below
            # already relied on), so this outer except ValueError is what
            # now catches an invalid name -- entering the `async with`
            # raises before the inner try block below even starts.
            async with store.locked(session):
                # asyncio.to_thread: a real bug found by a fresh-eyes
                # sweep, the identical bug class already fixed at least
                # five separate times in this project (four times in
                # agent/tools.py's memory tools, once more for that same
                # file's ReadFileTool/WriteFileTool/EditFileTool) --
                # SessionStore.load()/.save() do real, synchronous
                # filesystem I/O (a read_text() call, an atomic_write_
                # bytes() open/write/fsync/rename), called directly on
                # the event loop despite every OTHER blocking call in
                # this same handler (build_router, build_providers,
                # run_diagnostics, save_config, even the session lock's
                # own flock/msvcrt.locking acquire inside store.locked()
                # itself) already being wrapped. Confirmed live: a
                # simulated slow disk froze the whole process for the
                # entire call, zero heartbeat ticks recorded -- on a
                # real busy/network-mounted filesystem this stalls every
                # OTHER concurrent user's in-flight /chat or /ws/chat
                # turn too, not just the one doing the session I/O.
                history = await asyncio.to_thread(store.load, session) if session else []
                # A real bug found by actually sending {"image_base64":
                # "not-valid-base64!!!", ...}: base64.b64decode() raises
                # binascii.Error (a ValueError subclass) for malformed
                # input, and nothing here caught it -- a genuine unhandled
                # 500, the same "raw traceback instead of a clean message"
                # bug class already fixed for an invalid --session name.
                # Sharing this except block means both failure modes get
                # the identical clean ChatResponse(state=failed, detail=...)
                # treatment.
                extra_content = _extra_content_blocks(req.image_base64, req.image_media_type)

                try:
                    # asyncio.to_thread: see /models' own comment -- both
                    # calls ultimately make a blocking Ollama-probe network
                    # call that would otherwise freeze this whole process's
                    # event loop, stalling every other concurrent request.
                    router = await asyncio.to_thread(build_router)
                    providers = await asyncio.to_thread(build_providers)
                    loop = AgentLoop(
                        router=router,
                        providers=providers,
                        tools=[],
                        confirm=always_allow,
                        degraders=default_degraders(),
                        verify=req.verify,
                    )
                except ConfigError as e:
                    # The /chat-specific counterpart to the global
                    # ConfigError handler above: this endpoint's own
                    # established shape for "this request can't run" is a
                    # real ChatResponse with state=failed and the reason
                    # in `detail`, not the generic {"detail": ...} 500 the
                    # global handler returns elsewhere -- kept consistent
                    # with the unknown-model and invalid-session failure
                    # modes above, which callers already parse the same
                    # way.
                    return ChatResponse(
                        state=AgentState.FAILED, message=None, spend=Spend(), detail=str(e)
                    )

                state = AgentState.FAILED
                final_message: Message | None = None
                spend = Spend()
                last_detail: str | None = None
                transcript: list[Message] = []
                async for event in loop.run(
                    req.message,
                    history=history,
                    model_override=req.model,
                    extra_content=extra_content,
                    transcript_out=transcript,
                    session_id=session,
                ):
                    if event.type == "state_changed" and event.detail:
                        last_detail = event.detail
                    if event.type == "run_done":
                        state = event.state
                        final_message = event.final_message
                        spend = event.spend

                if session and state == AgentState.DONE:
                    # The save half of the same asyncio.to_thread fix as
                    # the load above.
                    await asyncio.to_thread(store.save, session, transcript)

                return ChatResponse(
                    state=state,
                    message=final_message.text() if final_message else None,
                    spend=spend,
                    detail=last_detail if state != AgentState.DONE else None,
                )
        except ValueError as e:
            # A real bug found by actually sending {"session": "bad
            # name!"}: SessionStore._sanitize() raises a plain ValueError
            # for any name outside [A-Za-z0-9_-], and nothing here caught
            # it -- a genuine unhandled 500, not a clean failure, the
            # exact "raw traceback instead of a clean message" bug class
            # already fixed for eval/distill's --model. Reported the same
            # way an unknown --model already is: a real ChatResponse with
            # state=failed and the actual reason in `detail`, not a
            # differently-shaped HTTP error for what's semantically the
            # same "this request can't run" case.
            return ChatResponse(state=AgentState.FAILED, message=None, spend=Spend(), detail=str(e))

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        """Single-turn per connection, tools enabled (mirrors `sarva run`,
        not `sarva chat`). The client sends one
        {"message": ..., "session": ..., "auto": false} frame and receives
        streamed AgentEvent JSON frames ending with a run_done frame.
        Optional "image_base64"/"image_media_type" attach one image to the
        turn, the same shape and meaning as /chat's own fields -- this is
        the desktop app's only chat surface (it never calls /chat), so
        until this existed there was genuinely no way to send an image
        through the web UI at all despite the CLI and REST endpoint both
        already supporting it. Optional "model" forces a specific model
        id (same meaning as the CLI's own --model), bypassing the
        router's default selection entirely -- an unknown id surfaces as
        a real, visible `state_changed` frame with a `detail` message
        naming it, then a `run_done` frame with state "failed", never a
        silent fallback to a different model (see UnknownModelError in
        sarva.providers.registry for why that distinction needed its own
        exception type).

        When a destructive tool needs approval, the loop's `NeedsConfirmationEvent`
        arrives as usual, and the *next* value the client sends on this same
        socket — {"approved": bool} — is consumed as the answer before
        anything else continues. This works because AgentLoop.run() is a
        generator: it only calls the confirm policy after the event that
        announces the need for one has already been yielded (and therefore
        already sent to the client) — see sarva.agent.loop for the
        sequencing this depends on.

        Set "auto": true to skip confirmation prompts, equivalent to
        `sarva run --auto`. Subtlety worth knowing if you're writing a
        client: `needs_confirmation` is emitted by the loop whenever a
        destructive call happens at all — it is NOT suppressed by "auto".
        What changes is that the confirm *policy* becomes `always_allow`,
        which resolves immediately without reading from the socket. A
        client in auto mode must therefore treat `needs_confirmation` as
        purely informational and must NOT send an {"approved": ...} reply
        for it — there's nothing waiting to consume one, and doing so risks
        a stray reply being read as the answer to a later, real prompt.
        """

        async def _send_failure(detail: str) -> None:
            # Shared by every clean-failure path below (a malformed
            # frame, an invalid session name, a non-string field, a
            # corrupted config) -- the exact same state_changed + run_done
            # pair an unknown --model already produces, so App.tsx's
            # existing failure-detail handling shows it with no
            # client-side changes needed.
            await websocket.send_text(
                StateChangedEvent(state=AgentState.FAILED, detail=detail).model_dump_json()
            )
            await websocket.send_text(
                RunDoneEvent(
                    state=AgentState.FAILED, final_message=None, spend=Spend()
                ).model_dump_json()
            )

        if not _is_same_origin(websocket.headers.get("origin"), websocket.headers.get("host")):
            # Rejected before accept() -- an ordinary WebSocket close
            # handshake, not a "clean failure" JSON frame pair, since a
            # cross-origin caller is exactly the audience this check
            # exists to give nothing useful to.
            await websocket.close(code=1008)  # 1008: Policy Violation
            return
        await websocket.accept()
        try:
            try:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict):
                    # Valid JSON, but not an object -- e.g. a bare list or
                    # string -- makes every payload.get(...) call below
                    # raise AttributeError. Folded into the same ValueError
                    # handling as a malformed-JSON frame rather than left
                    # to surface as a different, uncaught exception type.
                    raise TypeError(f"expected a JSON object, got {type(payload).__name__}")
            except ValueError as e:
                # A real bug found by actually sending a non-JSON first
                # frame: Starlette's receive_json() does a bare
                # json.loads() with no error handling of its own --
                # confirmed directly with a real TestClient WebSocket
                # session, a malformed frame raised an uncaught
                # json.JSONDecodeError (a ValueError subclass) here,
                # crashing the whole ASGI call with no frame ever sent at
                # all -- the client just saw a bare ClosedResourceError,
                # the identical "the client gets nothing explaining why"
                # failure mode already fixed below for an invalid session
                # name, a malformed image_base64, and a corrupted
                # config.json.
                await _send_failure(f"malformed request: {e}")
                return
            except TypeError as e:
                await _send_failure(str(e))
                return

            message = payload.get("message", "")
            # The WS counterpart to the identical real bug just fixed for
            # /chat: raw, schema-less JSON means nothing stops a caller
            # from sending `"session": ""` -- normalized here, the same
            # way, so `store.locked(session)`'s `name is None` check and
            # every truthy `if session` check below agree on what "no
            # session" means, instead of "" reaching SessionStore.
            # _sanitize() and raising before the agent ever ran.
            session = payload.get("session") or None
            # A real bug found by a fresh-eyes sweep, the identical shape
            # already found and fixed a few lines below for `approved`
            # (see ws_confirm's own comment), just never propagated to
            # these two sibling caller-supplied JSON values in the same
            # handler: `bool(...)` is a Python-truthiness cast, not the
            # strict boolean check a flag this consequential needs --
            # `bool("false")` is `True`, since any non-empty string is
            # truthy. `auto` selects `always_allow` (zero user
            # confirmation) over the real `ws_confirm` gate for every
            # destructive tool call in the whole session -- confirmed
            # live: a client sending `"auto": "false"` (a JSON STRING,
            # exactly what a form/select's `.value`, or any client
            # serializing booleans as strings, produces) got EVERY
            # destructive tool call auto-approved with zero confirmation
            # prompts, the file written with no reply ever sent by the
            # client -- the opposite of what a caller sending a falsy-
            # looking value would reasonably expect, and materially more
            # severe than the `approved` bug (that one could misapprove
            # one already-surfaced prompt; this one disables the
            # confirmation system for the entire session). `verify` has
            # the identical gap, lower stakes but the same fix.
            auto = payload.get("auto") is True
            model = payload.get("model")
            verify = payload.get("verify") is True

            async def ws_confirm(call: ToolCallBlock) -> bool:
                # A real bug found by actually sending a malformed
                # confirmation reply (a JSON array instead of
                # {"approved": bool}), and by actually never replying
                # at all: `reply.get(...)` on a non-dict raised an
                # uncaught AttributeError that crashed the whole ASGI
                # call with no run_done frame sent (the same bare
                # ClosedResourceError this file's other raw-JSON fixes
                # exist to prevent), and receive_json() had no timeout
                # at all, so a client that never replies -- deliberately
                # or by just closing its own tab without the browser
                # noticing -- hung the connection forever with no
                # recovery, the confirmation-layer counterpart to the
                # already-fixed hung-tool-call bug (a different call
                # site entirely: this is AgentLoop's own confirm
                # callback, not `run_one`'s tool.run() wrapper). Both
                # failure modes are treated as a plain decline, not a
                # crash or a hang: a destructive action given an
                # ambiguous or absent confirmation signal must never
                # default to running, the same "reject, don't guess"
                # discipline this file already applies to malformed
                # --mcp-header/--mcp-env-style input elsewhere in this
                # project.
                try:
                    reply = await asyncio.wait_for(
                        websocket.receive_json(), timeout=_CONFIRM_TIMEOUT_SECONDS
                    )
                except (TimeoutError, ValueError):
                    return False
                if not isinstance(reply, dict):
                    return False
                # A real bug found by a fresh-eyes sweep: `bool(...)` is a
                # Python-truthiness cast, not the strict boolean check the
                # documented `{"approved": bool}` wire contract (this
                # function's own docstring above) actually promises --
                # `bool("false")` is `True`, since any non-empty string is
                # truthy in Python. Confirmed live: a client that sends
                # `{"approved": "false"}` (a JSON string, not a JSON
                # boolean -- exactly what a form/select's `.value`, or any
                # client serializing booleans as strings, produces with no
                # explicit `=== "true"` conversion) got a destructive tool
                # call APPROVED, the opposite of the sender's intent, with
                # no error and no signal anything was wrong. This directly
                # contradicts the "ambiguous or absent confirmation signal
                # must never default to running" discipline the comment
                # above already states for the timeout/non-dict cases --
                # a non-boolean truthy value isn't ambiguous by accident,
                # it was silently ACCEPTED as approval. `is True` is a
                # strict identity check: only the real JSON boolean `true`
                # approves: any other value (a string, a number, missing
                # entirely) fails closed the same way the two checks above
                # already do.
                return reply.get("approved") is True

            store = SessionStore()
            try:
                # See SessionStore.locked's own docstring: the whole load-
                # through-save span of the turn is inside the session lock,
                # the WS counterpart to the identical wrapping /chat just
                # got. store.locked() itself validates the session name,
                # so entering the `async with` below raises before the
                # inner try block even starts for an invalid name -- this
                # outer except (ValueError, TypeError) is what now catches
                # that case (session=123 raises TypeError the same way
                # SessionStore._sanitize()'s regex match already did).
                async with store.locked(session):
                    try:
                        if model is not None and not isinstance(model, str):
                            # A real bug found by actually sending {"model":
                            # ["a", "b"]}: AgentLoop.run()'s
                            # router.pick(override=model) call, several
                            # frames deep in the async generator below, does
                            # a dict lookup keyed on this value -- a
                            # non-string JSON type raised an uncaught
                            # TypeError ("unhashable type: 'list'") well
                            # after streaming had already started. /chat
                            # never sees this because Pydantic validates
                            # ChatRequest.model's type before the handler
                            # runs; /ws/chat parses raw, schema-less JSON,
                            # so nothing validated this until now.
                            raise TypeError(f"model must be a string, got {type(model).__name__}")
                        # asyncio.to_thread: the WS counterpart to the
                        # identical fix just applied to /chat -- see that
                        # call site's own comment for the full history of
                        # this bug class.
                        history = await asyncio.to_thread(store.load, session) if session else []
                        # The WS counterpart to the same real bug just fixed
                        # for /chat: a malformed "image_base64" made
                        # base64.b64decode() raise binascii.Error (a
                        # ValueError subclass) with nothing here to catch it
                        # -- the whole ASGI call crashed with no frame sent
                        # at all, the client saw a bare ClosedResourceError,
                        # confirmed directly with a real TestClient
                        # WebSocket session before this fix. Sharing this
                        # try block gives it the identical clean failure
                        # treatment the invalid-session-name case below
                        # already has.
                        extra_content = _extra_content_blocks(
                            payload.get("image_base64"), payload.get("image_media_type")
                        )
                    except (ValueError, TypeError) as e:
                        await _send_failure(str(e))
                        return

                    try:
                        # asyncio.to_thread: see /models' own comment.
                        router = await asyncio.to_thread(build_router)
                        providers = await asyncio.to_thread(build_providers)
                        loop = AgentLoop(
                            router=router,
                            providers=providers,
                            tools=BUILTIN_TOOLS,
                            confirm=always_allow if auto else ws_confirm,
                            degraders=default_degraders(),
                            verify=verify,
                        )
                    except ConfigError as e:
                        # The WS counterpart to the same real bug just fixed
                        # for /chat: a corrupted ~/.sarva/config.json raised
                        # a raw json.JSONDecodeError with nothing here to
                        # catch it -- the whole ASGI call crashed with no
                        # frame sent at all, the client seeing a bare
                        # ClosedResourceError, the exact same failure mode
                        # already fixed here for an invalid session name and
                        # a malformed image_base64 field.
                        await _send_failure(str(e))
                        return
                    state = AgentState.FAILED
                    transcript: list[Message] = []
                    async for event in loop.run(
                        message,
                        history=history,
                        model_override=model,
                        extra_content=extra_content,
                        transcript_out=transcript,
                        session_id=session,
                    ):
                        await websocket.send_text(event.model_dump_json())
                        if event.type == "run_done":
                            state = event.state

                    if session and state == AgentState.DONE:
                        # The save half of the same fix as the load above.
                        await asyncio.to_thread(store.save, session, transcript)
            except (ValueError, TypeError) as e:
                # SessionStore._sanitize() raises a plain ValueError for an
                # invalid session name (or TypeError for a non-string one),
                # and reaching this point uncaught didn't even give the
                # client the REST endpoint's own clean detail message -- it
                # crashed the whole ASGI call with no frame sent at all, and
                # the client saw a bare ClosedResourceError, confirmed
                # directly with a real TestClient WebSocket session before
                # this fix. Reported as a real state_changed + run_done
                # pair -- the exact same shape an unknown --model already
                # produces -- so App.tsx's existing failure-detail handling
                # (see BUILD-JOURNAL.md) shows it with no client-side
                # changes needed.
                await _send_failure(str(e))
        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()

    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="web-ui")

    return app
