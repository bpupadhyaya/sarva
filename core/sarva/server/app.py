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

import base64
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from sarva.agent.budget import Spend
from sarva.agent.events import AgentState, RunDoneEvent, StateChangedEvent
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import BUILTIN_TOOLS, always_allow
from sarva.config import save_config
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

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", response_model=list[ModelInfoOut])
    async def models() -> list[ModelInfoOut]:
        router = build_router()
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
        return [DoctorCheckOut(name=c.name, ok=c.ok, detail=c.detail) for c in run_diagnostics()]

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
        }
        non_empty = {k: v for k, v in values.items() if v}
        if non_empty:
            save_config(non_empty)
        return [DoctorCheckOut(name=c.name, ok=c.ok, detail=c.detail) for c in run_diagnostics()]

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        store = SessionStore()
        try:
            history = store.load(req.session) if req.session else []
            # A real bug found by actually sending {"image_base64":
            # "not-valid-base64!!!", ...}: base64.b64decode() raises
            # binascii.Error (a ValueError subclass) for malformed input,
            # and nothing here caught it -- a genuine unhandled 500, the
            # same "raw traceback instead of a clean message" bug class
            # already fixed for an invalid --session name a few lines
            # above. Sharing this try block means both failure modes get
            # the identical clean ChatResponse(state=failed, detail=...)
            # treatment.
            extra_content = _extra_content_blocks(req.image_base64, req.image_media_type)
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

        loop = AgentLoop(
            router=build_router(),
            providers=build_providers(),
            tools=[],
            confirm=always_allow,
            degraders=default_degraders(),
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
            session_id=req.session,
        ):
            if event.type == "state_changed" and event.detail:
                last_detail = event.detail
            if event.type == "run_done":
                state = event.state
                final_message = event.final_message
                spend = event.spend

        if req.session and state == AgentState.DONE:
            store.save(req.session, transcript)

        return ChatResponse(
            state=state,
            message=final_message.text() if final_message else None,
            spend=spend,
            detail=last_detail if state != AgentState.DONE else None,
        )

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
        await websocket.accept()
        try:
            payload = await websocket.receive_json()
            message = payload.get("message", "")
            session = payload.get("session")
            auto = bool(payload.get("auto", False))
            model = payload.get("model")

            async def ws_confirm(call: ToolCallBlock) -> bool:
                reply = await websocket.receive_json()
                return bool(reply.get("approved", False))

            store = SessionStore()
            try:
                history = store.load(session) if session else []
                # The WS counterpart to the same real bug just fixed for
                # /chat: a malformed "image_base64" made
                # base64.b64decode() raise binascii.Error (a ValueError
                # subclass) with nothing here to catch it -- the whole
                # ASGI call crashed with no frame sent at all, the client
                # saw a bare ClosedResourceError, confirmed directly with
                # a real TestClient WebSocket session before this fix.
                # Sharing this try block gives it the identical clean
                # failure treatment the invalid-session-name case below
                # already has.
                extra_content = _extra_content_blocks(
                    payload.get("image_base64"), payload.get("image_media_type")
                )
            except ValueError as e:
                # SessionStore._sanitize() raises a plain ValueError for
                # an invalid session name, and reaching this point
                # uncaught didn't even give the client the REST
                # endpoint's own clean detail message -- it crashed the
                # whole ASGI call with no frame sent at all, and the
                # client saw a bare ClosedResourceError, confirmed
                # directly with a real TestClient WebSocket session
                # before this fix. Reported as a real state_changed +
                # run_done pair -- the exact same shape an unknown
                # --model already produces -- so App.tsx's existing
                # failure-detail handling (see BUILD-JOURNAL.md) shows it
                # with no client-side changes needed.
                await websocket.send_text(
                    StateChangedEvent(state=AgentState.FAILED, detail=str(e)).model_dump_json()
                )
                await websocket.send_text(
                    RunDoneEvent(
                        state=AgentState.FAILED, final_message=None, spend=Spend()
                    ).model_dump_json()
                )
                return

            loop = AgentLoop(
                router=build_router(),
                providers=build_providers(),
                tools=BUILTIN_TOOLS,
                confirm=always_allow if auto else ws_confirm,
                degraders=default_degraders(),
            )
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
                store.save(session, transcript)
        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()

    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="web-ui")

    return app
