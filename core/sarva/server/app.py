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
from sarva.agent.events import AgentState
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import BUILTIN_TOOLS, always_allow
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ContentBlock, ImageBlock, Message, ToolCallBlock
from sarva.runtime import build_providers, build_router
from sarva.server.schemas import ChatRequest, ChatResponse, ModelInfoOut

_STATIC_DIR = Path(__file__).parent / "static"


def _extra_content_from(req: ChatRequest) -> list[ContentBlock]:
    if req.image_base64 and req.image_media_type:
        return [
            ImageBlock(
                media_type=req.image_media_type,
                data=base64.b64decode(req.image_base64),
            )
        ]
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

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        store = SessionStore()
        history = store.load(req.session) if req.session else []
        extra_content = _extra_content_from(req)

        loop = AgentLoop(
            router=build_router(), providers=build_providers(), tools=[], confirm=always_allow
        )

        state = AgentState.FAILED
        final_message: Message | None = None
        spend = Spend()
        transcript: list[Message] = []
        async for event in loop.run(
            req.message, history=history, extra_content=extra_content, transcript_out=transcript
        ):
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
        )

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        """Single-turn per connection, tools enabled (mirrors `sarva run`,
        not `sarva chat`). The client sends one
        {"message": ..., "session": ..., "auto": false} frame and receives
        streamed AgentEvent JSON frames ending with a run_done frame.

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

            async def ws_confirm(call: ToolCallBlock) -> bool:
                reply = await websocket.receive_json()
                return bool(reply.get("approved", False))

            store = SessionStore()
            history = store.load(session) if session else []

            loop = AgentLoop(
                router=build_router(),
                providers=build_providers(),
                tools=BUILTIN_TOOLS,
                confirm=always_allow if auto else ws_confirm,
            )
            state = AgentState.FAILED
            transcript: list[Message] = []
            async for event in loop.run(message, history=history, transcript_out=transcript):
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
