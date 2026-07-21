"""sarva.server.app — the FastAPI server: REST + WebSocket over the agent loop.

Zero-config by default, same as the CLI: with no ANTHROPIC_API_KEY set,
everything routes to the offline MockProvider. `/chat` is single-turn
non-streaming (mirrors `sarva chat`); `/ws/chat` streams the same
AgentEvents the CLI renders, one JSON frame per event, ending with a
`run_done` frame.
"""

from __future__ import annotations

import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from sarva.agent.budget import Spend
from sarva.agent.events import AgentState
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import always_allow
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ContentBlock, ImageBlock, Message
from sarva.runtime import build_providers, build_router
from sarva.server.schemas import ChatRequest, ChatResponse, ModelInfoOut


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
        """Single-turn per connection, mirroring /chat: the client sends one
        {"message": ..., "session": ...} frame and receives streamed
        AgentEvent JSON frames ending with a run_done frame, then the
        connection closes."""
        await websocket.accept()
        try:
            payload = await websocket.receive_json()
            message = payload.get("message", "")
            session = payload.get("session")

            store = SessionStore()
            history = store.load(session) if session else []

            loop = AgentLoop(
                router=build_router(), providers=build_providers(), tools=[], confirm=always_allow
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

    return app
