"""sarva.server.schemas — request/response models for the HTTP/WS API."""

from __future__ import annotations

from pydantic import BaseModel

from sarva.agent.budget import Spend


class ChatRequest(BaseModel):
    message: str
    session: str | None = None
    image_base64: str | None = None
    image_media_type: str | None = None


class ChatResponse(BaseModel):
    state: str
    message: str | None
    spend: Spend


class ModelInfoOut(BaseModel):
    id: str
    display_name: str
    available: bool
