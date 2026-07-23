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


class SaveConfigRequest(BaseModel):
    """Only the four provider-key names `sarva.config` knows about are
    accepted — an explicit allowlist (validated in the route handler, not
    just documented here) rather than writing arbitrary caller-supplied
    keys straight into a config file the backend later trusts."""

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None


class DoctorCheckOut(BaseModel):
    name: str
    ok: bool
    detail: str
