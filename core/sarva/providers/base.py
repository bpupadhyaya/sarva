"""sarva.providers.base — the provider contract.

Every model backend (Anthropic, OpenAI, Google, local Ollama/MLX, and future
Sarva-foundry models) implements the single `Provider` protocol here. Adding
a model to Sarva means writing one thin adapter against this file plus a
`models.yaml` entry — nothing in the agent loop, tools, memory, or any skin
ever changes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

from sarva.multimodal.content import ContentBlock, Message, Modality, ToolCallBlock

# ---------- Registry data model (models.yaml rows) ----------


class ModelCapabilities(BaseModel):
    model_config = {"frozen": True}
    modalities_in: set[Modality]
    modalities_out: set[Modality]  # v1: {TEXT}; image-out models later
    tool_use: bool
    thinking: bool
    context_window: int  # tokens
    max_output: int  # tokens


class ModelCost(BaseModel):
    model_config = {"frozen": True}
    input_per_mtok: float = 0.0  # USD; 0 for local models
    output_per_mtok: float = 0.0


class ModelInfo(BaseModel):
    model_config = {"frozen": True}
    id: str  # e.g. "claude-opus-4-8", "ollama/qwen3:8b"
    provider: str  # registry key of the Provider impl
    display_name: str
    capabilities: ModelCapabilities
    cost: ModelCost
    local: bool = False


# ---------- Request ----------


class ToolSpec(BaseModel):
    model_config = {"frozen": True}
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema
    destructive: bool = False  # consumed by the agent loop, not providers


class GenerateConfig(BaseModel):
    model_config = {"frozen": True}
    max_tokens: int = 8192
    effort: Literal["low", "medium", "high", "max"] | None = None
    thinking: bool | None = None  # None = provider/model default
    stop_sequences: list[str] = []


class GenerateRequest(BaseModel):
    model_config = {"frozen": True}
    model: str  # ModelInfo.id
    system: str | None = None
    messages: list[Message]
    tools: list[ToolSpec] = []
    config: GenerateConfig = GenerateConfig()


# ---------- Stream events ----------


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"


class Usage(BaseModel):
    model_config = {"frozen": True}
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0  # computed from ModelCost by the adapter


class _Event(BaseModel):
    model_config = {"frozen": True}


class TextDeltaEvent(_Event):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDeltaEvent(_Event):
    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class ToolCallEvent(_Event):
    type: Literal["tool_call"] = "tool_call"
    call: ToolCallBlock  # complete — args fully parsed


class ContentEvent(_Event):
    """Non-text output blocks (e.g. images from image-out models)."""

    type: Literal["content"] = "content"
    block: ContentBlock


class DoneEvent(_Event):
    type: Literal["done"] = "done"
    stop_reason: StopReason
    message: Message  # the fully-assembled assistant message
    usage: Usage


class StreamErrorEvent(_Event):
    type: Literal["stream_error"] = "stream_error"
    code: str  # "rate_limit" | "overloaded" | "network" | "provider" | ...
    detail: str
    retryable: bool


ProviderEvent = Annotated[
    (
        TextDeltaEvent
        | ThinkingDeltaEvent
        | ToolCallEvent
        | ContentEvent
        | DoneEvent
        | StreamErrorEvent
    ),
    Field(discriminator="type"),
]


# ---------- Errors (pre-stream only) ----------


class ProviderError(Exception):
    pass


class AuthError(ProviderError):
    pass


class ModelNotFoundError(ProviderError):
    pass


class ContextOverflowError(ProviderError):
    pass


class RateLimitError(ProviderError):
    def __init__(self, detail: str, retry_after_s: float | None = None):
        super().__init__(detail)
        self.retry_after_s = retry_after_s


# ---------- The contract ----------


class Provider(Protocol):
    """One implementation per backend. Adapters are THIN: translate request
    to wire format, translate wire events to ProviderEvent. No retries beyond
    the SDK's own, no routing, no degradation — those are the caller's job."""

    name: str

    def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]: ...

    async def close(self) -> None: ...


async def complete(provider: Provider, request: GenerateRequest) -> DoneEvent:
    """Non-streaming convenience: drain the stream, return the DoneEvent."""
    async for event in provider.generate(request):
        if isinstance(event, DoneEvent):
            return event
        if isinstance(event, StreamErrorEvent):
            raise ProviderError(f"{event.code}: {event.detail}")
    raise RuntimeError("provider stream ended without a DoneEvent")
