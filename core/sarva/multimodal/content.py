"""sarva.multimodal.content — the typed multimodal content model.

The universal currency of Sarva: every input (user text, image, PDF, audio,
video), every model output (text, thinking, tool calls), and every tool
result is a typed, immutable block. Providers, the agent loop, memory, and
every skin speak only this vocabulary — nothing passes a raw provider dict
across a module boundary.

Design notes:
  * Blocks are frozen Pydantic models (pure data). Behavior lives in
    functions that pattern-match on `.type`, not in subclass methods.
  * A media block carries exactly one of data/path/url; loading bytes is
    lazy and explicit via `resolve_bytes()`.
  * Degradation is a registry of converters, applied recursively, that
    either produces blocks the target model supports or raises — content
    is never silently dropped.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class _Block(BaseModel):
    # ser_json_bytes/val_json_bytes=base64: bytes fields (media data) are
    # binary, not UTF-8 text — Pydantic's plain JSON mode tries UTF-8 decode
    # by default and breaks on real image/audio/video bytes. Base64 is
    # required for any block carrying binary `data`.
    model_config = {
        "frozen": True,
        "extra": "forbid",
        "ser_json_bytes": "base64",
        "val_json_bytes": "base64",
    }


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(_Block):
    type: Literal["thinking"] = "thinking"
    text: str  # may be "" for providers that omit thinking text
    provider_data: dict[str, Any] | None = None  # opaque; echo back unmodified


class _MediaBlock(_Block):
    media_type: str  # IANA type, e.g. "image/png"
    data: bytes | None = None  # exactly one of data/path/url
    path: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> _MediaBlock:
        sources = [s for s in (self.data, self.path, self.url) if s is not None]
        if len(sources) != 1:
            raise ValueError("exactly one of data/path/url must be set")
        return self

    def resolve_bytes(self) -> bytes:
        """Load the raw bytes. Explicit, possibly slow. url sources must be
        fetched via sarva.multimodal.fetch (not implemented here)."""
        if self.data is not None:
            return self.data
        if self.path is not None:
            return Path(self.path).read_bytes()
        raise ValueError("url sources must be fetched via sarva.multimodal.fetch")


class ImageBlock(_MediaBlock):
    type: Literal["image"] = "image"


class AudioBlock(_MediaBlock):
    type: Literal["audio"] = "audio"
    duration_s: float | None = None


class VideoBlock(_MediaBlock):
    type: Literal["video"] = "video"
    duration_s: float | None = None


class DocumentBlock(_MediaBlock):
    type: Literal["document"] = "document"  # PDFs, docx, ...
    title: str | None = None


class ToolCallBlock(_Block):
    type: Literal["tool_call"] = "tool_call"
    id: str  # provider-assigned or uuid; unique within a transcript
    name: str
    arguments: dict[str, Any]


class ToolResultBlock(_Block):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: list[ContentBlock]  # usually TextBlock/ImageBlock
    is_error: bool = False


ContentBlock = Annotated[
    (
        TextBlock
        | ThinkingBlock
        | ImageBlock
        | AudioBlock
        | VideoBlock
        | DocumentBlock
        | ToolCallBlock
        | ToolResultBlock
    ),
    Field(discriminator="type"),
]

# ToolResultBlock.content references ContentBlock before it exists at class
# definition time; resolve the forward reference now that it's in scope.
ToolResultBlock.model_rebuild()


class Message(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    role: Literal["system", "user", "assistant"]
    content: list[ContentBlock]

    def text(self) -> str:
        """Concatenated text of all TextBlocks — the 'just give me the words' helper."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))


MODALITY_OF: dict[str, Modality] = {
    "text": Modality.TEXT,
    "thinking": Modality.TEXT,
    "tool_call": Modality.TEXT,
    "tool_result": Modality.TEXT,
    "image": Modality.IMAGE,
    "audio": Modality.AUDIO,
    "video": Modality.VIDEO,
    "document": Modality.DOCUMENT,
}


def modality_of(block: Any) -> Modality:
    return MODALITY_OF[block.type]


# ---------- Degradation (registry of converters, never silent drops) ----------


class UnsupportedModalityError(Exception):
    """No degradation path exists from a block's modality to the model's inputs."""


class Degrader(Protocol):
    """Converts blocks of `source` modality into blocks of strictly 'lower'
    modalities (e.g. video -> [image frames + text transcript])."""

    source: Modality

    async def degrade(self, block: Any) -> list[Any]: ...


async def degrade_message(
    msg: Message,
    supported: set[Modality],
    degraders: dict[Modality, Degrader],
) -> Message:
    """Return a Message every block of which the target model can consume.
    Applies degraders recursively (video->frames+audio, audio->text) until
    all blocks are supported. Raises UnsupportedModalityError if any block
    has no path. Never silently drops a block."""
    out: list[Any] = []
    for block in msg.content:
        out.extend(await _degrade_block(block, supported, degraders, depth=0))
    return Message(role=msg.role, content=out)


async def _degrade_block(
    block: Any,
    supported: set[Modality],
    degraders: dict[Modality, Degrader],
    depth: int,
) -> list[Any]:
    m = modality_of(block)
    if m in supported:
        return [block]
    if depth > 3 or m not in degraders:
        raise UnsupportedModalityError(f"no path for {m} -> {supported}")
    produced = await degraders[m].degrade(block)
    out: list[Any] = []
    for b in produced:
        out.extend(await _degrade_block(b, supported, degraders, depth + 1))
    return out
