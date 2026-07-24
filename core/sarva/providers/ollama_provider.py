"""sarva.providers.ollama_provider — the Ollama adapter.

Talks to a local Ollama server's `/api/chat` endpoint (newline-delimited
JSON streaming). This is what makes the "free & private" tier real: no
network egress beyond localhost, no API key, no cost.

Verified live, not just written to the documented shape: unlike
Anthropic/OpenAI/Google (which need a real credential this environment
doesn't have), Ollama needs only a locally running server — `brew
install ollama`, `ollama serve`, `ollama pull qwen2.5:0.5b` (a small
model, not the ~5GB `qwen3:8b` registered as the real default in
`models.yaml`), then a genuine `tests/live/test_live_providers.py::
test_ollama_terminal_event_law` run (see `OLLAMA_TEST_MODEL` there for
overriding the pulled model) plus direct streaming and tool-call checks
against the real running server — all passed. See BUILD-JOURNAL.md for
the full verification record.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from sarva.multimodal.content import ImageBlock, Message, TextBlock, ToolCallBlock, ToolResultBlock
from sarva.multimodal.fetch import resolve_media_bytes
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    ProviderEvent,
    StopReason,
    StreamErrorEvent,
    TextDeltaEvent,
    ToolCallEvent,
    Usage,
)

_DEFAULT_HOST = "http://localhost:11434"


async def _to_ollama_message(m: Message) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    images: list[str] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            text_parts.append(b.text)
        elif isinstance(b, ImageBlock):
            # resolve_media_bytes (not b.resolve_bytes()) so url-sourced
            # images work too, not just data/path -- see
            # sarva.multimodal.fetch's module docstring. Ollama's own
            # `/api/chat` wire format wants raw base64 (no data: URI
            # prefix, no media_type field) in a per-message `images`
            # array, confirmed against a real running server with a real
            # vision-capable model (moondream) before writing this --
            # not just written to documented shape.
            image_bytes = await resolve_media_bytes(b)
            images.append(base64.standard_b64encode(image_bytes).decode())
        elif isinstance(b, ToolCallBlock):
            tool_calls.append({"function": {"name": b.name, "arguments": b.arguments}})
        elif isinstance(b, ToolResultBlock):
            # Ollama has no dedicated tool-result role; render as a tool message.
            text_parts.append("".join(c.text for c in b.content if isinstance(c, TextBlock)))
        else:
            # A block type this adapter has no translation for at all.
            # Raising here is deliberate, matching the
            # Anthropic/OpenAI/Google/Foundry adapters' own guards:
            # silently omitting it would send the request missing
            # content the caller believes is present, and the model
            # would answer as if it had read something it never received
            # -- a materially misleading response, not a cosmetic gap.
            raise ValueError(
                f"OllamaProvider cannot translate a {type(b).__name__!r} content block "
                "(no wire-format mapping exists for it yet)"
            )
    out: dict[str, Any] = {"role": m.role, "content": "".join(text_parts)}
    if tool_calls:
        out["tool_calls"] = tool_calls
    if images:
        out["images"] = images
    return out


def _strip_local_prefix(model_id: str) -> str:
    """Registry ids are namespaced ("ollama/qwen3:8b"); Ollama's own API
    wants the bare tag ("qwen3:8b")."""
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


class OllamaProvider:
    name = "ollama"

    def __init__(self, host: str | None = None, client: httpx.AsyncClient | None = None):
        self._host = host or _DEFAULT_HOST
        self._client = client or httpx.AsyncClient(timeout=120.0)

    async def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]:
        messages = [await _to_ollama_message(m) for m in request.messages]
        payload: dict[str, Any] = {
            "model": _strip_local_prefix(request.model),
            "messages": (
                ([{"role": "system", "content": request.system}] if request.system else [])
                + messages
            ),
            "stream": True,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]

        text_acc = ""
        tool_calls_seen: list[dict[str, Any]] = []
        done_reason: StopReason = StopReason.END_TURN

        try:
            async with self._client.stream(
                "POST", f"{self._host}/api/chat", json=payload
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield StreamErrorEvent(
                        code="provider",
                        detail=f"{response.status_code}: {body.decode(errors='replace')}",
                        retryable=response.status_code >= 500,
                    )
                    return
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    msg = chunk.get("message", {})
                    if msg.get("content"):
                        text_acc += msg["content"]
                        yield TextDeltaEvent(text=msg["content"])
                    for tc in msg.get("tool_calls", []) or []:
                        tool_calls_seen.append(tc)
                        done_reason = StopReason.TOOL_USE
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as e:
            yield StreamErrorEvent(
                code="network",
                detail=f"cannot reach Ollama at {self._host}: {e}",
                retryable=True,
            )
            return
        except httpx.TimeoutException as e:
            yield StreamErrorEvent(code="network", detail=str(e), retryable=True)
            return

        content: list[object] = []
        if text_acc:
            content.append(TextBlock(text=text_acc))
        for i, tc in enumerate(tool_calls_seen):
            fn = tc.get("function", {})
            call = ToolCallBlock(
                id=f"ollama-{i}",
                name=fn.get("name", ""),
                arguments=fn.get("arguments", {}),
            )
            content.append(call)
            yield ToolCallEvent(call=call)

        yield DoneEvent(
            stop_reason=done_reason,
            message=Message(role="assistant", content=content),
            usage=Usage(),  # local inference — no token accounting from Ollama's chat API
        )

    async def close(self) -> None:
        await self._client.aclose()
