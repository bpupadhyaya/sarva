"""sarva.providers.ollama_provider — the Ollama adapter.

Talks to a local Ollama server's `/api/chat` endpoint (newline-delimited
JSON streaming). This is what makes the "free & private" tier real: no
network egress beyond localhost, no API key, no cost.

NOTE: written to Ollama's documented chat API shape but not yet exercised
against a running Ollama instance in this environment — mark its
conformance tests `@pytest.mark.live` (skipped unless OLLAMA_HOST is
reachable) until a real run validates it. See BUILD-JOURNAL.md.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from sarva.multimodal.content import Message, TextBlock, ToolCallBlock, ToolResultBlock
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


def _to_ollama_message(m: Message) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            text_parts.append(b.text)
        elif isinstance(b, ToolCallBlock):
            tool_calls.append({"function": {"name": b.name, "arguments": b.arguments}})
        elif isinstance(b, ToolResultBlock):
            # Ollama has no dedicated tool-result role; render as a tool message.
            text_parts.append("".join(c.text for c in b.content if isinstance(c, TextBlock)))
    out: dict[str, Any] = {"role": m.role, "content": "".join(text_parts)}
    if tool_calls:
        out["tool_calls"] = tool_calls
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
        payload: dict[str, Any] = {
            "model": _strip_local_prefix(request.model),
            "messages": (
                ([{"role": "system", "content": request.system}] if request.system else [])
                + [_to_ollama_message(m) for m in request.messages]
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
