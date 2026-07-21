"""sarva.providers.mock — deterministic, offline provider.

Drives the conformance suite (no network, no API key needed in CI) and
gives `sarva chat`/`sarva run` a zero-config path that works out of the box.

Default behavior (no script): echoes the last user message back as a single
streamed turn ending in END_TURN. Pass `script` to drive specific scenarios
for tests — tool calls, refusals, mid-stream errors, budget exhaustion.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sarva.multimodal.content import Message, TextBlock, ThinkingBlock, ToolCallBlock
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    ProviderEvent,
    StopReason,
    StreamErrorEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    Usage,
)


@dataclass
class ScriptedTurn:
    """One canned response for one generate() call. At most one of
    text/tool_calls/error/refuse is meaningful per turn."""

    text: str | None = None
    tool_calls: list[ToolCallBlock] | None = None
    thinking: str | None = None
    error: str | None = None  # simulate a mid-stream StreamErrorEvent, then stop
    error_retryable: bool = False
    refuse: bool = False


class MockProvider:
    name = "mock"

    def __init__(self, script: list[ScriptedTurn] | None = None):
        self._script = list(script) if script is not None else None
        self._turn = 0

    def _next_turn(self, request: GenerateRequest) -> ScriptedTurn:
        if self._script is None:
            last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
            echoed = last_user.text() if last_user else ""
            return ScriptedTurn(text=f"[mock] received: {echoed}")
        turn = self._script[min(self._turn, len(self._script) - 1)]
        self._turn += 1
        return turn

    async def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]:
        turn = self._next_turn(request)
        await asyncio.sleep(0)  # yield control at least once; keep it genuinely async

        if turn.refuse:
            yield DoneEvent(
                stop_reason=StopReason.REFUSAL,
                message=Message(role="assistant", content=[]),
                usage=Usage(),
            )
            return

        if turn.error is not None:
            yield StreamErrorEvent(
                code="mock_error", detail=turn.error, retryable=turn.error_retryable
            )
            return

        content: list[object] = []
        if turn.thinking:
            yield ThinkingDeltaEvent(text=turn.thinking)
            content.append(ThinkingBlock(text=turn.thinking))

        if turn.tool_calls:
            for call in turn.tool_calls:
                yield ToolCallEvent(call=call)
                content.append(call)
            stop_reason = StopReason.TOOL_USE
        else:
            text = turn.text or ""
            for word in text.split(" "):
                yield TextDeltaEvent(text=word + " ")
                await asyncio.sleep(0)
            content.append(TextBlock(text=text))
            stop_reason = StopReason.END_TURN

        yield DoneEvent(
            stop_reason=stop_reason,
            message=Message(role="assistant", content=content),
            usage=Usage(
                input_tokens=max(1, len(str(request.messages)) // 4),
                output_tokens=max(1, sum(len(str(c)) for c in content) // 4),
            ),
        )

    async def close(self) -> None:
        return None
