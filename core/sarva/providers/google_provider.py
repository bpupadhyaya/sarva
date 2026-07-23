"""sarva.providers.google_provider — the Google Gemini adapter.

Thin: translates GenerateRequest into `google-genai`'s `Content`/`Part`
shape and translates streamed `GenerateContentResponse` chunks into
ProviderEvent — the same contract every other adapter implements (§3.1).
Closes T1's last named provider gap ("Provider layer (Anthropic+OpenAI+
Google+Ollama)"); Anthropic, OpenAI, Ollama, and Mock already existed.

NOTE: written to the documented `google-genai` Python SDK's streaming
shape but not yet exercised against a live API key in this environment —
mark its conformance tests `@pytest.mark.live` (skipped without
GEMINI_API_KEY/GOOGLE_API_KEY) until a real run validates it, same
discipline as every other adapter before its first live run. See
BUILD-JOURNAL.md.

Same deliberate scope boundary as openai_provider.py: no entries added
to `providers/data/models.yaml`. That file states it's "re-validated at
every release," and this session has no verified-current Gemini model
catalog (IDs, capabilities, per-token pricing) to add responsibly rather
than guess. The adapter is the code-side half of "add a model = one
registry entry" — wiring a specific verified model in is left for
whoever has that data.

Also deliberately unmapped: `GenerateConfig.effort`/`.thinking`. Gemini's
"thinking" models use a separate `thinking_config` shape this session
has not verified against a live model, and applying it blindly to
non-thinking registry entries risks a real request failure rather than a
hypothetical one — same reasoning openai_provider.py names for
`reasoning_effort`.

Also honestly named as unhandled: network-level connection failures.
Unlike the `anthropic`/`openai` SDKs, which document a dedicated
`APIConnectionError`, this session found no equivalent documented
exception type for `google-genai` to catch with confidence -- only
`errors.ClientError`/`errors.ServerError` (both `errors.APIError`
subclasses, covering HTTP-level failures) are handled below. A real
connection failure will surface as an uncaught exception rather than a
`StreamErrorEvent` until verified against a live run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import errors, types

from sarva.multimodal.content import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from sarva.multimodal.fetch import resolve_media_bytes
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

_STOP_REASON_MAP = {
    types.FinishReason.STOP: StopReason.END_TURN,
    types.FinishReason.MAX_TOKENS: StopReason.MAX_TOKENS,
    types.FinishReason.SAFETY: StopReason.REFUSAL,
    types.FinishReason.PROHIBITED_CONTENT: StopReason.REFUSAL,
    types.FinishReason.BLOCKLIST: StopReason.REFUSAL,
}


def _tool_call_names(messages: list[Message]) -> dict[str, str]:
    """Gemini's `FunctionResponse` requires the function's `name`, but
    Sarva's `ToolResultBlock` only carries `tool_call_id` (matching every
    other provider's tool-result shape, which needs no name). Resolve it
    by scanning the earlier `ToolCallBlock` that made the call."""
    names: dict[str, str] = {}
    for m in messages:
        for b in m.content:
            if isinstance(b, ToolCallBlock):
                names[b.id] = b.name
    return names


async def _to_gemini_content(m: Message, call_names: dict[str, str]) -> types.Content:
    role = "model" if m.role == "assistant" else "user"
    parts: list[types.Part] = []
    for b in m.content:
        if isinstance(b, TextBlock):
            parts.append(types.Part(text=b.text))
        elif isinstance(b, ImageBlock):
            # resolve_media_bytes (not b.resolve_bytes()) so url-sourced
            # images work too, not just data/path.
            image_bytes = await resolve_media_bytes(b)
            blob = types.Blob(data=image_bytes, mime_type=b.media_type)
            parts.append(types.Part(inline_data=blob))
        elif isinstance(b, ToolCallBlock):
            call = types.FunctionCall(id=b.id, name=b.name, args=b.arguments)
            parts.append(types.Part(function_call=call))
        elif isinstance(b, ToolResultBlock):
            text = "".join(c.text for c in b.content if isinstance(c, TextBlock))
            response = {"error": text} if b.is_error else {"output": text}
            name = call_names.get(b.tool_call_id, b.tool_call_id)
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=b.tool_call_id, name=name, response=response
                    )
                )
            )
        elif isinstance(b, ThinkingBlock):
            # Deliberately, explicitly dropped -- not silently: Gemini's
            # "thought" parts are surfaced on the way out (see
            # ThinkingDeltaEvent in generate() below) but there's no
            # documented way to feed one back in as part-of-request
            # content yet. Explicit here so it's a named, intentional
            # skip rather than an unhandled type quietly falling through
            # with no case at all.
            continue
        else:
            # A block type this adapter has no translation for at all
            # (e.g. DocumentBlock reaching this adapter directly,
            # unconverted -- it has a degrader now, but only
            # degrade_message()'s opt-in fallback path uses it; a caller
            # that skips degradation, or a model whose registry entry
            # claims document support it doesn't actually have wire-level
            # code for, still reaches here). Raising here is deliberate:
            # silently omitting it would send the request missing content
            # the caller believes is present, and the model would answer
            # as if it had read something it never received -- a
            # materially misleading response, not a cosmetic gap. See
            # docs/multimodal.md for the fuller story.
            raise ValueError(
                f"GoogleProvider cannot translate a {type(b).__name__!r} content block "
                "(no wire-format mapping exists for it yet)"
            )
    return types.Content(role=role, parts=parts)


class GoogleProvider:
    name = "google"

    def __init__(self, client: genai.Client | None = None):
        self._client = client or genai.Client()

    async def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]:
        call_names = _tool_call_names(request.messages)
        contents = [await _to_gemini_content(m, call_names) for m in request.messages]

        config = types.GenerateContentConfig(
            max_output_tokens=request.config.max_tokens,
            stop_sequences=request.config.stop_sequences or None,
        )
        if request.system:
            config.system_instruction = request.system
        if request.tools:
            config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters_json_schema=t.input_schema,
                        )
                        for t in request.tools
                    ]
                )
            ]

        text_acc = ""
        blocks: list[object] = []
        finish_reason: types.FinishReason | None = None
        usage: types.GenerateContentResponseUsageMetadata | None = None

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=request.model, contents=contents, config=config
            )
            async for chunk in stream:
                if chunk.usage_metadata:
                    usage = chunk.usage_metadata
                if not chunk.candidates:
                    continue
                candidate = chunk.candidates[0]
                if candidate.finish_reason:
                    finish_reason = candidate.finish_reason
                if not candidate.content or not candidate.content.parts:
                    continue
                for part in candidate.content.parts:
                    if part.text and not part.thought:
                        text_acc += part.text
                        yield TextDeltaEvent(text=part.text)
                    elif part.text and part.thought:
                        yield ThinkingDeltaEvent(text=part.text)
                    elif part.function_call:
                        call = ToolCallBlock(
                            id=part.function_call.id or part.function_call.name,
                            name=part.function_call.name or "",
                            arguments=part.function_call.args or {},
                        )
                        blocks.append(call)
                        yield ToolCallEvent(call=call)
        except errors.ClientError as e:
            yield StreamErrorEvent(
                code="rate_limit" if e.code == 429 else "provider",
                detail=str(e),
                retryable=e.code == 429,
            )
            return
        except errors.ServerError as e:
            yield StreamErrorEvent(code="provider", detail=str(e), retryable=True)
            return

        if text_acc:
            blocks.insert(0, TextBlock(text=text_acc))

        # Gemini has no distinct "made a tool call" finish_reason -- it
        # reports STOP even when the response includes function_call
        # parts (unlike Anthropic/OpenAI, whose finish reason says so
        # directly). Presence of a tool call block always wins over the
        # raw finish_reason, which would otherwise misreport TOOL_USE
        # turns as END_TURN.
        if any(isinstance(b, ToolCallBlock) for b in blocks):
            stop_reason = StopReason.TOOL_USE
        elif finish_reason:
            stop_reason = _STOP_REASON_MAP.get(finish_reason, StopReason.END_TURN)
        else:
            stop_reason = StopReason.END_TURN
        yield DoneEvent(
            stop_reason=stop_reason,
            message=Message(role="assistant", content=blocks),
            usage=Usage(
                input_tokens=usage.prompt_token_count if usage and usage.prompt_token_count else 0,
                output_tokens=(
                    usage.candidates_token_count if usage and usage.candidates_token_count else 0
                ),
                cache_read_tokens=(
                    usage.cached_content_token_count
                    if usage and usage.cached_content_token_count
                    else 0
                ),
                # Real per-token pricing needs a verified-current entry in
                # models.yaml (see module docstring) -- reporting cost_usd=0
                # here rather than a guessed/fabricated number.
                cost_usd=0.0,
            ),
        )

    async def close(self) -> None:
        pass
