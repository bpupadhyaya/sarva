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

Also translates image-out: an image-capable Gemini model can return a
`Part` with `inline_data` populated (the same `Blob` shape used to send
images in) instead of, or alongside, text. `ModelCapabilities.
modalities_out` and `ContentEvent` both named "image-out models" as
future work since T1; this is the first adapter to actually produce
one. Still gated the same way as the rest of this file: no
`models.yaml` entry claims an image-out-capable Gemini model id yet
(this session has no verified-current catalog of which Gemini model
variants actually support it, or their pricing), so this is real,
reachable code with no registry entry routing a real request to it —
the same "adapter exists, wiring a specific verified model in is
separate" scoping this file already applies to Gemini generally.

Also translates native video-in: `VideoBlock` -> `types.Blob(data=...,
mime_type=...)`, the exact same `inline_data` shape already used for
images, sent to Gemini instead of degraded to sampled frames first.
Named directly in the design doc's own T5 roadmap line ("MCP client,
video input") as a still-open deliverable; `sarva.multimodal.degraders.
VideoToTextDegrader` (frame-sampling fallback for models with no native
video support) stays exactly as useful as before for every OTHER
model, or for a caller who skips this adapter's real path -- this is
additive, not a replacement. Honestly scoped on size: inline `Blob`
data is base64-encoded in the wire request, which Gemini's documented
limits cap at roughly 20MB total request size -- fine for the short
clips this project's own examples/tests use, but a real caller with a
long video would need Gemini's separate Files API (upload once,
reference by URI), left as real, separate, deferred follow-up rather
than silently mishandled here.

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

A real bug found (and fixed) in the same family: `errors.
UnknownApiResponseError` -- raised by the SDK's own
`_load_json_from_response()` when a streaming response chunk fails
`json.loads()` -- is a `ValueError` subclass, NOT an `errors.APIError`
subclass, so `errors.ClientError`/`errors.ServerError` never touch it.
Reproduced with a duck-typed fake client (no live API key needed, the
same discipline the identical Anthropic/OpenAI gaps were found and
fixed with) and given its own `except` clause below.
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
    VideoBlock,
)
from sarva.multimodal.fetch import resolve_media_bytes
from sarva.providers.base import (
    ContentEvent,
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
        elif isinstance(b, VideoBlock):
            # Same inline_data shape as images -- Gemini's real, native
            # video understanding, not a degraded-to-frames fallback.
            # See the module docstring for the real ~20MB inline-request
            # size caveat.
            video_bytes = await resolve_media_bytes(b)
            blob = types.Blob(data=video_bytes, mime_type=b.media_type)
            parts.append(types.Part(inline_data=blob))
        elif isinstance(b, ToolCallBlock):
            call = types.FunctionCall(id=b.id, name=b.name, args=b.arguments)
            parts.append(types.Part(function_call=call))
        elif isinstance(b, ToolResultBlock):
            # A real bug found by actually constructing a ToolResultBlock
            # carrying an ImageBlock (e.g. a screenshot/image-generation
            # tool's result -- ToolResultBlock.content's own type comment
            # already names this as an anticipated shape, not a
            # hypothetical): the plain `"".join(... TextBlock)` here
            # silently dropped it with no error. Gemini's own SDK type
            # (`FunctionResponse.parts: list[FunctionResponsePart]`)
            # genuinely supports attaching inline media alongside a
            # function response, confirmed by reading it directly, not
            # assumed -- so, like Anthropic and unlike OpenAI, this is a
            # real capability gap, not a wire-format limitation.
            text = "".join(c.text for c in b.content if isinstance(c, TextBlock))
            response_parts: list[types.FunctionResponsePart] = []
            for c in b.content:
                if isinstance(c, TextBlock):
                    continue
                elif isinstance(c, ImageBlock):
                    c_bytes = await resolve_media_bytes(c)
                    response_parts.append(
                        types.FunctionResponsePart.from_bytes(data=c_bytes, mime_type=c.media_type)
                    )
                else:
                    raise ValueError(
                        f"GoogleProvider cannot translate a {type(c).__name__!r} content "
                        "block inside a tool result (no wire-format mapping exists for it "
                        "yet)"
                    )
            response = {"error": text} if b.is_error else {"output": text}
            name = call_names.get(b.tool_call_id, b.tool_call_id)
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=b.tool_call_id,
                        name=name,
                        response=response,
                        parts=response_parts or None,
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
        # A real bug found by giving this adapter's tool-call id handling
        # its own fresh-eyes sweep: `function_call.id` is documented as
        # optional on the wire (google-genai's own `FunctionCall.id`
        # docstring), and Gemini frequently leaves it unset -- the
        # previous fallback (`id or name`) collapsed every tool call
        # sharing a name onto the SAME id whenever id was missing.
        # Confirmed live: two ordinary parallel calls to the same tool in
        # one turn (e.g. `get_weather("NYC")` + `get_weather("LA")`, a
        # completely ordinary agentic pattern) both got
        # `id="get_weather"`, and that collision propagated all the way
        # to the wire -- `_to_gemini_content` echoes `tool_call_id` into
        # `FunctionResponse.id`, a field Gemini's own docs say exists so
        # the model can match a response back to the call that produced
        # it, so both responses reached the model carrying the identical
        # id, defeating that correlation for exactly the two calls that
        # most needed it disambiguated. Counter-based synthetic ids are
        # unique within this turn regardless of how many calls share a
        # name or how many distinct names appear.
        _synthetic_call_id = 0

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
                        # A real bug found by giving this adapter's block
                        # ordering its own fresh-eyes sweep: text was
                        # accumulated into `text_acc` across the WHOLE
                        # stream and only ever spliced into `blocks` once,
                        # at the very front, after the loop ended --
                        # unlike ToolCallBlock/ImageBlock below, which
                        # were already appended in true chronological
                        # order as each part arrived. An ordinary
                        # sequential-tool-calling turn (reasoning text,
                        # then a call, then more reasoning text, then
                        # another call -- documented Gemini behavior, not
                        # contrived) confirmed live: both text segments
                        # got concatenated into ONE TextBlock hoisted
                        # ahead of BOTH tool calls, misrepresenting which
                        # reasoning text justified which call in the
                        # persisted Message -- exactly what AgentLoop
                        # appends to transcript_out/SessionStore and
                        # re-sends as history on the next turn. Any text
                        # accumulated so far is flushed into its own
                        # TextBlock, in place, before this tool call is
                        # appended -- text within one uninterrupted run
                        # still merges into a single block (unaffected),
                        # but a tool call between two text runs no longer
                        # gets silently reordered around.
                        if text_acc:
                            blocks.append(TextBlock(text=text_acc))
                            text_acc = ""
                        call_id = part.function_call.id
                        if not call_id:
                            call_id = f"{part.function_call.name}-{_synthetic_call_id}"
                            _synthetic_call_id += 1
                        call = ToolCallBlock(
                            id=call_id,
                            name=part.function_call.name or "",
                            arguments=part.function_call.args or {},
                        )
                        blocks.append(call)
                        yield ToolCallEvent(call=call)
                    elif part.inline_data:
                        if text_acc:
                            blocks.append(TextBlock(text=text_acc))
                            text_acc = ""
                        # image-out: an image-capable Gemini model (e.g.
                        # a "-image" model variant) returns generated
                        # image bytes the same way images are sent IN
                        # (types.Blob with .data/.mime_type) -- ModelCapabilities.
                        # modalities_out's own comment ("v1: {TEXT};
                        # image-out models later") and ContentEvent's own
                        # docstring ("e.g. images from image-out models")
                        # both named this before any adapter actually
                        # produced one; this is that first real producer.
                        image = ImageBlock(
                            media_type=part.inline_data.mime_type,
                            data=part.inline_data.data,
                        )
                        blocks.append(image)
                        yield ContentEvent(block=image)
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
        except errors.UnknownApiResponseError as e:
            # A real bug found by reading google-genai's own source, the
            # same way as the identical gap fixed in ollama_provider.py
            # and openai_provider.py: `_api_client._load_json_from_response()`
            # wraps a failed `json.loads()` on a streaming response chunk
            # in this exception -- but it's a `ValueError` subclass, NOT
            # an `errors.APIError` subclass like ClientError/ServerError
            # above, so it propagated uncaught. Same "server sent
            # something we can't make sense of" shape as the Ollama
            # streaming-JSON bug, just one layer down inside the SDK.
            yield StreamErrorEvent(code="provider", detail=str(e), retryable=True)
            return

        # Appended, not inserted at the front -- any trailing text (the
        # ordinary case: a plain END_TURN reply, or reasoning text after
        # the last tool call in a turn) belongs after every block that
        # chronologically preceded it, matching the flush-before-append
        # ordering used for text preceding a tool call/image above.
        if text_acc:
            blocks.append(TextBlock(text=text_acc))

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
