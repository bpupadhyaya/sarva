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
            #
            # A real bug found by a fresh-eyes sweep, the identical gap
            # already found and fixed in the Anthropic/OpenAI/Google
            # adapters, just never applied here: the plain `"".join(...
            # TextBlock)` silently dropped any non-text content --
            # ImageBlock most importantly (a screenshot/image-generation
            # tool's result; ToolResultBlock.content's own type comment
            # already names this as an anticipated shape, not a
            # hypothetical) -- sending Ollama a tool result missing
            # content the caller believes is present, the model
            # answering as if it had never received it. Confirmed live:
            # a ToolResultBlock with a TextBlock and an ImageBlock
            # produced a wire message with the image gone, no trace
            # anywhere. Unlike OpenAI (whose tool-message content field
            # only accepts text), Ollama's own `/api/chat` wire format
            # has no per-tool-result content shape at all -- it's a
            # flat, message-level `images` array, the identical one a
            # top-level `ImageBlock` above already populates, confirmed
            # against a real running server. So a tool-result image
            # genuinely CAN be sent, just via that same message-level
            # array rather than nested inside the tool result's own
            # text -- fixed by extracting it there instead of dropping
            # it. Any other, genuinely untranslatable block type still
            # raises, matching every other adapter's own discipline for
            # this exact case.
            result_text_parts: list[str] = []
            for c in b.content:
                if isinstance(c, TextBlock):
                    result_text_parts.append(c.text)
                elif isinstance(c, ImageBlock):
                    c_bytes = await resolve_media_bytes(c)
                    images.append(base64.standard_b64encode(c_bytes).decode())
                else:
                    raise ValueError(
                        f"OllamaProvider cannot translate a {type(c).__name__!r} content "
                        "block inside a tool result (no wire-format mapping exists for it "
                        "yet)"
                    )
            text_parts.append("".join(result_text_parts))
        elif isinstance(b, ThinkingBlock):
            # A real bug found by a fresh-eyes sweep, the identical gap
            # already closed in the OpenAI/Google adapters, never applied
            # here: this adapter's own generate() produces a ThinkingBlock
            # (inserted into the assistant Message whenever a hybrid-
            # thinking model streams a populated `message.thinking`
            # field, confirmed live against a real running server -- see
            # this file's own thinking-field comment below), but this
            # function -- the one that translates a Message BACK into a
            # wire request for the next turn -- had no case for it at
            # all, falling through to the catch-all raise. Confirmed
            # live: any ordinary multi-turn conversation with a hybrid-
            # thinking Ollama model (this project's own confirmed-live
            # default local model family) crashed the very next turn with
            # `state=FAILED` the moment AgentLoop resent the accumulated
            # history containing the just-produced ThinkingBlock.
            # Deliberately, explicitly dropped -- not silently: Ollama's
            # /api/chat has no documented way to feed a prior turn's
            # reasoning trace back in as request content, the same
            # reasoning OpenAI's/Google's adapters already apply to this
            # exact block type.
            continue
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
    # A real bug found by giving this function its own fresh-eyes sweep:
    # `"".join(text_parts)` with NO separator between entries -- harmless
    # for the overwhelmingly common one-TextBlock message, but AgentLoop
    # builds a `Message(role="user", content=list(results))` with one
    # `ToolResultBlock` per concurrently-dispatched tool call every time
    # a model turn issues more than one (`asyncio.gather` over the whole
    # round, see agent/loop.py), which is ordinary use, not an edge
    # case. Confirmed live: two tool results ("4" and "2") glued into a
    # single wire-format string "42" -- not just losing which result
    # came from which tool (an honest, already-documented limitation of
    # this backend having no dedicated tool-result role at all), but
    # actively fusing two distinct values into a different, misleading
    # one. A newline separator is a strict improvement with no
    # regression for the common single-block case (joining a
    # one-element list is unaffected either way).
    out: dict[str, Any] = {"role": m.role, "content": "\n".join(text_parts)}
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
        # A real bug found by a fresh-eyes sweep, applying the "one
        # sibling has the feature, others silently diverge" lens that
        # already caught round 149's DPO-schedule bug: GenerateConfig.
        # stop_sequences is a shared, provider-agnostic field every
        # adapter is supposed to honor, but only GoogleProvider ever
        # actually translated it -- Anthropic, OpenAI, and Ollama each
        # silently dropped it. Ollama's real /api/chat endpoint
        # genuinely documents an `options.stop` field (a list of
        # strings). Confirmed live: a request with stop_sequences=[
        # "STOP_HERE"] sent no `stop` anywhere in the payload/options.
        #
        # A later fresh-eyes sweep found the identical gap for
        # max_tokens, the sibling field to stop_sequences on this same
        # GenerateConfig: Anthropic/OpenAI/Google all unconditionally
        # send it (as max_tokens/max_completion_tokens/max_output_tokens
        # respectively, every single call), but this adapter never
        # translated it into Ollama's real, documented
        # `options.num_predict` field at all. Ollama's own server-side
        # default for num_predict is -1 (generate until the model stops
        # on its own or the context window is exhausted), so a caller
        # relying on max_tokens to bound response length/cost -- the
        # ordinary, documented way to do so -- got silently no
        # enforcement when routed to Ollama, this project's "free &
        # private" local-model tier and the one most likely to run
        # unattended with no API-cost pressure to notice runaway
        # generation. Confirmed live: max_tokens=16 produced no `options`
        # key in the payload at all.
        options: dict[str, Any] = {"num_predict": request.config.max_tokens}
        if request.config.stop_sequences:
            options["stop"] = request.config.stop_sequences
        payload["options"] = options

        text_acc = ""
        # A real bug found by a much later fresh-eyes sweep, the same
        # "this adapter never translates a real GenerateConfig field"
        # shape already found and fixed twice on this same file
        # (stop_sequences, max_tokens) -- but this one is inbound, not
        # outbound, and deliberately NOT paired with an outbound `think`
        # request field: confirmed live against a real running Ollama
        # server that sending `think: true` to a model that doesn't
        # support it produces a real HTTP 400 ("does not support
        # thinking"), the identical risk OpenAI's/Google's own adapters
        # already cite as their reason for leaving `effort` unmapped --
        # this adapter has no way to know a given model's capabilities
        # at this layer, so requesting thinking unconditionally would be
        # a real regression for every non-thinking Ollama model. But a
        # real hybrid-thinking model (confirmed live: `qwen3:0.6b`)
        # emits a real, populated `message.thinking` field on every
        # streamed chunk BY DEFAULT, with no `think` request field sent
        # at all -- and this adapter never read it, silently discarding
        # the model's entire real reasoning trace, never surfaced as a
        # ThinkingDeltaEvent and never persisted in the transcript's
        # ThinkingBlock, unlike the identical feature already working
        # correctly for AnthropicProvider. This is a total, silent loss
        # of real data the server sends unprompted, not a hypothetical
        # or request-config-dependent gap -- confirmed live streaming a
        # real "what is 2+2?" turn through the real qwen3:0.6b model.
        thinking_acc = ""
        content: list[object] = []
        call_index = 0
        done_reason: StopReason = StopReason.END_TURN
        # A real bug found by a fresh-eyes sweep, the identical
        # "sibling adapter has a real stop/finish-reason mapping, this
        # one never got one" gap already found and fixed in all three
        # OTHER provider adapters (anthropic_provider.py's pause_turn,
        # openai_provider.py's function_call, google_provider.py's
        # RECITATION/etc.): Ollama's real streaming `/api/chat`
        # response includes a `done_reason` field on its final chunk
        # ("stop" for a normal end, "length" when generation was cut
        # off by hitting num_predict/the context window -- documented
        # Ollama wire behavior), but nothing here ever read it --
        # `done_reason` only ever changed away from its END_TURN
        # default when a tool call was seen. Confirmed live with a
        # mocked Ollama stream whose final chunk says `"done_reason":
        # "length"`: this adapter reported `StopReason.END_TURN`
        # anyway, silently indistinguishable from a genuinely complete
        # response -- a caller branching on `StopReason.MAX_TOKENS` to
        # retry with a larger budget or warn the user never gets the
        # chance to for any real Ollama-served conversation that hits
        # its own generation limit. Deliberately narrower than the
        # other three adapters' fixes: Ollama's `done_reason` has no
        # closed, vendor-typed enum to check exhaustively against (`
        # "stop"`/`"length"` are the only two values documented with
        # confidence), so unrecognized/missing values still default to
        # the existing END_TURN behavior rather than a new fail-safe
        # default -- there's no verified evidence an unlisted value
        # ever means anything other than a normal completion.
        raw_done_reason: str | None = None
        # A real bug found by a later fresh-eyes sweep: the comment right
        # below this one previously claimed Ollama's /api/chat "doesn't
        # return token usage" -- factually wrong for the real, current
        # API. The same final chunk this adapter already reads
        # `done_reason` off (see the comment above) also carries
        # `prompt_eval_count` (input tokens) and `eval_count` (output
        # tokens), documented Ollama wire fields, confirmed live against
        # a real running Ollama server. Every other real provider adapter
        # computes real Usage from real response fields; this was the
        # one that hardcoded zero. AgentLoop's own `spend.total_tokens +=
        # done.usage.input_tokens + done.usage.output_tokens` means a
        # hardcoded-zero Usage silently makes Budget.max_total_tokens
        # unenforceable for every Ollama-routed call -- the "free &
        # private" local-model tier this file's own docstring calls a
        # first-class, verified-live supported path, and the one most
        # likely to run unattended for a long session with no API-cost
        # pressure prompting a user to notice.
        prompt_eval_count = 0
        eval_count = 0

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
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as e:
                        # A real bug found by actually feeding a
                        # genuinely truncated NDJSON line through a
                        # mocked transport (the real-world trigger: a
                        # network glitch or proxy cutting the stream
                        # mid-line): the raw JSONDecodeError propagated
                        # uncaught out of this async generator instead
                        # of the clean StreamErrorEvent the
                        # status_code >= 400 branch above already
                        # produces for a malformed *response*. retryable
                        # for the same reason ConnectError/
                        # TimeoutException below are: a transient
                        # network hiccup mid-stream is a reasonable
                        # thing to retry, not a hard failure.
                        yield StreamErrorEvent(
                            code="provider",
                            detail=f"malformed streaming response from Ollama: {e}",
                            retryable=True,
                        )
                        return
                    # A real bug found by a fresh-eyes sweep, one shape the
                    # neighboring JSONDecodeError branch above doesn't
                    # cover: Ollama's real wire protocol has a third line
                    # shape, `{"error": "..."}`, sent when generation fails
                    # *after* streaming has already started (context
                    # canceled, GPU OOM, a crashed backend) -- too late for
                    # the `response.status_code >= 400` pre-flight check
                    # above to ever see it, since the 200 response line and
                    # some real content may already have streamed. This
                    # loop only ever inspected `"message"`/`"done"`, so an
                    # error line was silently ignored: the `async for` just
                    # fell through, `done_reason` stayed defaulted to
                    # END_TURN, and `prompt_eval_count`/`eval_count` stayed
                    # 0 -- every signal said "normal, complete, successful
                    # turn." Confirmed live with a mocked transport
                    # streaming real content then an error line: a
                    # genuinely truncated, wrong answer came back as a
                    # clean DoneEvent with stop_reason=END_TURN and zero
                    # usage, no StreamErrorEvent at all -- AgentLoop treats
                    # END_TURN as success, so the truncated text gets
                    # persisted to the transcript and shown to the user
                    # with no error and no retry offered.
                    if chunk.get("error"):
                        yield StreamErrorEvent(
                            code="provider",
                            detail=f"Ollama reported an error mid-stream: {chunk['error']}",
                            retryable=True,
                        )
                        return
                    msg = chunk.get("message", {})
                    if msg.get("thinking"):
                        thinking_acc += msg["thinking"]
                        yield ThinkingDeltaEvent(text=msg["thinking"])
                    if msg.get("content"):
                        text_acc += msg["content"]
                        yield TextDeltaEvent(text=msg["content"])
                    # A real bug found by a fresh-eyes sweep, the identical
                    # gap already found and fixed in google_provider.py,
                    # just never propagated here: text was accumulated
                    # into ONE running string across the WHOLE stream and
                    # only ever spliced into `content` once, unconditionally
                    # first, after the loop ended -- tool calls were
                    # appended in true chronological order, but text
                    # occurring between or after them got silently pulled
                    # forward and merged, ahead of every tool call, even
                    # ones that chronologically preceded it. Confirmed
                    # live: an ordinary sequential-tool-calling turn
                    # (reasoning text, a call, more reasoning text, another
                    # call -- ordinary ReAct-style behavior for local
                    # tool-using models served via Ollama, not contrived)
                    # produced one TextBlock with both segments concatenated,
                    # hoisted ahead of BOTH tool calls, corrupting the
                    # persisted Message AgentLoop appends straight to
                    # transcript_out/SessionStore and resends as history.
                    # Fixed by flushing any accumulated text into its own
                    # TextBlock, in place, immediately before each tool
                    # call -- text within one uninterrupted run still
                    # merges into a single block, but a tool call between
                    # two text runs no longer gets silently reordered
                    # around.
                    for tc in msg.get("tool_calls", []) or []:
                        if text_acc:
                            content.append(TextBlock(text=text_acc))
                            text_acc = ""
                        fn = tc.get("function", {})
                        call = ToolCallBlock(
                            id=f"ollama-{call_index}",
                            name=fn.get("name", ""),
                            arguments=fn.get("arguments", {}),
                        )
                        call_index += 1
                        content.append(call)
                        yield ToolCallEvent(call=call)
                        done_reason = StopReason.TOOL_USE
                    if chunk.get("done"):
                        raw_done_reason = chunk.get("done_reason")
                        prompt_eval_count = chunk.get("prompt_eval_count") or 0
                        eval_count = chunk.get("eval_count") or 0
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

        # Inserted at the front, unconditionally before every other
        # block: confirmed live against a real streaming Ollama response
        # that `thinking` deltas always precede `content`/`tool_calls`
        # deltas for the whole turn (the model reasons, then answers),
        # so there's exactly one thinking span per turn, always first --
        # unlike `text_acc`/tool calls below, which can genuinely
        # interleave and need their own chronological-order tracking.
        if thinking_acc:
            content.insert(0, ThinkingBlock(text=thinking_acc))

        # Appended, not inserted at the front -- any trailing text (the
        # ordinary case: a plain END_TURN reply, or reasoning text after
        # the last tool call in a turn) belongs after every block that
        # chronologically preceded it, matching the flush-before-append
        # ordering used for text preceding a tool call above.
        if text_acc:
            content.append(TextBlock(text=text_acc))

        # Presence of a tool call always wins over the raw done_reason,
        # same as every sibling adapter's own "a tool call always means
        # TOOL_USE regardless of what the raw finish/stop signal says"
        # rule -- Ollama reports "stop" as the done_reason even for a
        # turn that ended in a tool call, so checking this second would
        # silently misreport it as END_TURN/MAX_TOKENS instead.
        if done_reason != StopReason.TOOL_USE and raw_done_reason == "length":
            done_reason = StopReason.MAX_TOKENS

        yield DoneEvent(
            stop_reason=done_reason,
            message=Message(role="assistant", content=content),
            # cost_usd stays 0.0 -- local inference is genuinely free,
            # unlike input_tokens/output_tokens which Ollama's final
            # chunk does report (see the comment above `raw_done_reason`).
            usage=Usage(input_tokens=prompt_eval_count, output_tokens=eval_count),
        )

    async def close(self) -> None:
        await self._client.aclose()
