"""Hermetic tests for image-out: `google_provider.py` translating a
Gemini response part's `inline_data` into a `ContentEvent(block=
ImageBlock(...))`. `ModelCapabilities.modalities_out` and `ContentEvent`
have both named "image-out models" as anticipated future work since T1
(`modalities_out`'s own comment: "v1: {TEXT}; image-out models later";
`ContentEvent`'s own docstring: "e.g. images from image-out models") —
until now no adapter ever actually produced one, confirmed by `grep -rn
"ContentEvent" core/sarva` returning only the type's own definition and
its use in the discriminated `ProviderEvent` union.

Uses the same duck-typed `SimpleNamespace` stand-ins as
test_google_provider_streaming.py rather than real SDK response types --
this test's job is proving our own translation is correct, not
re-verifying the SDK's own wire parsing. No network, no API key."""

from __future__ import annotations

from types import SimpleNamespace

from sarva.multimodal.content import ImageBlock, Message, TextBlock
from sarva.providers.base import ContentEvent, DoneEvent, GenerateRequest
from sarva.providers.google_provider import GoogleProvider


def _inline_data(mime_type, data):
    return SimpleNamespace(mime_type=mime_type, data=data)


def _part(text=None, thought=False, function_call=None, inline_data=None):
    return SimpleNamespace(
        text=text, thought=thought, function_call=function_call, inline_data=inline_data
    )


def _chunk(parts=None, finish_reason=None, usage=None):
    content = SimpleNamespace(parts=parts) if parts else None
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _usage(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        cached_content_token_count=0,
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _FakeClient:
    def __init__(self, chunks):
        async def generate_content_stream(**kwargs):
            return _FakeStream(chunks)

        models = SimpleNamespace(generate_content_stream=generate_content_stream)
        self.aio = SimpleNamespace(models=models)


def _simple_request(model: str = "gemini-image-x") -> GenerateRequest:
    return GenerateRequest(
        model=model, messages=[Message(role="user", content=[TextBlock(text="draw a cat")])]
    )


async def test_inline_data_part_becomes_a_content_event_with_an_image_block():
    raw = b"\x89PNG\r\n\x1a\n fake png bytes"
    chunks = [
        _chunk(
            parts=[_part(inline_data=_inline_data("image/png", raw))],
            finish_reason="STOP",
            usage=_usage(10, 5),
        ),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1
    image = content_events[0].block
    assert isinstance(image, ImageBlock)
    assert image.media_type == "image/png"
    assert image.data == raw


async def test_generated_image_ends_up_in_the_final_assistant_message():
    raw = b"fake jpeg bytes"
    chunks = [
        _chunk(
            parts=[_part(inline_data=_inline_data("image/jpeg", raw))],
            finish_reason="STOP",
            usage=_usage(3, 2),
        ),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    images = [b for b in done.message.content if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert images[0].data == raw


async def test_text_and_image_in_the_same_response_both_survive():
    # A real model turn can mix a text caption with a generated image --
    # proves the two paths (text_acc accumulation, per-part image
    # translation) don't clobber each other.
    chunks = [
        _chunk(parts=[_part(text="Here's your cat:")]),
        _chunk(
            parts=[_part(inline_data=_inline_data("image/png", b"cat bytes"))],
            finish_reason="STOP",
            usage=_usage(8, 4),
        ),
    ]
    provider = GoogleProvider(client=_FakeClient(chunks))
    events = [e async for e in provider.generate(_simple_request())]

    done = events[-1]
    texts = [b.text for b in done.message.content if isinstance(b, TextBlock)]
    images = [b for b in done.message.content if isinstance(b, ImageBlock)]
    assert texts == ["Here's your cat:"]
    assert len(images) == 1
    assert images[0].data == b"cat bytes"
