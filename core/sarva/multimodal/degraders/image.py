"""sarva.multimodal.degraders.image — the degradation registry's first
real converter (`Degrader` in content.py had a proven, tested framework
since T0 but zero concrete implementations until now).

`ImageToTextDegrader` turns an `ImageBlock` a text-only model can't
consume into a `TextBlock`. What it deliberately does NOT do: describe
the image's actual visual content. Doing that would require a
vision-capable model call — a decision for the router/agent loop to make
explicitly (send it to a vision model, or don't), not something a
"degrade this content" converter should do as an implicit side effect
buried inside content-model plumbing. Instead this degrader reports only
objectively verifiable metadata decoded directly from the image bytes
(dimensions, format, size), which keeps the design principle "content is
never silently dropped" honest: the target model learns an image was
present and what it technically was, with no fabricated description of
what it contains.
"""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from sarva.multimodal.content import ImageBlock, Modality, TextBlock
from sarva.multimodal.fetch import resolve_media_bytes


class ImageDecodeError(Exception):
    """The image's bytes could not be decoded well enough to degrade it."""


class ImageToTextDegrader:
    source = Modality.IMAGE

    async def degrade(self, block: ImageBlock) -> list[TextBlock]:
        # resolve_media_bytes (not block.resolve_bytes()) so this degrader
        # also handles url-sourced images, not just data/path.
        raw = await resolve_media_bytes(block)
        try:
            with Image.open(io.BytesIO(raw)) as img:
                width, height = img.size
                image_format = img.format or block.media_type
        except UnidentifiedImageError as e:
            raise ImageDecodeError(f"could not decode image for degradation: {e}") from e

        size_kb = len(raw) / 1024
        text = (
            f"[Image attached: {width}x{height} {image_format}, ~{size_kb:.0f}KB. "
            "The current model does not support image input, so its visual "
            "content could not be described.]"
        )
        return [TextBlock(text=text)]
