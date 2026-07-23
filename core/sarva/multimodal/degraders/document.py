"""sarva.multimodal.degraders.document — DocumentToTextDegrader, the
fourth degrader closing the last remaining named modality gap:
`DocumentBlock` has existed since T0 and `models.yaml` even marks
`claude-opus-4-8` as accepting document input, but there was never a
degrader for it — confirmed empty by grep before starting this. A
`DocumentBlock` sent toward a text-only model raised
`UnsupportedModalityError` with no fallback path at all, unlike every
other modality, which is exactly the gap this closes.

Same honesty principle as the other three degraders: real extracted
text where a real extractor exists, never a fabricated summary.
`pypdf` (pure Python, MIT-licensed) is the new commodity-substrate
dependency for PDF text extraction — the same tier as Pillow for images
and PyAV for video, not a "black box" in the sense this project's "no
black boxes" principle (§2.9) actually means. Plain-text-adjacent media
types (`text/plain`, `text/markdown`, `text/csv`, `text/html`,
`application/json`) need no library at all — a straight UTF-8 decode of
the block's own bytes IS the real content.

Honestly scoped, not silently assumed comprehensive: `.docx` and other
binary office formats have no extractor here yet — a heavier dependency
(e.g. `python-docx`) isn't justified by a single format the same way
`pypdf` is justified by PDF being ubiquitous, so `.docx` (and anything
else unrecognized) falls back to the same declared-metadata-only report
the audio/video degraders use for their own undecodable cases — a real,
named, deferred gap, not an implicit one.
"""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from sarva.multimodal.content import DocumentBlock, Modality, TextBlock
from sarva.multimodal.fetch import resolve_media_bytes

_PLAIN_TEXT_MEDIA_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "application/json",
}

# Bounds the degraded text the same way the corpus pipeline's length
# filters bound a single document's size -- an attached 300-page PDF
# shouldn't blow the target model's context window on its own.
_MAX_EXTRACTED_CHARS = 20_000


def _extract_pdf_text(raw: bytes) -> str | None:
    """Real per-page text extraction via `pypdf`, or `None` if the bytes
    aren't a readable PDF at all, OR if they are but every page's text
    layer is empty -- the common real case of a scanned/image-only PDF
    with no embedded text, which is the same "nothing to extract" outcome
    as a read error from this degrader's point of view, not a bug."""
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError):
        return None
    text = "\n\n".join(p for p in pages if p)
    return text or None


def _truncate(text: str) -> tuple[str, int]:
    """Returns `(possibly-truncated text, original length)` -- the
    original length is kept so the degraded message can honestly report
    how much was cut, not just that it was."""
    if len(text) <= _MAX_EXTRACTED_CHARS:
        return text, len(text)
    return text[:_MAX_EXTRACTED_CHARS], len(text)


class DocumentToTextDegrader:
    source = Modality.DOCUMENT

    async def degrade(self, block: DocumentBlock) -> list[TextBlock]:
        raw = await resolve_media_bytes(block)
        size_kb = len(raw) / 1024
        title_part = f" titled {block.title!r}" if block.title else ""

        extracted: str | None = None
        if block.media_type == "application/pdf":
            extracted = _extract_pdf_text(raw)
        elif block.media_type in _PLAIN_TEXT_MEDIA_TYPES:
            try:
                extracted = raw.decode("utf-8")
            except UnicodeDecodeError:
                extracted = None

        if extracted is None:
            text = (
                f"[Document attached{title_part}: {block.media_type}, ~{size_kb:.0f}KB. "
                "Its text could not be extracted (an unsupported format, or a "
                "scanned/image-only document with no text layer), so its content "
                "could not be read.]"
            )
            return [TextBlock(text=text)]

        body, original_len = _truncate(extracted)
        truncated_note = (
            f" [truncated to {_MAX_EXTRACTED_CHARS:,} of {original_len:,} characters]"
            if original_len > len(body)
            else ""
        )
        text = (
            f"[Document attached{title_part}: {block.media_type}, ~{size_kb:.0f}KB. "
            f"Extracted text follows{truncated_note}:]\n\n{body}"
        )
        return [TextBlock(text=text)]
