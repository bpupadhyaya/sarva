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

import asyncio
import io

from pypdf import PdfReader
from pypdf.errors import LimitReachedError, PdfReadError

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

# A real bug found by a fresh-eyes sweep applying the same lens that
# already gave every OTHER CPU-bound media-processing call in this
# project an explicit ceiling (sarva.audio's _DECODE_TIMEOUT_SECONDS/
# _TTS_TIMEOUT_SECONDS, the video degrader's subprocess timeout):
# `_extract_pdf_text`'s own `asyncio.to_thread` wrapper (below) bounds
# the event-loop-freeze this degrader's own docstring already fixed
# once, but does nothing to bound its WALL-CLOCK time -- pypdf's cyclic-
# page-reference guard and LimitReachedError decompression-bomb guard
# both already fire fast (confirmed live against real crafted PDFs
# exercising each), but neither protects against a legitimately valid,
# just extremely large PDF (a real, ordinary document -- a scanned
# archive or a programmatically generated report can genuinely run to
# tens of thousands of pages), whose full per-page extract_text() this
# degrader always runs to completion BEFORE _truncate (above) ever gets
# a chance to cut the output down. Confirmed live: this degrader's own
# `degrade()` call blocked past a 5-second deadline against a scripted
# slow extraction with no recovery -- the whole agent turn would hang
# indefinitely, since AgentLoop's own Budget.max_wall_seconds check only
# fires BETWEEN await points, never inside one still in flight. Bounded
# the same way every sibling call already is, with one honest caveat
# documented at the call site: unlike the audio/video degraders' real
# subprocess isolation, `asyncio.to_thread`'s underlying OS thread
# cannot actually be killed on timeout -- this stops the AGENT TURN from
# hanging (the real, user-facing symptom), not the abandoned thread
# itself, matching this project's own "an honest partial mitigation,
# not a magic bullet" posture elsewhere (e.g. the subagent-cancellation
# spend-release comment in agent/loop.py).
_PDF_EXTRACT_TIMEOUT_SECONDS = 30


def _extract_pdf_text(raw: bytes) -> str | None:
    """Real per-page text extraction via `pypdf`, or `None` if the bytes
    aren't a readable PDF at all, OR if they are but every page's text
    layer is empty -- the common real case of a scanned/image-only PDF
    with no embedded text, which is the same "nothing to extract" outcome
    as a read error from this degrader's point of view, not a bug."""
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, LimitReachedError):
        # A real bug found by actually building a PDF whose /Contents
        # stream is a FlateDecode zlib bomb (a small, highly-compressible
        # payload that decompresses to 100MB): pypdf's own internal
        # decompression-bomb guard (ZLIB_MAX_OUTPUT_LENGTH) raises
        # LimitReachedError, but that's a direct sibling of PdfReadError
        # under PyPdfError, not a subclass of it -- confirmed via the
        # real class MRO, not assumed from the name. The same
        # "tiny-file-declares-something-implausibly-huge" DoS shape
        # already fixed for ImageToTextDegrader's DecompressionBombError.
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
            # A real bug found by giving this degrader the same
            # event-loop-freeze lens that had just found AudioToTextDegrader
            # blocking the whole process (see sarva.multimodal.degraders.
            # audio): pypdf's own parsing + per-page extract_text() is
            # synchronous, CPU-bound work called directly here with no
            # `asyncio.to_thread`. Confirmed live: a realistic 300-page
            # PDF (this module's own docstring above already treats 300
            # pages as a plausible real attachment size, not an extreme)
            # took 0.52s of real wall-clock extraction time -- during
            # which every other concurrent request in a real `sarva
            # serve` process would have been frozen, the same shape as
            # the audio bug, just a smaller and more variable magnitude
            # depending on document size.
            try:
                extracted = await asyncio.wait_for(
                    asyncio.to_thread(_extract_pdf_text, raw),
                    timeout=_PDF_EXTRACT_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                # See _PDF_EXTRACT_TIMEOUT_SECONDS' own comment -- treated
                # the same as "couldn't extract text" below, the same
                # honest fallback an unsupported format or a corrupt PDF
                # already gets.
                extracted = None
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
