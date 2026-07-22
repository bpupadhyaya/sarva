"""sarva_foundry.data.provenance — source and license tracking through
the corpus pipeline (spec §3.6c: "each recipe documented with provenance
and license notes"). `sarva_foundry.data.corpus`/`near_dedup`'s
functions operate on plain `list[str]` and stay that way — untouched,
already tested, and simplest for callers who don't need tracking. This
module is a thin layer for callers who do: `SourcedDocument` carries a
document's source path and license through the same load → dedup →
filter → near-dedup stages, by calling the exact same generic, tested
`_dedup_by_key`/`_filter_by_length_key`/`_dedup_near_duplicates_by_key`
helpers those modules use internally — not a reimplementation, and not a
fragile "run the string pipeline, then guess which output belongs to
which input" reconstruction, which breaks the moment two different
source files happen to contain identical text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sarva_foundry.data.corpus import _dedup_by_key, _filter_by_length_key
from sarva_foundry.data.near_dedup import _dedup_near_duplicates_by_key


@dataclass(frozen=True)
class SourcedDocument:
    text: str
    source_path: str
    license: str | None = None


def load_text_files_with_provenance(
    directory: Path,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    license: str | None = None,
) -> list[SourcedDocument]:
    """Like `load_text_files`, but keeps each document's source path
    attached, plus an optional `license` applied uniformly to every file
    this call loads. Real per-file license variation within one
    directory needs a manifest (e.g. a sidecar JSON/CSV mapping path ->
    license) — not implemented here; call this once per license-uniform
    directory in the meantime, which is what most real corpora with a
    handful of distinct sources actually look like."""
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise ValueError(f"no files matching {pattern!r} found under {directory}")
    return [
        SourcedDocument(text=p.read_text(encoding=encoding), source_path=str(p), license=license)
        for p in paths
    ]


def dedup_sourced_documents(docs: list[SourcedDocument]) -> list[SourcedDocument]:
    """Exact-duplicate removal, provenance preserved — first occurrence's
    source_path/license wins for byte-identical text, mirroring
    `dedup_documents`'s own first-occurrence-wins rule exactly (same
    underlying helper, keyed on `.text` instead of the string itself)."""
    return _dedup_by_key(docs, key=lambda d: d.text)


def filter_sourced_documents_by_length(
    docs: list[SourcedDocument], min_chars: int = 1, max_chars: int | None = None
) -> list[SourcedDocument]:
    return _filter_by_length_key(
        docs, key=lambda d: d.text, min_chars=min_chars, max_chars=max_chars
    )


def dedup_near_duplicate_sourced_documents(
    docs: list[SourcedDocument],
    threshold: float = 0.8,
    num_hashes: int = 128,
    shingle_size: int = 5,
) -> list[SourcedDocument]:
    return _dedup_near_duplicates_by_key(
        docs,
        key=lambda d: d.text,
        threshold=threshold,
        num_hashes=num_hashes,
        shingle_size=shingle_size,
    )
