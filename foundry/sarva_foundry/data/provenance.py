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

import json
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


def load_text_files_from_manifest(
    manifest_path: Path, encoding: str = "utf-8"
) -> list[SourcedDocument]:
    """Load exactly the files a provenance manifest names, each with its
    own license — the real per-file license variation
    `load_text_files_with_provenance`'s docstring named as needing a
    manifest, not covered by that function's single uniform license.

    The manifest is a JSON object mapping each file's path (relative to
    the *manifest's own directory*, so the manifest travels with its
    corpus without needing path edits) to that file's license string:

        {"articles/a.txt": "CC-BY-4.0", "books/b.txt": "public-domain"}

    Raises clearly — the same "loud, fixable, not silently wrong"
    principle as `load_text_files` — on a malformed manifest, a missing
    file, or a manifest entry that resolves outside the manifest's own
    directory (guards against path traversal, e.g. `"../../etc/passwd"`
    or an absolute path — `Path("/safe") / "/etc/passwd"` silently
    discards the base and evaluates to `/etc/passwd` alone, a well-known
    pathlib gotcha this check catches by validating the final resolved
    path rather than the raw string)."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"manifest must be a JSON object mapping path -> license, got {type(raw).__name__}"
        )

    base_dir = manifest_path.parent.resolve()
    docs: list[SourcedDocument] = []
    for rel_path, license in raw.items():
        file_path = (base_dir / rel_path).resolve()
        if base_dir not in file_path.parents:
            raise ValueError(f"manifest entry {rel_path!r} resolves outside {base_dir}")
        if not file_path.is_file():
            raise ValueError(f"manifest names {rel_path!r} but no such file exists at {file_path}")
        docs.append(
            SourcedDocument(
                text=file_path.read_text(encoding=encoding),
                source_path=str(file_path),
                license=license,
            )
        )
    return docs


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
