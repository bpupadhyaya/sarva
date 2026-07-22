"""sarva_foundry.data.corpus — local corpus sourcing: load, dedup, filter
(spec §3.6c). Not the full pipeline that section describes (Common
Crawl-scale sourcing, license-aware filtering, synthetic-data
generation) — these are three sourcing/cleaning stages every larger
pipeline still needs at its base, implemented at the scale this project
can actually run and test today: a local directory of text files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def load_text_files(directory: Path, pattern: str = "*.txt", encoding: str = "utf-8") -> list[str]:
    """Read every file matching `pattern` under `directory` (sorted for
    deterministic ordering) as one document each. Raises on a decode
    error rather than silently skipping or corrupting a document — a bad
    file should be a loud, fixable problem, not quietly missing data no
    one notices until the model trained on it behaves strangely."""
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise ValueError(f"no files matching {pattern!r} found under {directory}")
    return [p.read_text(encoding=encoding) for p in paths]


def dedup_documents(docs: list[str]) -> list[str]:
    """Drop exact-duplicate documents, keeping each one's first
    occurrence and the original relative order. Near-duplicate detection
    (minhash/simhash-based, catching two documents that differ by a
    sentence or a timestamp) is real, separate scope beyond exact-match —
    named here rather than silently assumed covered."""
    seen: set[str] = set()
    out: list[str] = []
    for doc in docs:
        digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            out.append(doc)
    return out


def filter_by_length(
    docs: list[str], min_chars: int = 1, max_chars: int | None = None
) -> list[str]:
    """Drop documents outside `[min_chars, max_chars]` — the crudest real
    quality filter (too-short is usually boilerplate/navigation junk;
    too-long is often scrape garbage or a parsing failure), and the one
    every larger pipeline layers richer heuristics on top of, not a
    replacement for them."""
    out = []
    for doc in docs:
        length = len(doc)
        if length < min_chars:
            continue
        if max_chars is not None and length > max_chars:
            continue
        out.append(doc)
    return out
