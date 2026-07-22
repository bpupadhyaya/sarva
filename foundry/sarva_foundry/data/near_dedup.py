"""sarva_foundry.data.near_dedup — near-duplicate detection via MinHash,
the real, separate scope `dedup_documents`'s own docstring named and
deferred: exact-hash dedup catches byte-identical documents, but two
documents that differ by a sentence, a timestamp, or a scraped ad banner
are common in real corpora and need similarity-based detection instead.

MinHash estimates the Jaccard similarity between two documents' shingle
sets without ever computing the sets' full intersection directly (which
would need to hold every shingle set in memory and compare pairwise —
expensive at real corpus scale). Each document is reduced to a
fixed-size signature (one minimum hash value per hash function), and the
fraction of matching signature positions between two documents is an
unbiased estimator of their true Jaccard similarity — the more hash
functions, the lower the estimator's variance. Implemented entirely from
the underlying hashing, not vendored from an external minhash/datasketch
library: hashlib's SHA-256 (truncated, per-function-salted) stands in
for the "hash function family" a textbook MinHash description assumes,
which is exactly what real implementations do too — the algorithm is
the contribution, not the hash primitive underneath it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

_UINT64_MAX = 2**64 - 1


def _shingles(text: str, size: int) -> set[str]:
    """Character k-shingles — robust to word-boundary differences
    (punctuation, whitespace, minor rewording) that word-level shingling
    would be more sensitive to losing overlap from."""
    if len(text) < size:
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def _minhash_signature(shingles: set[str], num_hashes: int) -> tuple[int, ...]:
    """One minimum hash value per (deterministically salted) hash
    function. An empty shingle set gets a sentinel all-max signature so
    two empty documents compare as identical (correct: two empty strings
    genuinely are duplicates) without a special-cased empty-set branch
    in the similarity/dedup logic downstream."""
    if not shingles:
        return (_UINT64_MAX,) * num_hashes
    signature = []
    for h in range(num_hashes):
        salt = f"minhash:{h}:".encode()
        min_hash = min(
            int.from_bytes(hashlib.sha256(salt + s.encode("utf-8")).digest()[:8], "big")
            for s in shingles
        )
        signature.append(min_hash)
    return tuple(signature)


def _estimated_jaccard_similarity(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
    return matches / len(sig_a)


def _dedup_near_duplicates_by_key[T](
    items: list[T],
    key: Callable[[T], str],
    threshold: float,
    num_hashes: int,
    shingle_size: int,
) -> list[T]:
    kept: list[T] = []
    kept_signatures: list[tuple[int, ...]] = []
    for item in items:
        signature = _minhash_signature(_shingles(key(item), shingle_size), num_hashes)
        is_near_duplicate = any(
            _estimated_jaccard_similarity(signature, kept_sig) >= threshold
            for kept_sig in kept_signatures
        )
        if not is_near_duplicate:
            kept.append(item)
            kept_signatures.append(signature)
    return kept


def dedup_near_duplicates(
    docs: list[str],
    threshold: float = 0.8,
    num_hashes: int = 128,
    shingle_size: int = 5,
) -> list[str]:
    """Drop documents estimated to be near-duplicates (Jaccard similarity
    >= `threshold`) of an earlier-kept document, keeping first-occurrence
    order — the near-duplicate counterpart to `dedup_documents`'s
    exact-hash dedup. Run `dedup_documents` first in a real pipeline:
    it's O(n) and cheap, shrinking the corpus before this O(kept^2)
    pairwise comparison pass (each new document compared against every
    document kept so far) — fine at the scale this project's own tests
    and examples run at; a web-scale corpus would need an LSH banding
    index on top, not implemented here."""
    return _dedup_near_duplicates_by_key(
        docs, key=lambda d: d, threshold=threshold, num_hashes=num_hashes, shingle_size=shingle_size
    )
