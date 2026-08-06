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
    # A real bug found by a fresh-eyes sweep, the identical "unvalidated
    # numeric parameter used as a range() bound/divisor" shape round 141
    # already fixed for ablation.py's record_every: num_hashes is a
    # plain, unvalidated public int kwarg on both entry points
    # (dedup_near_duplicates, dedup_near_duplicate_sourced_documents).
    # num_hashes=0 makes every document's MinHash signature an empty
    # tuple regardless of content -- the first document is always kept
    # (any() over an empty kept_signatures list is vacuously False), but
    # the second document processed hits _estimated_jaccard_similarity's
    # `matches / len(sig_a)`, a raw, undocumented ZeroDivisionError with
    # zero indication of the real cause. A caller tuning num_hashes down
    # for a fast smoke-test run, or computing it programmatically from a
    # compute/time budget, can land on 0 with nothing to catch it.
    # Confirmed live at both public entry points. Checked once here,
    # the single choke point both funnel through, rather than
    # duplicating the check in each public wrapper.
    if num_hashes <= 0:
        raise ValueError(f"num_hashes must be positive, got {num_hashes}")
    # A real bug found by a later fresh-eyes sweep, the sibling
    # parameter one over from num_hashes above, in this exact function:
    # shingle_size was likewise never validated. _shingles' own `len(
    # text) < size` guard is always False for size <= 0, so every
    # comprehension iteration slices `text[i:i+0]`, which is always "" --
    # every non-empty document in the corpus collapses to the identical
    # single-element shingle set {""}, regardless of actual content.
    # Their MinHash signatures become bit-identical, so every pair
    # scores similarity 1.0 and every document after the first is
    # silently treated as a near-duplicate and dropped -- pure silent
    # data loss in a corpus-cleaning pipeline, with no crash, no
    # warning, and no record that a collision occurred. Confirmed live:
    # three genuinely unrelated real documents collapsed from 3 kept
    # down to 1 with shingle_size=0.
    if shingle_size <= 0:
        raise ValueError(f"shingle_size must be positive, got {shingle_size}")
    # A real bug found by a later fresh-eyes sweep, the third sibling
    # parameter in this exact function to get this treatment: threshold
    # was likewise never validated. _estimated_jaccard_similarity always
    # returns a value in [0.0, 1.0], so any threshold <= 0.0 makes
    # `similarity >= threshold` true for every single pairwise
    # comparison -- every document after the first is unconditionally
    # treated as a near-duplicate of the first and silently dropped,
    # collapsing an entire, genuinely diverse corpus down to one
    # document. Confirmed live: four topically unrelated real sentences
    # collapsed from 4 kept down to 1 the instant threshold was 0.0 or
    # negative. A caller deriving threshold programmatically (e.g.
    # `1 - estimated_noise_level`, where the noise estimate saturates at
    # or above 1.0) can land on exactly this with no adversarial intent.
    # A threshold above 1.0 is comparatively harmless (dedup becomes a
    # no-op, no data lost) but is rejected too since Jaccard similarity
    # is only ever in [0.0, 1.0] -- such a threshold could never match
    # anything, and silently doing nothing is itself worth surfacing.
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold}")
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
