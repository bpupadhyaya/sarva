"""sarva.memory.vector — a local, dependency-light semantic memory
store: SQLite for storage, TF-IDF + cosine similarity for retrieval.
This is exactly what `sarva.memory`'s own module docstring names as
future work ("a vector index or database-backed store can layer on top
later without changing this contract") — layered on top, `session.py`
completely untouched.

Not neural embeddings. A real embedding pipeline needs a live
embedding-model API — this project has no configured embeddings
provider (Sarva's provider-agnostic design means this store shouldn't
hard-code one either), so building against one now would be unverifiable
without credentials this environment doesn't have, the same trap a web-
search tool would fall into. TF-IDF is the honest, fully local, fully
testable first tier instead: a genuine vector representation (a sparse
weighted term-frequency vector, not a dense neural one) scored with a
genuine similarity metric — cosine similarity, precisely what dense-
embedding retrieval uses too — built from classical information
retrieval rather than a neural encoder. A real embedding-provider-backed
store can slot in alongside this later without changing the storage
contract, matching the module docstring's own framing.

Deliberately not `sqlite-vec` (the design doc's tech-stack table names
it): that extension indexes *dense* vectors for approximate
nearest-neighbor search at scale. These are sparse, per-query-computed
TF-IDF vectors scored exactly (no ANN index needed at this project's
memory-store scale) — plain SQLite for storage, plain Python for the
scoring math.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MEMORY_DB_PATH = Path.home() / ".sarva" / "memory.db"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    return {term: count * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common_terms = vec_a.keys() & vec_b.keys()
    dot = sum(vec_a[t] * vec_b[t] for t in common_terms)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class MemoryEntry:
    id: int
    session_id: str
    text: str


class VectorMemoryStore:
    """A SQLite-backed store of text entries, searchable by TF-IDF +
    cosine similarity. `db_path` is a real file — the same "just a file
    you can inspect" philosophy as `sarva.memory.session`'s JSON files,
    here via `sqlite3 db_path` instead of `cat`."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add(self, session_id: str, text: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO entries (session_id, text) VALUES (?, ?)", (session_id, text)
        )
        self._conn.commit()
        return cursor.lastrowid

    def search(
        self, query: str, top_k: int = 5, session_id: str | None = None
    ) -> list[tuple[MemoryEntry, float]]:
        """Return up to `top_k` entries most similar to `query`, ranked
        by cosine similarity, highest first. IDF is computed over
        exactly the candidate set being searched (all entries, or just
        `session_id`'s if given) — a query scoped to one session is
        scored against that session's own term statistics, not polluted
        by unrelated sessions' vocabulary."""
        if session_id is not None:
            rows = self._conn.execute(
                "SELECT id, session_id, text FROM entries WHERE session_id = ?", (session_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT id, session_id, text FROM entries").fetchall()

        if not rows:
            return []

        documents = [_tokenize(text) for (_, _, text) in rows]
        doc_freq: Counter[str] = Counter()
        for doc in documents:
            doc_freq.update(set(doc))
        n_docs = len(documents)
        idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in doc_freq.items()}

        query_vec = _tfidf_vector(_tokenize(query), idf)

        scored = []
        for (row_id, row_session_id, text), doc_tokens in zip(rows, documents, strict=True):
            doc_vec = _tfidf_vector(doc_tokens, idf)
            score = _cosine_similarity(query_vec, doc_vec)
            scored.append((MemoryEntry(id=row_id, session_id=row_session_id, text=text), score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def close(self) -> None:
        self._conn.close()
