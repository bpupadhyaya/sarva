"""A minimal language-model dataset (spec §3.6c, the small end of it):
concatenate a corpus of documents into one token stream, separated by a
document-boundary special token, then slice it into fixed-length chunks
for next-token-prediction training.

This is the "concatenate and chunk" approach every real pretraining
pipeline uses at some layer (it avoids wasting compute on padding), not a
simplified stand-in for it — what's missing relative to the full §3.6c
scope is corpus *sourcing* (web/code/books/math crawling, cleaning,
dedup, quality filtering, mixing recipes), not this chunking mechanism
itself.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from sarva_foundry.tokenizer import ByteLevelBPETokenizer

DOCUMENT_SEPARATOR = "<|endoftext|>"


def tokenize_corpus(
    texts: Iterable[str],
    tokenizer: ByteLevelBPETokenizer,
    document_separator: str = DOCUMENT_SEPARATOR,
) -> list[int]:
    """Encode every document and concatenate, inserting `document_separator`
    between them so the model can learn document boundaries instead of
    treating the corpus as one continuous, unrelated stream. Requires
    `tokenizer` to have been trained with `document_separator` in its
    `special_tokens` (raises a clear `KeyError` via the tokenizer itself
    otherwise, rather than silently encoding it as ordinary bytes)."""
    if document_separator not in tokenizer.special_tokens:
        raise ValueError(
            f"tokenizer was not trained with {document_separator!r} as a special token"
        )
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        ids.extend(tokenizer.encode(document_separator))
    return ids


class TextChunkDataset(Dataset):
    """Fixed-length `(input, target)` chunks from one token stream, where
    `target` is `input` shifted right by one (standard next-token-prediction
    framing). The final `(len(token_ids) - 1) % seq_len` leftover tokens
    that don't fill a whole chunk are dropped, not padded — documented
    behavior, not a silent truncation (see `test_dataset.py`)."""

    def __init__(self, token_ids: list[int], seq_len: int):
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if len(token_ids) < seq_len + 1:
            raise ValueError(
                f"need at least seq_len + 1 = {seq_len + 1} tokens for one chunk, "
                f"got {len(token_ids)}"
            )
        self.seq_len = seq_len
        usable_len = (len(token_ids) - 1) // seq_len * seq_len + 1
        self.token_ids = token_ids[:usable_len]

    def __len__(self) -> int:
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.seq_len
        chunk = self.token_ids[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
