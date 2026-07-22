"""sarva_foundry.data — corpus-to-training-batch plumbing (spec §3.6c)."""

from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR, TextChunkDataset, tokenize_corpus

__all__ = ["DOCUMENT_SEPARATOR", "TextChunkDataset", "tokenize_corpus"]
