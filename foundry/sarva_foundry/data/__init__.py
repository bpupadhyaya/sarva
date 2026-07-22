"""sarva_foundry.data — corpus-to-training-batch plumbing (spec §3.6c)."""

from sarva_foundry.data.corpus import dedup_documents, filter_by_length, load_text_files
from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR, TextChunkDataset, tokenize_corpus
from sarva_foundry.data.near_dedup import dedup_near_duplicates

__all__ = [
    "DOCUMENT_SEPARATOR",
    "TextChunkDataset",
    "dedup_documents",
    "dedup_near_duplicates",
    "filter_by_length",
    "load_text_files",
    "tokenize_corpus",
]
