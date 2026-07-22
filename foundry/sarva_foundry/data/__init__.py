"""sarva_foundry.data — corpus-to-training-batch plumbing (spec §3.6c)."""

from sarva_foundry.data.corpus import dedup_documents, filter_by_length, load_text_files
from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR, TextChunkDataset, tokenize_corpus
from sarva_foundry.data.near_dedup import dedup_near_duplicates
from sarva_foundry.data.provenance import (
    SourcedDocument,
    dedup_near_duplicate_sourced_documents,
    dedup_sourced_documents,
    filter_sourced_documents_by_length,
    load_text_files_with_provenance,
)

__all__ = [
    "DOCUMENT_SEPARATOR",
    "SourcedDocument",
    "TextChunkDataset",
    "dedup_documents",
    "dedup_near_duplicate_sourced_documents",
    "dedup_near_duplicates",
    "dedup_sourced_documents",
    "filter_by_length",
    "filter_sourced_documents_by_length",
    "load_text_files",
    "load_text_files_with_provenance",
    "tokenize_corpus",
]
