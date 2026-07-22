"""Conformance tests for sarva_foundry.data.provenance — source/license
tracking through the corpus pipeline."""

from __future__ import annotations

import pytest
from sarva_foundry.data import (
    SourcedDocument,
    dedup_near_duplicate_sourced_documents,
    dedup_sourced_documents,
    filter_sourced_documents_by_length,
    load_text_files_with_provenance,
)


def test_load_text_files_with_provenance_attaches_source_path(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")

    docs = load_text_files_with_provenance(tmp_path)

    assert [d.text for d in docs] == ["hello", "world"]
    assert all(d.source_path.endswith(".txt") for d in docs)
    assert docs[0].source_path != docs[1].source_path


def test_load_text_files_with_provenance_applies_a_uniform_license(tmp_path):
    (tmp_path / "a.txt").write_text("hello")

    docs = load_text_files_with_provenance(tmp_path, license="CC-BY-4.0")

    assert docs[0].license == "CC-BY-4.0"


def test_load_text_files_with_provenance_defaults_license_to_none(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    docs = load_text_files_with_provenance(tmp_path)
    assert docs[0].license is None


def test_load_text_files_with_provenance_raises_on_no_matches(tmp_path):
    with pytest.raises(ValueError, match="no files matching"):
        load_text_files_with_provenance(tmp_path)


def test_dedup_sourced_documents_keeps_first_occurrence_provenance():
    # Two different source files, byte-identical text: the SECOND file's
    # content must be dropped, but the FIRST file's source_path is what
    # survives -- exactly dedup_documents' first-occurrence-wins rule,
    # now verified to carry the correct provenance with it, not just the
    # correct text.
    docs = [
        SourcedDocument(text="same content", source_path="first.txt", license="MIT"),
        SourcedDocument(text="same content", source_path="second.txt", license="MIT"),
        SourcedDocument(text="different", source_path="third.txt", license="MIT"),
    ]

    result = dedup_sourced_documents(docs)

    assert len(result) == 2
    assert result[0].source_path == "first.txt"
    assert result[1].source_path == "third.txt"


def test_filter_sourced_documents_by_length_preserves_provenance():
    docs = [
        SourcedDocument(text="hi", source_path="short.txt"),
        SourcedDocument(text="a longer document", source_path="long.txt"),
    ]

    result = filter_sourced_documents_by_length(docs, min_chars=5)

    assert len(result) == 1
    assert result[0].source_path == "long.txt"


def test_dedup_near_duplicate_sourced_documents_preserves_provenance():
    base = (
        "The quick brown fox jumps over the lazy dog near the old wooden "
        "fence at the edge of the sleepy little village."
    )
    near_dup = base.replace("sleepy", "quiet")
    docs = [
        SourcedDocument(text=base, source_path="original.txt"),
        SourcedDocument(text=near_dup, source_path="republished.txt"),
    ]

    result = dedup_near_duplicate_sourced_documents(docs, threshold=0.75)

    assert len(result) == 1
    assert result[0].source_path == "original.txt"


def test_full_sourced_pipeline_composes(tmp_path):
    (tmp_path / "a.txt").write_text("the quick brown fox jumps over the lazy dog")
    (tmp_path / "b.txt").write_text("the quick brown fox jumps over the lazy dog")  # exact dup
    (tmp_path / "c.txt").write_text("hi")  # too short

    docs = load_text_files_with_provenance(tmp_path, license="public-domain")
    docs = dedup_sourced_documents(docs)
    docs = filter_sourced_documents_by_length(docs, min_chars=10)
    docs = dedup_near_duplicate_sourced_documents(docs)

    assert len(docs) == 1
    assert docs[0].source_path.endswith("a.txt")
    assert docs[0].license == "public-domain"


def test_sourced_document_is_frozen():
    doc = SourcedDocument(text="x", source_path="x.txt")
    with pytest.raises(AttributeError):
        doc.text = "y"
