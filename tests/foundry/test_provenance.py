"""Conformance tests for sarva_foundry.data.provenance — source/license
tracking through the corpus pipeline."""

from __future__ import annotations

import json

import pytest
from sarva_foundry.data import (
    SourcedDocument,
    dedup_near_duplicate_sourced_documents,
    dedup_sourced_documents,
    filter_sourced_documents_by_length,
    load_text_files_from_manifest,
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


def _write_manifest(tmp_path, mapping: dict) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(mapping))


def test_load_from_manifest_assigns_a_distinct_license_per_file(tmp_path):
    (tmp_path / "a.txt").write_text("first document")
    (tmp_path / "b.txt").write_text("second document")
    _write_manifest(tmp_path, {"a.txt": "CC-BY-4.0", "b.txt": "public-domain"})

    docs = load_text_files_from_manifest(tmp_path / "manifest.json")

    by_license = {d.license: d.text for d in docs}
    assert by_license["CC-BY-4.0"] == "first document"
    assert by_license["public-domain"] == "second document"


def test_load_from_manifest_resolves_paths_relative_to_the_manifest_directory(tmp_path):
    (tmp_path / "articles").mkdir()
    (tmp_path / "articles" / "a.txt").write_text("content")
    _write_manifest(tmp_path, {"articles/a.txt": "MIT"})

    docs = load_text_files_from_manifest(tmp_path / "manifest.json")

    assert len(docs) == 1
    assert docs[0].source_path.endswith("articles/a.txt")
    assert docs[0].license == "MIT"


def test_load_from_manifest_raises_on_a_missing_file(tmp_path):
    _write_manifest(tmp_path, {"does_not_exist.txt": "MIT"})
    with pytest.raises(ValueError, match="no such file"):
        load_text_files_from_manifest(tmp_path / "manifest.json")


def test_load_from_manifest_raises_when_manifest_is_not_a_json_object(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_text_files_from_manifest(tmp_path / "manifest.json")


def test_load_from_manifest_rejects_path_traversal_outside_manifest_directory(tmp_path):
    _write_manifest(tmp_path, {"../../etc/passwd": "MIT"})
    with pytest.raises(ValueError, match="resolves outside"):
        load_text_files_from_manifest(tmp_path / "manifest.json")


def test_load_from_manifest_rejects_an_absolute_path_entry(tmp_path):
    # Path("/safe/dir") / "/etc/passwd" silently discards the base and
    # evaluates to "/etc/passwd" alone -- a well-known pathlib gotcha.
    # The traversal check must catch this via the final resolved path,
    # not just by looking for ".." in the raw string.
    _write_manifest(tmp_path, {"/etc/passwd": "MIT"})
    with pytest.raises(ValueError, match="resolves outside"):
        load_text_files_from_manifest(tmp_path / "manifest.json")


def test_load_from_manifest_composes_with_dedup_and_filter(tmp_path):
    (tmp_path / "a.txt").write_text("the quick brown fox jumps over the lazy dog")
    (tmp_path / "b.txt").write_text("the quick brown fox jumps over the lazy dog")  # exact dup
    (tmp_path / "c.txt").write_text("hi")  # too short
    _write_manifest(
        tmp_path, {"a.txt": "CC-BY-4.0", "b.txt": "CC-BY-4.0", "c.txt": "public-domain"}
    )

    docs = load_text_files_from_manifest(tmp_path / "manifest.json")
    docs = dedup_sourced_documents(docs)
    docs = filter_sourced_documents_by_length(docs, min_chars=10)

    assert len(docs) == 1
    assert docs[0].license == "CC-BY-4.0"
