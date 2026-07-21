"""Conformance tests for sarva.memory.session — the file-based session store."""

from __future__ import annotations

import pytest
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ImageBlock, Message, TextBlock


@pytest.fixture
def store(tmp_path):
    return SessionStore(root=tmp_path)


def test_load_missing_session_returns_empty(store):
    assert store.load("does-not-exist") == []


def test_save_then_load_round_trips(store):
    messages = [
        Message(role="user", content=[TextBlock(text="hi")]),
        Message(role="assistant", content=[TextBlock(text="hello there")]),
    ]
    store.save("greeting", messages)
    assert store.load("greeting") == messages


def test_round_trip_preserves_binary_content(store):
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="what's this?"),
                ImageBlock(media_type="image/png", data=b"\x89PNG\r\n\x1a\n"),
            ],
        )
    ]
    store.save("with-image", messages)
    restored = store.load("with-image")
    assert restored == messages


def test_clear_removes_the_session(store):
    store.save("temp", [Message(role="user", content=[TextBlock(text="x")])])
    assert store.load("temp") != []
    store.clear("temp")
    assert store.load("temp") == []


def test_clear_missing_session_does_not_raise(store):
    store.clear("never-existed")  # must not raise


def test_list_sessions(store):
    assert store.list_sessions() == []
    store.save("alpha", [])
    store.save("beta", [Message(role="user", content=[TextBlock(text="hi")])])
    assert store.list_sessions() == ["alpha", "beta"]


def test_session_name_traversal_is_rejected(store):
    with pytest.raises(ValueError, match="invalid session name"):
        store.load("../../etc/passwd")


def test_session_name_with_invalid_characters_is_rejected(store):
    # Reject rather than silently sanitize — silent stripping risks two
    # distinct names colliding onto the same file (e.g. "my session" and
    # "mysession" both stripping to the same thing).
    with pytest.raises(ValueError, match="invalid session name"):
        store.save("my session!", [])
