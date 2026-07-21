"""Conformance tests for the FastAPI server — REST + WebSocket over the
agent loop. Uses FastAPI's in-process TestClient — no real network, no
running server process."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sarva.memory import session as session_module
from sarva.memory.session import SessionStore
from sarva.server.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_models_lists_mock_as_available():
    resp = _client().get("/models")
    assert resp.status_code == 200
    models = resp.json()
    mock = next(m for m in models if m["id"] == "mock")
    assert mock["available"] is True


def test_chat_zero_config_uses_mock():
    resp = _client().post("/chat", json={"message": "hello server"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "done"
    assert "hello server" in body["message"]
    assert "spend" in body


def test_chat_with_session_persists_across_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path)

    client = _client()
    r1 = client.post("/chat", json={"message": "first", "session": "web-test"})
    r2 = client.post("/chat", json={"message": "second", "session": "web-test"})
    assert r1.status_code == 200
    assert r2.status_code == 200

    store = SessionStore()
    assert len(store.load("web-test")) == 4  # 2 turns * (user + assistant)


def test_chat_without_session_does_not_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path)

    _client().post("/chat", json={"message": "no memory please"})

    assert SessionStore().list_sessions() == []


def test_websocket_streams_events_and_ends_with_run_done():
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi via websocket"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    assert events[-1]["type"] == "run_done"
    assert events[-1]["state"] == "done"
    assert any(e["type"] == "model_stream" for e in events)


def test_websocket_with_session_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path)
    client = _client()

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "remember me", "session": "ws-test"})
        while ws.receive_json()["type"] != "run_done":
            pass

    assert len(SessionStore().load("ws-test")) == 2
