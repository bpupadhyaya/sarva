"""Conformance tests for the FastAPI server — REST + WebSocket over the
agent loop. Uses FastAPI's in-process TestClient — no real network, no
running server process."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sarva.memory import session as session_module
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ToolCallBlock
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, load_routing
from sarva.server import app as app_module
from sarva.server.app import create_app

_DATA_DIR = Path(__file__).parent.parent.parent / "core" / "sarva" / "providers" / "data"


def _client() -> TestClient:
    return TestClient(create_app())


def _mock_only_router() -> Router:
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    return Router(registry, routing, available={"mock"})


def _use_scripted_mock(monkeypatch, script: list[ScriptedTurn]) -> MockProvider:
    """Server code imports build_providers/build_router directly into its
    own module namespace (`from sarva.runtime import ...`), so patching
    sarva.runtime doesn't reach it — the patch target must be the names as
    bound inside sarva.server.app."""
    provider = MockProvider(script=script)
    monkeypatch.setattr(app_module, "build_providers", lambda: {"mock": provider})
    monkeypatch.setattr(app_module, "build_router", _mock_only_router)
    return provider


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


def test_websocket_tool_confirmation_approved_runs_the_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # write_file resolves relative to the server's cwd
    call = ToolCallBlock(id="c1", name="write_file", arguments={"path": "hi.txt", "content": "hi"})
    _use_scripted_mock(
        monkeypatch,
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="wrote it")],
    )

    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "write a file for me"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "needs_confirmation":
                ws.send_json({"approved": True})
            if data["type"] == "run_done":
                break

    finished = [e for e in events if e["type"] == "tool_finished"]
    assert len(finished) == 1
    assert finished[0]["result"]["is_error"] is False
    assert (tmp_path / "hi.txt").read_text() == "hi"
    assert events[-1]["state"] == "done"


def test_websocket_tool_confirmation_denied_skips_the_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    call = ToolCallBlock(id="c1", name="write_file", arguments={"path": "hi.txt", "content": "hi"})
    _use_scripted_mock(
        monkeypatch,
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="ok, skipped")],
    )

    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "write a file for me"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "needs_confirmation":
                ws.send_json({"approved": False})
            if data["type"] == "run_done":
                break

    finished = [e for e in events if e["type"] == "tool_finished"]
    assert len(finished) == 1
    assert finished[0]["result"]["is_error"] is True
    assert not (tmp_path / "hi.txt").exists()
    assert events[-1]["state"] == "done"


def test_websocket_auto_true_never_blocks_on_a_client_reply(tmp_path, monkeypatch):
    """`auto: true` still emits `needs_confirmation` (a destructive call did
    happen — that's informational, from the loop itself, not policy-gated),
    but `always_allow` never reads from the socket, so the loop must not
    block waiting for one. This test deliberately never sends a reply — if
    auto mode were wired wrong and the server *did* wait for one, this
    would hang until the test's own timeout instead of reaching run_done."""
    monkeypatch.chdir(tmp_path)
    call = ToolCallBlock(id="c1", name="write_file", arguments={"path": "hi.txt", "content": "hi"})
    _use_scripted_mock(
        monkeypatch,
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done")],
    )

    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "write a file for me", "auto": True})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    assert (tmp_path / "hi.txt").read_text() == "hi"
    assert events[-1]["state"] == "done"
