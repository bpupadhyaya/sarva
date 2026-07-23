"""Conformance tests for sarva.runtime.build_router's Ollama availability
logic -- specifically, the real bug found and fixed by actually running
Sarva against a real local Ollama server with a small model pulled
instead of the registry's own default `qwen3:8b`.

Before this fix, `build_router()` marked EVERY registered `ollama/*`
model "available" the instant the server was merely reachable, with no
regard for which model tag was actually pulled. A real request then
routed straight to an unpulled model and failed outright -- the
zero-config Mock fallback never got a chance, because the router
believed an unpulled model was a working one. Reproduced directly in
this environment (Ollama reachable, only a small model pulled, `sarva
run` failing) before writing the fix."""

from __future__ import annotations

import sarva.runtime as runtime


def _clear_frontier_keys(monkeypatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_ollama_model_is_unavailable_when_reachable_but_not_the_pulled_tag(monkeypatch):
    # The real scenario this session hit directly: Ollama running, but
    # only a small model pulled -- NOT the registry's registered
    # `qwen3:8b`. Before the fix, this router would have marked
    # `ollama/qwen3:8b` available anyway.
    _clear_frontier_keys(monkeypatch)
    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: True)
    monkeypatch.setattr(runtime, "ollama_pulled_models", lambda *a, **kw: {"qwen2.5:0.5b"})

    router = runtime.build_router()

    assert "ollama/qwen3:8b" not in router.available
    assert "mock" in router.available


def test_ollama_model_is_available_when_the_exact_pulled_tag_matches(monkeypatch):
    _clear_frontier_keys(monkeypatch)
    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: True)
    monkeypatch.setattr(runtime, "ollama_pulled_models", lambda *a, **kw: {"qwen3:8b"})

    router = runtime.build_router()

    assert "ollama/qwen3:8b" in router.available


def test_no_ollama_model_is_available_when_the_server_is_unreachable(monkeypatch):
    _clear_frontier_keys(monkeypatch)
    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: False)
    # Deliberately not mocking ollama_pulled_models -- build_router()
    # must not call it at all when the server itself isn't reachable
    # (short-circuited), proven by never having to answer.
    monkeypatch.setattr(
        runtime,
        "ollama_pulled_models",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    router = runtime.build_router()

    assert "ollama/qwen3:8b" not in router.available
    assert "mock" in router.available


def test_ollama_pulled_models_parses_the_real_api_tags_response_shape(monkeypatch):
    # Ollama's real /api/tags response nests each model under a "models"
    # list, keyed by "name" -- confirmed against a real running server
    # while building this. A wrong key here would silently return an
    # empty set forever, not a loud failure.
    import httpx

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "qwen2.5:0.5b"}, {"name": "llama3.2:1b"}]}

    def _fake_get(url, timeout):
        return _FakeResponse()

    monkeypatch.setattr(httpx, "get", _fake_get)

    assert runtime.ollama_pulled_models("http://fake-host") == {"qwen2.5:0.5b", "llama3.2:1b"}
