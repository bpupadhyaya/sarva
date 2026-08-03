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

import json

import sarva.runtime as runtime


def _clear_frontier_keys(monkeypatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _stub_bundle(directory, config_data: dict) -> None:
    # model_info_for_bundle() is documented as torch-free -- it reads
    # only config.json -- so a real trained checkpoint isn't needed to
    # exercise it; tokenizer.json/model.pt just need to exist for
    # discover_checkpoint_bundles' own file-presence check.
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config_data))
    (directory / "tokenizer.json").write_text("{}")
    (directory / "model.pt").write_bytes(b"")


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


def test_build_router_skips_a_corrupted_foundry_checkpoint_instead_of_crashing(
    monkeypatch, tmp_path
):
    # A real bug found by actually corrupting a bundle's config.json two
    # ways: model_info_for_bundle() raised an uncaught
    # json.JSONDecodeError (malformed JSON) or KeyError (valid JSON
    # missing the required "max_seq_len" key), and build_router()'s own
    # loop over discover_checkpoint_bundles() had no error handling at
    # all -- one corrupted bundle crashed the whole router, and
    # therefore every caller of build_router() (every CLI command,
    # /chat, /ws/chat, server startup), not just a request that tried to
    # use that specific checkpoint. discover_checkpoint_bundles only
    # checks that the three bundle files *exist*, never that config.json
    # is actually valid. The same "corrupted on-disk state" bug class
    # already fixed for FoundryProvider.__init__ (a separate call site,
    # used by build_providers()), just not closed here until now.
    _clear_frontier_keys(monkeypatch)
    monkeypatch.setenv("SARVA_FOUNDRY_CHECKPOINTS", str(tmp_path))
    _stub_bundle(tmp_path / "good", {"max_seq_len": 128})
    (tmp_path / "malformed-json").mkdir()
    (tmp_path / "malformed-json" / "config.json").write_text("{not valid json")
    (tmp_path / "malformed-json" / "tokenizer.json").write_text("{}")
    (tmp_path / "malformed-json" / "model.pt").write_bytes(b"")
    _stub_bundle(tmp_path / "missing-key", {"not_max_seq_len": 1})

    router = runtime.build_router()

    assert "foundry/good" in router.available
    assert "foundry/malformed-json" not in router.available
    assert "foundry/missing-key" not in router.available


def test_ollama_probe_is_cached_across_build_router_and_build_providers(monkeypatch):
    # A real bug found by actually running build_router() immediately
    # followed by build_providers() the way sarva.server.app's /chat and
    # /ws/chat handlers do it on every request: ollama_reachable() and
    # ollama_pulled_models() each independently hit Ollama's real
    # /api/tags endpoint, so one request made 2-3 redundant network
    # calls -- for every request, even ones never routed to Ollama at
    # all, since routing hasn't been decided yet when these run.
    import httpx
    import sarva.runtime as runtime_module

    _clear_frontier_keys(monkeypatch)
    monkeypatch.delenv("SARVA_FOUNDRY_CHECKPOINTS", raising=False)
    runtime_module._ollama_probe_cache.clear()

    calls = {"n": 0}

    def counting_get(*args, **kwargs):
        calls["n"] += 1
        raise httpx.ConnectError("simulated unreachable")

    monkeypatch.setattr(httpx, "get", counting_get)

    runtime.build_router()
    runtime.build_providers()

    assert calls["n"] == 1


def test_ollama_probe_reachability_and_pulled_models_semantics_survive_caching(monkeypatch):
    # The cache must preserve the two functions' independent pre-existing
    # semantics: ollama_reachable() is True whenever the connection itself
    # succeeds, regardless of HTTP status, while ollama_pulled_models()
    # only returns real data on an actual 2xx response.
    import httpx
    import sarva.runtime as runtime_module

    class _FakeErrorResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError("bad status", request=None, response=None)

    runtime_module._ollama_probe_cache.clear()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _FakeErrorResponse())

    assert runtime.ollama_reachable("http://fake-500-host") is True
    assert runtime.ollama_pulled_models("http://fake-500-host") == set()


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
