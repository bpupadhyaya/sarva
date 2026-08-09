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


def test_build_providers_skips_a_foundry_dir_where_every_bundle_is_corrupted(monkeypatch, tmp_path):
    # A real bug found by a fresh-eyes sweep, one layer beyond
    # FoundryProvider.__init__'s own per-bundle try/except (which already
    # skips an individually-corrupted bundle and records it in
    # broken_bundles): if EVERY discovered bundle fails to load, the
    # constructor itself raises ValueError -- "zero out of N loaded", a
    # real, separate failure mode its per-bundle handling can't paper
    # over. build_providers() had no error handling around this
    # constructor call at all. Confirmed live through the real CLI:
    # `sarva eval` with SARVA_FOUNDRY_CHECKPOINTS pointing at a directory
    # of only corrupted bundles crashed with a raw ValueError traceback
    # -- eval_cmd/_eval() has no try/except of its own around
    # build_providers(), unlike /chat, /ws/chat, and sarva chat/run,
    # which only happen to catch this by accident (a broad
    # `except ValueError` originally added for an unrelated concern,
    # invalid session names).
    _clear_frontier_keys(monkeypatch)
    monkeypatch.setenv("SARVA_FOUNDRY_CHECKPOINTS", str(tmp_path))
    bad = tmp_path / "totally-broken"
    bad.mkdir()
    (bad / "config.json").write_text("not valid json at all")
    (bad / "tokenizer.json").write_text("also not valid")
    (bad / "model.pt").write_bytes(b"garbage")

    providers = runtime.build_providers()  # must not raise

    assert "foundry" not in providers
    assert "mock" in providers


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


async def test_ollama_probe_cache_is_a_real_single_flight_under_concurrent_callers(monkeypatch):
    # A real bug found by a fresh-eyes sweep, one layer beyond the fix
    # above: the cache's own read-check and write-back are two separate
    # steps with a real, blocking httpx.get() in between -- an
    # unguarded check-then-act race. build_router()/build_providers()
    # are called from every /chat, /ws/chat, /models, and /doctor
    # handler via asyncio.to_thread, so genuine concurrent requests
    # (the desktop app's own two independent useEffect hooks fire
    # /doctor and /models simultaneously on page load, not contrived)
    # land in separate real OS threads. Confirmed live before this fix:
    # 6 concurrent build_router() calls against a cold cache made 6
    # real network calls, not the 1 the cache exists to guarantee.
    import asyncio
    import threading
    import time

    import httpx
    import sarva.runtime as runtime_module

    _clear_frontier_keys(monkeypatch)
    monkeypatch.delenv("SARVA_FOUNDRY_CHECKPOINTS", raising=False)
    runtime_module._ollama_probe_cache.clear()

    calls = {"n": 0}
    calls_lock = threading.Lock()

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"models": []}

    def slow_get(*args, **kwargs):
        with calls_lock:
            calls["n"] += 1
        time.sleep(0.1)  # simulate real network latency, forcing genuine overlap
        return _FakeResponse()

    monkeypatch.setattr(httpx, "get", slow_get)

    await asyncio.gather(*[asyncio.to_thread(runtime.build_router) for _ in range(6)])

    assert calls["n"] == 1, f"expected exactly 1 real network call, got {calls['n']}"


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


def test_ollama_probe_survives_a_reachable_host_answering_with_a_non_json_body(monkeypatch):
    # A real bug found by a fresh-eyes sweep: `except httpx.HTTPError`
    # alone doesn't cover whatever's actually listening at OLLAMA_HOST
    # answering with a 200 whose body isn't the JSON shape Ollama's own
    # /api/tags returns -- a real, ordinary condition (a corporate
    # captive portal, a stale/misconfigured reverse proxy, simple port
    # reuse by an unrelated service), not a contrived attack.
    # response.json() raises json.JSONDecodeError on a non-JSON body,
    # confirmed live before this fix -- not a subclass of
    # httpx.HTTPError, so it propagated straight out of
    # ollama_reachable()/ollama_pulled_models() into build_router()/
    # run_diagnostics(), crashing GET /models and GET /doctor (neither
    # has its own try/except, nor is there a generic exception handler
    # registered besides ConfigError's own) with a raw, plain-text 500.
    import httpx

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            import json

            return json.loads("<html>captive portal</html>")

    def _fake_get(url, timeout):
        return _FakeResponse()

    monkeypatch.setattr(httpx, "get", _fake_get)
    runtime._ollama_probe_cache.clear()

    # Must not raise -- the whole point of a "best-effort" probe.
    assert runtime.ollama_reachable("http://weird-host") is True  # connection itself succeeded
    runtime._ollama_probe_cache.clear()
    assert runtime.ollama_pulled_models("http://weird-host") == set()


def test_ollama_probe_survives_a_reachable_host_returning_a_differently_shaped_json_body(
    monkeypatch,
):
    # A sibling gap found by the same sweep: a *valid* JSON body that
    # isn't Ollama's own shape (a top-level list instead of a dict, or a
    # model entry missing "name") raises AttributeError/KeyError instead
    # of json.JSONDecodeError -- neither a subclass of httpx.HTTPError
    # either, and neither of these narrower types alone would have
    # covered the other, the same "one exception type at a time" gap
    # already found and closed elsewhere in this project.
    import httpx

    class _FakeListResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return ["not", "the", "expected", "shape"]

    def _fake_get(url, timeout):
        return _FakeListResponse()

    monkeypatch.setattr(httpx, "get", _fake_get)
    runtime._ollama_probe_cache.clear()

    assert runtime.ollama_pulled_models("http://another-weird-host") == set()
