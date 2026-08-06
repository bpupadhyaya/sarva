"""Conformance tests for the FastAPI server — REST + WebSocket over the
agent loop. Uses FastAPI's in-process TestClient — no real network, no
running server process."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sarva.memory import session as session_module
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ImageBlock, Modality, ToolCallBlock
from sarva.providers.base import GenerateRequest, ModelCapabilities, ModelCost, ModelInfo
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass, load_routing
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


class _CapturingProvider(MockProvider):
    """Records the real GenerateRequest each call receives, on top of
    MockProvider's own default echo behavior -- proves an attached image
    actually reached the provider as a real ImageBlock, not just that the
    endpoint returned 200/a run_done frame without erroring."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: GenerateRequest | None = None

    async def generate(self, request: GenerateRequest):
        self.last_request = request
        async for event in super().generate(request):
            yield event


def _use_capturing_mock(monkeypatch) -> _CapturingProvider:
    provider = _CapturingProvider()
    monkeypatch.setattr(app_module, "build_providers", lambda: {"mock": provider})
    monkeypatch.setattr(app_module, "build_router", _mock_only_router)
    return provider


def _vision_capable_mock_router() -> Router:
    """A router whose sole "mock" entry genuinely declares IMAGE support
    -- unlike the real production registry's mock entry (see models.yaml's
    own comment on the bug this fix closes), which no longer does, since
    that lie was what silently defeated AgentLoop's degradation-fallback
    for every real image attachment. The two tests using this router only
    care whether image_base64/image_media_type on the wire correctly
    becomes a real ImageBlock that reaches provider.generate() unmodified
    -- not about routing/degradation policy -- so they get their own
    router with a genuinely vision-capable "mock" id instead of relying on
    the real registry's (now honest) capabilities."""
    model = ModelInfo(
        id="mock",
        provider="mock",
        display_name="Vision-capable mock (test-only)",
        capabilities=ModelCapabilities(
            modalities_in={Modality.TEXT, Modality.IMAGE},
            modalities_out={Modality.TEXT},
            tool_use=True,
            thinking=False,
            context_window=100_000,
            max_output=8_000,
        ),
        cost=ModelCost(),
    )
    registry = Registry(models={"mock": model})
    return Router(registry, routing={TaskClass.MAIN: ["mock"]}, available={"mock"})


def _use_capturing_vision_mock(monkeypatch) -> _CapturingProvider:
    provider = _CapturingProvider()
    monkeypatch.setattr(app_module, "build_providers", lambda: {"mock": provider})
    monkeypatch.setattr(app_module, "build_router", _vision_capable_mock_router)
    return provider


def _force_mock_only(monkeypatch) -> MockProvider:
    """For tests that want the real, DEFAULT (unscripted) Mock echo
    behavior rather than a custom script -- same patch target as
    `_use_scripted_mock`, real reason this exists: a "zero-config"
    assertion here was silently depending on unstated machine state (no
    real Ollama server happening to be reachable), true in CI and in
    this environment's own sandbox until this session actually installed
    and ran one to verify the Ollama adapter live -- which then broke
    every one of these tests, since the real router legitimately started
    preferring `ollama/qwen3:8b` (reachable but not pulled) over falling
    back to Mock. A real, latent test-isolation bug any contributor
    running this suite with their own local Ollama already running would
    have hit too, not unique to this session."""
    provider = MockProvider()
    monkeypatch.setattr(app_module, "build_providers", lambda: {"mock": provider})
    monkeypatch.setattr(app_module, "build_router", _mock_only_router)
    return provider


def _raise_config_error(*args, **kwargs):
    from sarva.config import ConfigError

    raise ConfigError(
        "config file at ~/.sarva/config.json is corrupted (invalid JSON): mock detail"
    )


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


def test_models_with_a_corrupted_config_file_fails_cleanly_not_a_500_traceback(monkeypatch):
    # A real bug found by actually corrupting ~/.sarva/config.json and
    # hitting GET /models: get_env() backs nearly every provider-
    # availability check build_router() makes, so a bad file crashed
    # this (and almost every other) endpoint with a raw
    # json.JSONDecodeError -- an unhandled 500 with a full traceback body,
    # not even a clean error response. The global ConfigError exception
    # handler now returns a clean {"detail": ...} 500 instead.
    monkeypatch.setattr(app_module, "build_router", _raise_config_error)

    resp = _client().get("/models")

    assert resp.status_code == 500
    assert "corrupted" in resp.json()["detail"]


def test_doctor_with_a_corrupted_config_file_fails_cleanly_not_a_500_traceback(monkeypatch):
    monkeypatch.setattr(app_module, "run_diagnostics", _raise_config_error)

    resp = _client().get("/doctor")

    assert resp.status_code == 500
    assert "corrupted" in resp.json()["detail"]


def test_post_config_with_a_corrupted_config_file_fails_cleanly_not_a_500_traceback(monkeypatch):
    # POST /config always calls run_diagnostics() at the end (to return
    # the fresh check results), even with an empty body that never
    # reaches save_config() -- this is the shortest real path to the bug.
    monkeypatch.setattr(app_module, "run_diagnostics", _raise_config_error)

    resp = _client().post("/config", json={})

    assert resp.status_code == 500
    assert "corrupted" in resp.json()["detail"]


def test_chat_zero_config_uses_mock(monkeypatch):
    _force_mock_only(monkeypatch)
    resp = _client().post("/chat", json={"message": "hello server"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "done"
    assert "hello server" in body["message"]
    assert "spend" in body


def test_chat_with_session_persists_across_requests(tmp_path, monkeypatch):
    _force_mock_only(monkeypatch)
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path)

    client = _client()
    r1 = client.post("/chat", json={"message": "first", "session": "web-test"})
    r2 = client.post("/chat", json={"message": "second", "session": "web-test"})
    assert r1.status_code == 200
    assert r2.status_code == 200

    store = SessionStore()
    assert len(store.load("web-test")) == 4  # 2 turns * (user + assistant)


def test_chat_with_an_explicit_empty_session_is_treated_the_same_as_no_session(monkeypatch):
    # A real bug found by a fresh-eyes sweep: `session=""` reached
    # `store.locked(req.session)` (which only special-cases `name is
    # None`, not falsy strings) and raised `ValueError: invalid session
    # name: ''` from SessionStore._sanitize() -- before the agent even
    # ran -- even though every OTHER use of `req.session` in this same
    # handler already treats "" identically to no session at all
    # (`store.load(req.session) if req.session else []`). Confirmed live
    # before this fix: an otherwise-identical request succeeded with no
    # `session` field and with `session="default"`, but failed outright
    # with `session=""` -- an ordinary client pattern (a form/state
    # field initialized to "" and always serialized, rather than
    # omitted or sent as null) hits this on every request.
    _force_mock_only(monkeypatch)

    resp = _client().post("/chat", json={"message": "hello", "session": ""})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "done"
    assert "hello" in body["message"]


async def test_two_concurrent_chat_turns_on_the_same_session_do_not_lose_a_message(
    tmp_path, monkeypatch
):
    # A real bug found by actually racing two concurrent turns on the
    # same session with asyncio.gather -- the exact concurrency model
    # /chat and /ws/chat already run under (a single asyncio event loop
    # interleaving two in-flight requests across the real `await`s an
    # agent turn makes, no threads needed to reproduce it). SessionStore.
    # load() at the start of a turn and .save() at the end are a classic
    # unlocked read-modify-write pair -- the identical bug class already
    # fixed for config.json's own concurrent-write race, just never
    # applied to session storage. Confirmed live before this fix: two
    # ordinary turns on the SAME session name (e.g. two browser tabs
    # both chatting into the default session) produced a session file
    # holding only ONE of the two turns' new messages, the other
    # silently discarded with no error to either client. Uses a real
    # httpx.AsyncClient over the app's own ASGI transport (not the
    # synchronous TestClient used elsewhere in this file, which can't
    # genuinely interleave two in-flight requests) so both turns are
    # truly concurrent, not merely sequential.
    _force_mock_only(monkeypatch)
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path)

    import httpx

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r1, r2 = await asyncio.gather(
            client.post("/chat", json={"message": "turn-A", "session": "shared"}),
            client.post("/chat", json={"message": "turn-B", "session": "shared"}),
        )
    assert r1.status_code == 200
    assert r2.status_code == 200

    store = SessionStore()
    saved = store.load("shared")
    # Both turns' user messages must survive -- 2 turns * (user +
    # assistant) = 4 messages, not 2 (one turn's save silently
    # overwriting the other's).
    assert len(saved) == 4, f"expected 4 messages from both turns, got {len(saved)}"
    user_texts = {m.text() for m in saved if m.role == "user"}
    assert user_texts == {"turn-A", "turn-B"}


async def test_a_slow_ollama_probe_does_not_stall_a_concurrent_request(monkeypatch):
    # A real bug found by a fresh-eyes sweep of runtime.py: build_router()
    # (and build_providers()/run_diagnostics(), which call it internally)
    # makes a real, SYNCHRONOUS httpx.get() call to probe whether Ollama
    # is reachable (runtime._probe_ollama) -- called directly, with no
    # asyncio.to_thread, from every one of /models, /doctor, POST /config,
    # /chat, and /ws/chat's own async handlers. A synchronous network call
    # inside an `async def` fully blocks this process's ENTIRE event loop
    # for the whole call, not just the one request making it -- confirmed
    # live before this fix, using a real httpx.AsyncClient over the app's
    # own ASGI transport (genuine concurrency, not the synchronous
    # TestClient) racing a slow /models call against an unrelated,
    # instant /health call: the slow call reliably starved /health of any
    # chance to run until the slow call finished, even though /health
    # touches none of the same code and should have returned in
    # microseconds. Fixed by wrapping each build_router()/
    # build_providers()/run_diagnostics() call at its 5 call sites in
    # asyncio.to_thread, so the blocking network I/O runs off the event
    # loop thread.
    import time

    import httpx
    import sarva.runtime as runtime_module

    def slow_get(url, timeout=0.3):
        time.sleep(0.2)  # simulate a slow/unreachable Ollama host's real network wait
        raise httpx.ConnectError("simulated slow-then-unreachable host")

    monkeypatch.setattr(runtime_module.httpx, "get", slow_get)
    runtime_module._ollama_probe_cache.clear()

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        loop = asyncio.get_event_loop()
        # Both tasks created back-to-back, timer started before either runs
        # a single step -- NOT `await`ing/sleeping between them, which
        # would let the slow task's blocking chunk run (and finish)
        # *during* that gap, making the fast request measure as fast
        # regardless of whether the bug is fixed (confirmed by hand: an
        # earlier version of this test with an intervening `await
        # asyncio.sleep(0.02)` before starting the timer passed even
        # against the unfixed code, for exactly this reason).
        t0 = loop.time()
        slow_task = asyncio.create_task(client.get("/models"))
        fast_task = asyncio.create_task(client.get("/health"))
        fast_resp = await fast_task
        fast_elapsed = loop.time() - t0
        slow_resp = await slow_task

    assert slow_resp.status_code == 200
    assert fast_resp.status_code == 200
    # The unrelated /health request must complete quickly -- if the event
    # loop were still frozen by the slow probe, this would take roughly
    # as long as the probe itself (~0.2s), not a few milliseconds.
    assert fast_elapsed < 0.1, f"/health took {fast_elapsed:.3f}s -- event loop was blocked"


async def test_a_slow_save_config_does_not_stall_a_concurrent_request(monkeypatch):
    # A real bug found by a fresh-eyes sweep, the identical shape as the
    # slow-Ollama-probe fix above but a sibling gap the original fix
    # never propagated to: sarva.config.save_config() acquires a real,
    # cross-process fcntl.flock()/msvcrt.locking() on config.json.lock
    # and blocks synchronously until it gets it -- called directly, with
    # no asyncio.to_thread, from POST /config's own async handler, even
    # though every OTHER blocking call in this same file (build_router,
    # build_providers, run_diagnostics) is already wrapped against
    # exactly this class of freeze. Confirmed live with a real second OS
    # process holding the OS-level flock for 3s: 0 of ~60 expected
    # heartbeat ticks landed on the event loop during that window.
    # Reachable with no adversarial input: the desktop app's onboarding
    # screen and `sarva config set` both write through this exact lock,
    # meant to be used interchangeably against the same running server.
    import time

    import httpx

    def slow_save_config(values, path=None):
        time.sleep(0.2)  # simulate a real, contended cross-process flock wait

    monkeypatch.setattr(app_module, "save_config", slow_save_config)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = asyncio.get_event_loop().time()
        slow_task = asyncio.create_task(
            client.post("/config", json={"anthropic_api_key": "sk-test"})
        )
        fast_task = asyncio.create_task(client.get("/health"))
        fast_resp = await fast_task
        fast_elapsed = asyncio.get_event_loop().time() - t0
        slow_resp = await slow_task

    assert slow_resp.status_code == 200
    assert fast_resp.status_code == 200
    assert fast_elapsed < 0.1, f"/health took {fast_elapsed:.3f}s -- event loop was blocked"


async def test_a_slow_session_load_does_not_freeze_the_event_loop(monkeypatch):
    # A real bug found by a fresh-eyes sweep, the identical bug class
    # already fixed at least five separate times in this project (four
    # times in agent/tools.py's memory tools, once more for that same
    # file's ReadFileTool/WriteFileTool/EditFileTool): SessionStore.
    # load() does real, synchronous filesystem I/O (a read_text() call),
    # called directly from this handler's own async def with no
    # asyncio.to_thread, even though every OTHER blocking call in this
    # same file (build_router, build_providers, run_diagnostics,
    # save_config, even the session lock's own flock/msvcrt.locking
    # acquire inside store.locked()) is already wrapped against exactly
    # this class of freeze. Confirmed live with a simulated slow disk: 0
    # of ~20 expected heartbeat ticks landed on the event loop during
    # the call.
    #
    # A concurrent-request race (the technique the save_config sibling
    # test above uses) doesn't reliably catch this specific call site:
    # store.locked(session) does its own real asyncio.to_thread dispatch
    # for the session flock *before* store.load() ever runs, and that
    # genuine yield point lets an unrelated fast request sneak in and
    # finish regardless of whether the later load() call blocks. A
    # direct heartbeat-coroutine measurement across the one request that
    # actually exercises the slow call -- the same technique this
    # project's own file/memory-tool tests use -- isn't fooled by that.
    import time

    import httpx

    _force_mock_only(monkeypatch)

    real_load = SessionStore.load

    def slow_load(self, name):
        time.sleep(0.3)  # simulate a slow/contended or network-mounted disk
        return real_load(self, name)

    monkeypatch.setattr(SessionStore, "load", slow_load)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        hb_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)  # let the heartbeat task actually start ticking first
        resp = await client.post("/chat", json={"message": "hi", "session": "default"})
        hb_task.cancel()

    assert resp.status_code == 200
    # The real, decisive assertion: while the slowed load() ran, the
    # event loop must have kept running other coroutines -- a near-zero
    # tick count is the literal old bug reproducing itself (the loop
    # frozen solid for the whole call).
    assert ticks >= 3, f"event loop only ticked {ticks} times -- looks frozen"


async def test_a_slow_session_save_does_not_freeze_the_event_loop(monkeypatch):
    # The save half of the same fix as the load test above:
    # SessionStore.save() (atomic_write_bytes' own open/write/fsync/
    # rename) has the identical blocking-I/O-with-no-asyncio.to_thread
    # shape, reached on every successful turn that has a session. Same
    # heartbeat-coroutine technique as the load test above, for the same
    # reason a concurrent-request race wouldn't reliably catch it.
    import time

    import httpx

    _force_mock_only(monkeypatch)

    real_save = SessionStore.save

    def slow_save(self, name, messages):
        time.sleep(0.3)  # simulate a slow/contended or network-mounted disk
        return real_save(self, name, messages)

    monkeypatch.setattr(SessionStore, "save", slow_save)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        hb_task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        resp = await client.post("/chat", json={"message": "hi", "session": "default"})
        hb_task.cancel()

    assert resp.status_code == 200
    assert ticks >= 3, f"event loop only ticked {ticks} times -- looks frozen"


def test_chat_without_session_does_not_persist(tmp_path, monkeypatch):
    _force_mock_only(monkeypatch)
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path)

    _client().post("/chat", json={"message": "no memory please"})

    assert SessionStore().list_sessions() == []


def test_chat_with_model_forces_that_exact_model(monkeypatch):
    provider = _use_capturing_mock(monkeypatch)

    resp = _client().post("/chat", json={"message": "hi", "model": "mock"})

    assert resp.status_code == 200
    assert resp.json()["state"] == "done"
    assert provider.last_request.model == "mock"


def test_chat_with_an_unknown_model_fails_cleanly_with_a_detail_message(monkeypatch):
    # The REST counterpart to the CLI's --model safety fix: an unknown
    # model must be a visible, clean failure -- not a 500, not a silent
    # fallback to a different model -- and (new) the actual reason must
    # reach the response body, not just a generic "failed" state.
    _force_mock_only(monkeypatch)

    resp = _client().post("/chat", json={"message": "hi", "model": "not-a-real-model"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert body["message"] is None
    assert "not-a-real-model" in body["detail"]


def test_chat_with_verify_true_rejects_a_verifier_flagged_answer(monkeypatch):
    # The REST counterpart to the CLI's --verify flag reaching AgentLoop's
    # own verify=True constructor argument -- proves the request field
    # genuinely threads through, not just that it's accepted as valid JSON.
    _use_scripted_mock(
        monkeypatch,
        [
            ScriptedTurn(text="a wrong or incomplete answer"),
            ScriptedTurn(text="REJECTED: this does not actually answer what was asked"),
        ],
    )

    resp = _client().post("/chat", json={"message": "hi", "verify": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert body["message"] is None
    assert "REJECTED" in body["detail"]


def test_chat_without_verify_never_spawns_a_verifier(monkeypatch):
    # Default False: an ordinary chat must cost exactly one real model
    # call, not two -- the same "existing behavior stays unaffected"
    # guarantee the AgentLoop-level tests already establish, checked here
    # too since this is a separate wiring path.
    _use_capturing_mock(monkeypatch)

    resp = _client().post("/chat", json={"message": "hi"})

    assert resp.status_code == 200
    assert resp.json()["state"] == "done"
    assert resp.json()["spend"]["model_calls"] == 1


def test_chat_with_an_invalid_session_name_fails_cleanly_not_a_500(monkeypatch):
    # A real bug found by actually POSTing {"session": "bad name!"}:
    # SessionStore._sanitize() raises a plain ValueError, and nothing
    # here caught it -- a genuine unhandled 500, confirmed directly
    # with raise_server_exceptions=False before this fix. Reported the
    # same shape an unknown --model already is.
    _force_mock_only(monkeypatch)

    resp = _client().post("/chat", json={"message": "hi", "session": "bad name!"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert "invalid session name" in body["detail"]


def test_chat_with_an_overly_long_session_name_fails_cleanly_not_a_500(monkeypatch):
    # A real bug found by a fresh-eyes sweep: SessionStore._sanitize()
    # validated a session name's characters but never its length, so a
    # name past the filesystem's max filename length reached os.open()
    # (inside store.locked()'s _acquire) and raised a raw OSError
    # (ENAMETOOLONG) -- a completely different exception type than the
    # ValueError this endpoint's except clause catches, since that's the
    # only exception type _sanitize() was ever documented to raise.
    # Confirmed live before the fix: this exact request crashed with a
    # raw 500 (`OSError: [Errno 63] File name too long`), not the clean
    # ChatResponse(state=failed) shape every other invalid-session-name
    # case already gets. ChatRequest.session has no length constraint,
    # so this is reachable through the server's real public REST
    # surface -- a buggy session-id generator or a very long user-typed
    # name, no adversarial intent needed.
    _force_mock_only(monkeypatch)

    resp = _client().post("/chat", json={"message": "hi", "session": "a" * 300})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert "session name too long" in body["detail"]


def test_chat_with_malformed_image_base64_fails_cleanly_not_a_500(monkeypatch):
    # A real bug found by actually POSTing {"image_base64": "not valid
    # base64!!!", ...}: base64.b64decode() raises binascii.Error (a
    # ValueError subclass), and nothing here caught it -- a genuine
    # unhandled 500, confirmed directly with raise_server_exceptions=False
    # before this fix. The exact same bug shape already fixed for an
    # invalid session name just above, now closed for images too.
    _force_mock_only(monkeypatch)

    resp = _client().post(
        "/chat",
        json={
            "message": "hi",
            "image_base64": "not valid base64!!!",
            "image_media_type": "image/png",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert body["message"] is None


def test_chat_with_a_corrupted_config_file_fails_cleanly_not_a_500_traceback(monkeypatch):
    # A real bug found by actually corrupting ~/.sarva/config.json and
    # POSTing to /chat: build_router()/build_providers() both raise
    # ConfigError uncaught, a genuine unhandled 500 -- worse, since the
    # rest of this endpoint already reports "this request can't run"
    # failures as a clean ChatResponse(state=failed, detail=...), the
    # exact same shape an unknown --model or invalid session name
    # already get, which this needed too rather than falling through to
    # a differently-shaped generic error.
    monkeypatch.setattr(app_module, "build_router", _raise_config_error)

    resp = _client().post("/chat", json={"message": "hi"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert "corrupted" in body["detail"]


def test_chat_with_an_attached_image_reaches_the_provider_as_a_real_image_block(monkeypatch):
    # Uses a vision-capable test router, not _mock_only_router -- with the
    # real production registry's mock entry now honestly declining IMAGE
    # (see models.yaml), an all-mock available set degrades any image
    # away before it reaches the provider at all. This test is about the
    # image_base64 -> ImageBlock wiring, not routing/degradation policy.
    provider = _use_capturing_vision_mock(monkeypatch)
    raw = b"\x89PNG\r\n\x1a\nreal enough bytes for this test"

    resp = _client().post(
        "/chat",
        json={
            "message": "what's in this image?",
            "image_base64": base64.b64encode(raw).decode(),
            "image_media_type": "image/png",
        },
    )

    assert resp.status_code == 200
    assert provider.last_request is not None
    user_msg = next(m for m in provider.last_request.messages if m.role == "user")
    images = [b for b in user_msg.content if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert images[0].data == raw
    assert images[0].media_type == "image/png"


def test_websocket_streams_events_and_ends_with_run_done(monkeypatch):
    _force_mock_only(monkeypatch)
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


async def test_websocket_send_side_disconnect_closes_cleanly_not_a_raw_runtimeerror(monkeypatch):
    # A real bug found by a fresh-eyes sweep: ws_chat's `finally` block
    # used to call `websocket.close()` unconditionally, but a SEND-side
    # disconnect (the client's TCP connection dropping while this
    # handler is mid-stream inside `websocket.send_text()` -- a phone
    # losing signal, a laptop sleeping, a proxy timeout; most of a
    # streaming turn's wall-clock time is spent sending, not waiting on
    # a client reply, so this is the more common disconnect shape, not
    # a rare one) already transitions Starlette's own
    # `application_state` to DISCONNECTED and raises
    # `WebSocketDisconnect`, caught cleanly by the `except` above. But
    # `close()` then unconditionally sent another `websocket.close`
    # ASGI message, and Starlette's own `send()` refuses that once
    # `application_state` is already DISCONNECTED, raising a raw
    # `RuntimeError('Cannot call "send" once a close message has been
    # sent.')` that propagated straight out of this handler -- unlike
    # every OTHER failure path in this file. Simulated here by making
    # the first `send_text` call reproduce exactly what Starlette's
    # real `send()` does internally when the underlying ASGI `send()`
    # raises `OSError` (confirmed directly against the installed
    # starlette/websockets.py source): mark the socket DISCONNECTED,
    # then raise `WebSocketDisconnect`.
    #
    # Driven directly against the raw ASGI callable (scope/receive/send),
    # not FastAPI's TestClient WebSocket session: TestClient's own
    # WebSocketTestSession runs the app in a background thread behind an
    # anyio memory-object-stream queue, and a genuinely unhandled
    # exception in the app (the exact old bug this test exists to catch)
    # leaves that queue with nothing ever sent to it -- `receive_json()`
    # on the client side then blocks forever instead of failing fast,
    # since nothing in that machinery is watching the app task's own
    # exception state. Driving the ASGI callable directly sidesteps that
    # entirely: this test simply awaits the app call and asserts on
    # what it does or doesn't raise.
    from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

    real_send_text = WebSocket.send_text
    calls = {"n": 0}

    async def flaky_send_text(self, data):
        calls["n"] += 1
        if calls["n"] == 1:
            self.application_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(code=1006)
        return await real_send_text(self, data)

    monkeypatch.setattr(WebSocket, "send_text", flaky_send_text)
    _force_mock_only(monkeypatch)
    app = create_app()

    scope = {
        "type": "websocket",
        "path": "/ws/chat",
        "headers": [(b"host", b"testserver"), (b"origin", b"http://testserver")],
        "query_string": b"",
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "subprotocols": [],
    }
    incoming = [
        {"type": "websocket.connect"},
        {"type": "websocket.receive", "text": json.dumps({"message": "hi"})},
        {"type": "websocket.disconnect", "code": 1000},
    ]

    async def receive():
        return incoming.pop(0) if incoming else {"type": "websocket.disconnect", "code": 1000}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    # Must not raise a raw RuntimeError -- the handler's own cleanup
    # must tolerate a socket that's already disconnected. This await
    # itself is the real assertion: it must complete normally.
    await app(scope, receive, send)
    assert calls["n"] >= 1  # the flaky send_text path was actually exercised


def test_websocket_rejects_a_cross_origin_connection(monkeypatch):
    # A real bug found by actually connecting a TestClient WebSocket
    # session with Origin: https://evil.example.com set: the handshake
    # was fully accepted and a real run completed. WebSocket connections
    # are NOT subject to the Same-Origin Policy/CORS the way fetch() is
    # -- any webpage a user's browser has open can drive this endpoint
    # regardless of which site's tab it's in, and ws_chat builds an
    # AgentLoop with BUILTIN_TOOLS (and, with "auto": true,
    # confirm=always_allow) -- a classic cross-site WebSocket hijacking
    # (CSWSH) vector giving a malicious page real shell/file-tool access
    # with every confirmation gate auto-approved.
    _force_mock_only(monkeypatch)
    client = _client()

    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(
            "/ws/chat", headers={"Origin": "https://evil.example.com"}
        ) as ws:
            ws.send_json({"message": "hi"})
            ws.receive_json()

    assert excinfo.value.code == 1008


def test_websocket_accepts_a_same_origin_connection_with_an_explicit_origin_header(monkeypatch):
    # The regression guard for the fix above: a real browser (including
    # the desktop app's own embedded webview, which loads its UI from
    # this exact server) always sends Origin on a WebSocket handshake --
    # the fix must not reject the ordinary, legitimate case.
    _force_mock_only(monkeypatch)
    client = _client()

    with client.websocket_connect("/ws/chat", headers={"Origin": "http://testserver"}) as ws:
        ws.send_json({"message": "hi"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    assert events[-1]["state"] == "done"


def test_websocket_with_session_persists(tmp_path, monkeypatch):
    _force_mock_only(monkeypatch)
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path)
    client = _client()

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "remember me", "session": "ws-test"})
        while ws.receive_json()["type"] != "run_done":
            pass

    assert len(SessionStore().load("ws-test")) == 2


def test_websocket_with_an_attached_image_reaches_the_provider_as_a_real_image_block(monkeypatch):
    # The real gap this closes: the desktop app's ONLY chat surface is
    # /ws/chat (see App.tsx -- it never calls /chat), and until this,
    # ws_chat never read image_base64/image_media_type from the frame at
    # all, so there was genuinely no way to send an image through the web
    # UI despite the CLI and /chat both already supporting it.
    # Same vision-capable test router as the /chat version above, for the
    # same reason -- the real registry's mock entry no longer lies about
    # IMAGE support, so this test needs its own genuinely capable model.
    provider = _use_capturing_vision_mock(monkeypatch)
    raw = b"\x89PNG\r\n\x1a\nreal enough bytes for this websocket test"

    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json(
            {
                "message": "what's in this image?",
                "image_base64": base64.b64encode(raw).decode(),
                "image_media_type": "image/png",
            }
        )
        while ws.receive_json()["type"] != "run_done":
            pass

    assert provider.last_request is not None
    user_msg = next(m for m in provider.last_request.messages if m.role == "user")
    images = [b for b in user_msg.content if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert images[0].data == raw
    assert images[0].media_type == "image/png"


def test_websocket_without_an_image_sends_no_image_block(monkeypatch):
    provider = _use_capturing_mock(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "no image here"})
        while ws.receive_json()["type"] != "run_done":
            pass

    assert provider.last_request is not None
    user_msg = next(m for m in provider.last_request.messages if m.role == "user")
    assert not any(isinstance(b, ImageBlock) for b in user_msg.content)


def test_websocket_with_model_forces_that_exact_model(monkeypatch):
    provider = _use_capturing_mock(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi", "model": "mock"})
        while ws.receive_json()["type"] != "run_done":
            pass

    assert provider.last_request is not None
    assert provider.last_request.model == "mock"


def test_websocket_with_verify_true_rejects_a_verifier_flagged_answer(monkeypatch):
    _use_scripted_mock(
        monkeypatch,
        [
            ScriptedTurn(text="a wrong or incomplete answer"),
            ScriptedTurn(text="REJECTED: this does not actually answer what was asked"),
        ],
    )
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi", "verify": True})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    assert events[-1]["state"] == "failed"
    state_changed_details = [
        e["detail"] for e in events if e["type"] == "state_changed" and e.get("detail")
    ]
    assert any("REJECTED" in d for d in state_changed_details)


def test_websocket_with_an_unknown_model_fails_cleanly_with_a_detail_frame(monkeypatch):
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi", "model": "not-a-real-model"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    state_changed = next(e for e in events if e["type"] == "state_changed" and e.get("detail"))
    assert "not-a-real-model" in state_changed["detail"]
    assert events[-1]["state"] == "failed"


def test_websocket_with_an_invalid_session_name_fails_cleanly_not_a_bare_disconnect(monkeypatch):
    # Before this fix, the client got nothing at all -- not even an
    # error frame, just a bare ClosedResourceError on the next receive,
    # confirmed directly with a real WebSocket session. Now reports the
    # identical state_changed + run_done shape the unknown-model case
    # above already does.
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi", "session": "bad name!"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    state_changed = next(e for e in events if e["type"] == "state_changed" and e.get("detail"))
    assert "invalid session name" in state_changed["detail"]
    assert events[-1]["state"] == "failed"


def test_websocket_with_an_explicit_empty_session_is_treated_the_same_as_no_session(monkeypatch):
    # The WS counterpart to the identical real bug just fixed for /chat:
    # raw, schema-less JSON means nothing stops a caller from sending
    # `"session": ""`, which used to reach `store.locked(session)` (only
    # special-cases `name is None`, not falsy strings) and raise the
    # same `invalid session name: ''` the test above pins for a
    # genuinely bad name -- even though `store.load(session) if session
    # else []` right below it already treats "" as no session.
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi", "session": ""})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    assert events[-1]["state"] == "done"


def test_websocket_with_malformed_image_base64_fails_cleanly_not_a_bare_disconnect(monkeypatch):
    # The WS counterpart to the same real bug just fixed for /chat:
    # before this fix, reaching this point uncaught crashed the whole
    # ASGI call with no frame sent at all -- a bare ClosedResourceError
    # on the next receive, confirmed directly with a real WebSocket
    # session, worse than even a REST 500.
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json(
            {
                "message": "hi",
                "image_base64": "not valid base64!!!",
                "image_media_type": "image/png",
            }
        )
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    assert events[-1]["state"] == "failed"


def test_websocket_with_a_malformed_non_json_frame_fails_cleanly_not_a_bare_disconnect(monkeypatch):
    # A real bug found by actually sending a non-JSON first frame:
    # Starlette's receive_json() does a bare json.loads() with no error
    # handling of its own -- a malformed frame raised an uncaught
    # json.JSONDecodeError here, crashing the whole ASGI call with no
    # frame ever sent at all, confirmed directly with a real WebSocket
    # session before this fix (a bare ClosedResourceError on the next
    # receive, worse than even a REST 500).
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text("not valid json{{{")
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    state_changed = next(e for e in events if e["type"] == "state_changed" and e.get("detail"))
    assert "malformed request" in state_changed["detail"]
    assert events[-1]["state"] == "failed"


def test_websocket_with_a_non_object_json_frame_fails_cleanly_not_a_bare_disconnect(monkeypatch):
    # A sibling of the malformed-JSON case: this frame IS valid JSON, but
    # not an object (a bare list) -- payload.get(...) would otherwise
    # raise an uncaught AttributeError, confirmed directly before this
    # fix.
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps([1, 2, 3]))
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    state_changed = next(e for e in events if e["type"] == "state_changed" and e.get("detail"))
    assert "expected a JSON object" in state_changed["detail"]
    assert events[-1]["state"] == "failed"


def test_websocket_with_a_non_string_session_fails_cleanly_not_a_bare_disconnect(monkeypatch):
    # A real bug found by actually sending {"session": 123}:
    # SessionStore._sanitize()'s regex match raises a plain TypeError
    # (not ValueError) on a non-string session, which the existing
    # invalid-session-name handling didn't catch -- confirmed directly
    # before this fix.
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi", "session": 123})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    assert events[-1]["state"] == "failed"


def test_websocket_with_a_non_string_model_fails_cleanly_not_a_bare_disconnect(monkeypatch):
    # A real bug found by actually sending {"model": ["a", "b"]}:
    # AgentLoop.run()'s router.pick(override=model) call, several frames
    # deep in the async generator, does a dict lookup keyed on this
    # value -- a non-string JSON type raised an uncaught TypeError
    # ("unhashable type: 'list'") well after streaming had already
    # started, confirmed directly before this fix. /chat never sees this
    # because Pydantic validates ChatRequest.model's type first.
    _force_mock_only(monkeypatch)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi", "model": ["a", "b"]})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    state_changed = next(e for e in events if e["type"] == "state_changed" and e.get("detail"))
    assert "model must be a string" in state_changed["detail"]
    assert events[-1]["state"] == "failed"


def test_websocket_with_a_corrupted_config_file_fails_cleanly_not_a_bare_disconnect(monkeypatch):
    # The WS counterpart to the same real bug just fixed for /chat: a
    # corrupted ~/.sarva/config.json made build_router() raise
    # ConfigError uncaught -- the whole ASGI call would crash with no
    # frame sent at all, a bare ClosedResourceError, the same failure
    # mode already fixed here for an invalid session name and a
    # malformed image_base64 field.
    monkeypatch.setattr(app_module, "build_router", _raise_config_error)
    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "hi"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    state_changed = next(e for e in events if e["type"] == "state_changed" and e.get("detail"))
    assert "corrupted" in state_changed["detail"]
    assert events[-1]["state"] == "failed"


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


def test_websocket_confirmation_reply_with_a_string_false_is_treated_as_a_decline(
    tmp_path, monkeypatch
):
    # A real bug found by a fresh-eyes sweep: `ws_confirm` cast the reply
    # with a bare `bool(reply.get("approved", False))` -- Python
    # truthiness, not the strict boolean check the documented
    # `{"approved": bool}` wire contract actually promises.
    # `bool("false")` is `True`, since any non-empty string is truthy in
    # Python. Confirmed live before this fix: a client sending
    # `{"approved": "false"}` (a JSON STRING, not a JSON boolean --
    # exactly what a form/select's `.value`, or any client serializing
    # booleans as strings, produces with no explicit `=== "true"`
    # conversion) got the destructive `write_file` call APPROVED, the
    # opposite of the sender's intent, with no error and no signal
    # anything was wrong -- directly contradicting this same function's
    # own "an ambiguous or absent confirmation signal must never default
    # to running" discipline already applied to the timeout/non-dict
    # cases right above this one.
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
                ws.send_json({"approved": "false"})
            if data["type"] == "run_done":
                break

    finished = [e for e in events if e["type"] == "tool_finished"]
    assert len(finished) == 1
    assert finished[0]["result"]["is_error"] is True
    assert not (tmp_path / "hi.txt").exists()
    assert events[-1]["state"] == "done"


def test_websocket_a_malformed_confirmation_reply_is_treated_as_a_decline_not_a_crash(
    tmp_path, monkeypatch
):
    # A real bug found by actually sending a JSON array instead of
    # {"approved": bool} as the confirmation reply: reply.get(...) on a
    # non-dict raised an uncaught AttributeError that crashed the whole
    # ASGI call with no run_done frame sent -- the same bare
    # ClosedResourceError this file's other raw-JSON fixes exist to
    # prevent, just one layer deeper (AgentLoop's own confirm callback,
    # not the initial payload). Treated the same way an explicit decline
    # already is: an ambiguous confirmation signal must never default to
    # running a destructive action.
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
                ws.send_json(["not", "a", "dict"])
            if data["type"] == "run_done":
                break

    finished = [e for e in events if e["type"] == "tool_finished"]
    assert len(finished) == 1
    assert finished[0]["result"]["is_error"] is True
    assert not (tmp_path / "hi.txt").exists()
    assert events[-1]["state"] == "done"


def test_websocket_a_confirmation_reply_that_never_arrives_times_out_as_a_decline(
    tmp_path, monkeypatch
):
    # A real bug found by actually running a turn that needs
    # confirmation and never replying at all: receive_json() had no
    # timeout, so the connection hung forever with no recovery -- the
    # confirmation-layer counterpart to the already-fixed hung-tool-call
    # bug, a different call site entirely (AgentLoop's own confirm
    # callback, not run_one's tool.run() wrapper).
    import sarva.server.app as app_module

    monkeypatch.setattr(app_module, "_CONFIRM_TIMEOUT_SECONDS", 0.2)
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
            # Deliberately never reply to needs_confirmation.
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


def test_websocket_auto_as_a_string_false_does_not_silently_enable_auto_mode(tmp_path, monkeypatch):
    # A real bug found by a fresh-eyes sweep, the identical shape
    # already found and fixed for the confirmation reply's own
    # `approved` field (see ws_confirm's own comment), just never
    # propagated to `auto` in the same handler: `bool(...)` is a
    # Python-truthiness cast, not a strict boolean check --
    # `bool("false")` is `True`, since any non-empty string is truthy.
    # `auto` selects `always_allow` (zero user confirmation) over the
    # real confirmation gate for every destructive tool call in the
    # whole session -- confirmed live before this fix: a client sending
    # `"auto": "false"` (a JSON STRING, exactly what a form/select's
    # `.value`, or any client serializing booleans as strings, produces)
    # got every destructive tool call auto-approved with zero
    # confirmation prompts, the opposite of what a caller sending a
    # falsy-looking value would reasonably expect -- materially more
    # severe than the `approved` bug, since that one could misapprove
    # only one already-surfaced prompt, while this one disables the
    # confirmation system for the entire session.
    import sarva.server.app as app_module

    monkeypatch.setattr(app_module, "_CONFIRM_TIMEOUT_SECONDS", 0.2)
    monkeypatch.chdir(tmp_path)
    call = ToolCallBlock(id="c1", name="write_file", arguments={"path": "hi.txt", "content": "hi"})
    _use_scripted_mock(
        monkeypatch,
        script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="ok, skipped")],
    )

    client = _client()
    with client.websocket_connect("/ws/chat") as ws:
        # Deliberately never replies to any needs_confirmation frame --
        # if "false" were wrongly treated as auto mode, the tool would
        # run immediately with no prompt at all; if correctly treated as
        # real confirmation-gated mode, this must time out as a decline.
        ws.send_json({"message": "write a file for me", "auto": "false"})
        events = []
        while True:
            data = ws.receive_json()
            events.append(data)
            if data["type"] == "run_done":
                break

    finished = [e for e in events if e["type"] == "tool_finished"]
    assert len(finished) == 1
    assert finished[0]["result"]["is_error"] is True
    assert not (tmp_path / "hi.txt").exists()


def test_doctor_endpoint_returns_the_same_checks_the_cli_command_reports(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    resp = _client().get("/doctor")

    assert resp.status_code == 200
    names = {c["name"] for c in resp.json()}
    assert names == {
        "Anthropic API key",
        "OpenAI API key",
        "Google API key",
        "Ollama (local models)",
        "Foundry (local from-scratch models)",
        "Speech-to-text (local Whisper)",
        "Text-to-speech (local)",
    }
    anthropic_check = next(c for c in resp.json() if c["name"] == "Anthropic API key")
    assert anthropic_check["ok"] is False


def test_post_config_persists_a_key_and_the_next_doctor_call_sees_it(tmp_path, monkeypatch):
    # The real end-to-end proof this endpoint exists for: a key saved via
    # POST /config must be reflected in a SEPARATE, subsequent GET
    # /doctor call -- proving it round-tripped through a real file, not
    # just an in-memory value that happened to still be set for this one
    # request.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import sarva.config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "config.json")

    client = _client()
    resp = client.post("/config", json={"anthropic_api_key": "sk-server-test"})
    assert resp.status_code == 200
    saved_check = next(c for c in resp.json() if c["name"] == "Anthropic API key")
    assert saved_check["ok"] is True

    doctor_resp = client.get("/doctor")
    fresh_check = next(c for c in doctor_resp.json() if c["name"] == "Anthropic API key")
    assert fresh_check["ok"] is True


def test_post_config_persists_a_google_api_key(tmp_path, monkeypatch):
    # A real bug found by a fresh-eyes sweep: SaveConfigRequest's own
    # docstring says "four provider-key names" are accepted, but only
    # three fields were ever declared -- google_api_key was missing.
    # GOOGLE_API_KEY has been a first-class entry in sarva.config.
    # KNOWN_KEYS/get_env's own Gemini fallback (and already correctly
    # reported by `sarva config show`) since it was added, but nothing
    # actually persisted it: because Pydantic ignores unknown fields by
    # default, POSTing google_api_key didn't error, it silently
    # vanished with a 200 OK. Confirmed live before this fix: the saved
    # config file ended up empty.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import sarva.config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "config.json")

    client = _client()
    resp = client.post("/config", json={"google_api_key": "AIzaSy-server-test"})
    assert resp.status_code == 200

    saved = config_module.load_config()
    assert saved == {"GOOGLE_API_KEY": "AIzaSy-server-test"}

    saved_check = next(c for c in resp.json() if c["name"] == "Google API key")
    assert saved_check["ok"] is True


def test_post_config_with_no_keys_does_not_write_a_file(tmp_path, monkeypatch):
    import sarva.config as config_module

    config_path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)

    resp = _client().post("/config", json={})

    assert resp.status_code == 200
    assert not config_path.exists()
