"""Conformance tests for sarva.config — the persistent API-key store the
desktop app's promised first-run flow depends on (T4's own definition of
done names "guided first-run... paste an API key," but until now there
was nowhere for a pasted key to actually go)."""

from __future__ import annotations

import stat
import sys
import threading
import time

import pytest
from sarva.config import get_env, load_config, save_config, unset_config

_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.chmod's real per-user isolation is POSIX-only -- see sarva.config's docstring",
)


def test_load_config_on_a_missing_file_returns_empty_not_a_crash(tmp_path):
    assert load_config(tmp_path / "does-not-exist.json") == {}


def test_save_and_load_config_round_trips(tmp_path):
    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-ant-test"}, path=path)

    assert load_config(path) == {"ANTHROPIC_API_KEY": "sk-ant-test"}


def test_save_config_merges_rather_than_overwriting_other_keys(tmp_path):
    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-ant-test"}, path=path)
    save_config({"OPENAI_API_KEY": "sk-oai-test"}, path=path)

    assert load_config(path) == {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "OPENAI_API_KEY": "sk-oai-test",
    }


def test_save_config_does_not_destroy_the_previous_file_if_interrupted_mid_write(
    tmp_path, monkeypatch
):
    # A real bug found by actually simulating an interrupted write: the
    # previous implementation opened the real config file directly with
    # O_TRUNC, truncating it to 0 bytes immediately -- before a single
    # byte of new content was written. A crash (OOM-kill, SIGKILL, power
    # loss) between that open() and the write completing destroyed every
    # previously-saved key, not just failed to save the new one.
    # Simulated by making os.replace() (the final, atomic commit step)
    # raise partway through a second save.
    import os as os_module

    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-real-important-key"}, path=path)
    assert path.stat().st_size > 0

    def crash_before_replace(*args, **kwargs):
        raise SystemExit("simulated crash before the atomic rename")

    monkeypatch.setattr(os_module, "replace", crash_before_replace)
    with pytest.raises(SystemExit):
        save_config({"OPENAI_API_KEY": "sk-new-key-never-commits"}, path=path)
    monkeypatch.undo()

    assert load_config(path) == {"ANTHROPIC_API_KEY": "sk-real-important-key"}


@_posix_only
def test_save_config_writes_the_file_with_owner_only_permissions(tmp_path):
    # The real gap this pins: Path.write_text's default open() mode
    # (0666, reduced by the process umask) left this file world-readable
    # -- confirmed with a real stat() call against an actual saved file
    # before writing this fix, not assumed from reading the stdlib docs.
    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-ant-test"}, path=path)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


@_posix_only
def test_save_config_tightens_permissions_on_a_file_that_already_existed_insecurely(tmp_path):
    # os.open's mode argument only applies when it actually creates a
    # new file -- a config.json written by a version of this module
    # predating this fix (or by anything else) must still get tightened
    # on the next save, not stay exposed forever.
    path = tmp_path / "config.json"
    path.write_text("{}")
    path.chmod(0o644)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644  # sanity: the insecure state is real

    save_config({"ANTHROPIC_API_KEY": "sk-ant-test"}, path=path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_exclusive_lock_actually_serializes_two_concurrent_acquirers(tmp_path):
    # Proves the primitive save_config/unset_config's race fix depends
    # on genuinely works: a second acquirer must block until the first
    # releases, not interleave. Widened with a real sleep while A holds
    # the lock (and a barrier so B's attempt genuinely overlaps with
    # A's critical section, not just runs after it by luck) so the
    # ordering below is deterministic, not a race the test itself could
    # flake on.
    from sarva.config import _exclusive_lock

    lock_path = tmp_path / "test.lock"
    order: list[str] = []
    barrier = threading.Barrier(2)

    def holder():
        with _exclusive_lock(lock_path):
            order.append("A-acquired")
            barrier.wait()
            time.sleep(0.2)
            order.append("A-released")

    def waiter():
        barrier.wait()
        time.sleep(0.05)  # give A a head start so B's attempt genuinely blocks
        with _exclusive_lock(lock_path):
            order.append("B-acquired")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert order == ["A-acquired", "A-released", "B-acquired"]


def test_save_config_survives_two_concurrent_callers_with_no_lost_update(tmp_path, monkeypatch):
    # A real bug found by actually simulating two concurrent callers
    # (e.g. the CLI and the desktop app, or two CLI invocations): the
    # unlocked read-modify-write in save_config/unset_config let both
    # read the SAME "before" state and each write their own merged dict
    # back -- the second write silently discarded the first's key, a
    # genuine lost update, confirmed live before this fix. A real sleep
    # injected into _write_config (called while the lock is held)
    # widens the critical section so a second, concurrent save_config
    # call genuinely has to wait rather than getting lucky with thread
    # scheduling -- the same "prove it deterministically, don't hope"
    # discipline as the lock-serialization test above.
    import sarva.config as config_module

    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-original"}, path=path)

    real_write_config = config_module._write_config

    def slow_write_config(write_path, data):
        time.sleep(0.1)
        real_write_config(write_path, data)

    monkeypatch.setattr(config_module, "_write_config", slow_write_config)

    t1 = threading.Thread(
        target=save_config, args=({"OPENAI_API_KEY": "sk-a"},), kwargs={"path": path}
    )
    t2 = threading.Thread(
        target=save_config, args=({"GEMINI_API_KEY": "sk-b"},), kwargs={"path": path}
    )
    t1.start()
    time.sleep(0.02)  # ensure t1 has entered the critical section first
    t2.start()
    t1.join()
    t2.join()

    result = load_config(path)
    assert result["ANTHROPIC_API_KEY"] == "sk-original"
    assert result["OPENAI_API_KEY"] == "sk-a"
    assert result["GEMINI_API_KEY"] == "sk-b"


def test_save_config_creates_the_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-ant-test"}, path=path)

    assert path.is_file()


def test_unset_config_removes_the_named_key_and_leaves_others(tmp_path):
    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-b"}, path=path)

    removed = unset_config(["ANTHROPIC_API_KEY"], path=path)

    assert removed == ["ANTHROPIC_API_KEY"]
    assert load_config(path) == {"OPENAI_API_KEY": "sk-b"}


def test_unset_config_on_a_missing_file_is_a_clean_no_op(tmp_path):
    path = tmp_path / "does-not-exist.json"

    removed = unset_config(["ANTHROPIC_API_KEY"], path=path)

    assert removed == []
    assert not path.exists()  # never created just to remove nothing from it


def test_unset_config_a_key_that_was_never_saved_is_a_no_op(tmp_path):
    path = tmp_path / "config.json"
    save_config({"OPENAI_API_KEY": "sk-b"}, path=path)

    removed = unset_config(["ANTHROPIC_API_KEY"], path=path)

    assert removed == []
    assert load_config(path) == {"OPENAI_API_KEY": "sk-b"}  # untouched


@_posix_only
def test_unset_config_preserves_owner_only_permissions_after_editing(tmp_path):
    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-b"}, path=path)

    unset_config(["ANTHROPIC_API_KEY"], path=path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_unset_config_does_not_destroy_the_previous_file_if_interrupted_mid_write(
    tmp_path, monkeypatch
):
    # unset_config shares _write_config with save_config, so it inherits
    # the same real bug and the same fix -- pinned separately since it's
    # a distinct call site (edit-then-write, not merge-then-write).
    import os as os_module

    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-b"}, path=path)

    def crash_before_replace(*args, **kwargs):
        raise SystemExit("simulated crash before the atomic rename")

    monkeypatch.setattr(os_module, "replace", crash_before_replace)
    with pytest.raises(SystemExit):
        unset_config(["ANTHROPIC_API_KEY"], path=path)
    monkeypatch.undo()

    assert load_config(path) == {"ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-b"}


def test_get_env_returns_none_when_nothing_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_env("ANTHROPIC_API_KEY", path=tmp_path / "config.json") is None


def test_get_env_falls_back_to_the_saved_config_value(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-from-config"}, path=path)

    assert get_env("ANTHROPIC_API_KEY", path=path) == "sk-from-config"


def test_get_env_prefers_a_real_environment_variable_over_the_saved_config(tmp_path, monkeypatch):
    # The precedence this module's own docstring states explicitly: an
    # env var a user actually exported must win over a stale saved file.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-from-config"}, path=path)

    assert get_env("ANTHROPIC_API_KEY", path=path) == "sk-from-env"


def test_a_config_file_only_key_actually_authenticates_the_real_sdk_client(tmp_path, monkeypatch):
    # The property that actually matters, not just that build_providers()
    # doesn't crash: the anthropic/openai/google SDKs each read
    # os.environ THEMSELVES if not told an api_key explicitly, so a key
    # that exists only in sarva.config's saved file (never a real env
    # var) would otherwise pass every availability check and then fail
    # to authenticate the moment a real request went out. This confirms
    # sarva.runtime.build_providers() constructs each SDK client with an
    # EXPLICIT api_key sourced via sarva.config.get_env, not left to the
    # SDK's own (config-file-blind) auto-detection.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    import sarva.config as config_module
    import sarva.runtime as runtime

    config_path = tmp_path / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-config-only-test"}, path=config_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: False)

    providers = runtime.build_providers()

    assert "anthropic" in providers
    assert providers["anthropic"]._client.api_key == "sk-config-only-test"
