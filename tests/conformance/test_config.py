"""Conformance tests for sarva.config — the persistent API-key store the
desktop app's promised first-run flow depends on (T4's own definition of
done names "guided first-run... paste an API key," but until now there
was nowhere for a pasted key to actually go)."""

from __future__ import annotations

import stat
import sys

import pytest
from sarva.config import get_env, load_config, save_config

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


def test_save_config_creates_the_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "config.json"
    save_config({"ANTHROPIC_API_KEY": "sk-ant-test"}, path=path)

    assert path.is_file()


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
