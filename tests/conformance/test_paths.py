"""Conformance tests for sarva.paths — the SARVA_HOME environment
variable override every ~/.sarva-relative path now goes through."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sarva.paths import sarva_home


def test_sarva_home_env_var_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SARVA_HOME", str(tmp_path / "custom"))
    assert sarva_home() == tmp_path / "custom"


def test_sarva_home_falls_back_to_dot_sarva_under_the_real_home(monkeypatch):
    monkeypatch.delenv("SARVA_HOME", raising=False)
    assert sarva_home() == Path.home() / ".sarva"


def test_a_fresh_subprocess_honors_sarva_home_for_session_storage(tmp_path):
    # A real gap found live, not hypothetical: this project's own testing
    # practice repeatedly assumed SARVA_HOME already worked as a sandbox
    # for exactly this kind of subprocess invocation -- it silently did
    # nothing, and real test session files landed in this machine's
    # actual ~/.sarva/sessions instead. This test proves the mechanism
    # works the one way that actually matters: a genuinely fresh Python
    # process (module-level constants computed once at import time, the
    # same way every real `sarva` CLI invocation works), not an
    # in-process CliRunner call reusing already-imported modules.
    home = tmp_path / "sandboxed-home"
    env = dict(os.environ)
    env["SARVA_HOME"] = str(home)
    core_src = str(Path(__file__).resolve().parent.parent.parent / "core")
    env["PYTHONPATH"] = core_src

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sarva.memory.session import DEFAULT_SESSIONS_DIR; print(DEFAULT_SESSIONS_DIR)",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(home / "sessions")


def test_a_fresh_subprocess_honors_sarva_home_for_config_and_memory_together(tmp_path):
    # The same real gap, confirmed for the other three call sites at
    # once: config.py, memory/vector.py, memory/longterm.py each
    # independently hardcoded Path.home() / ".sarva" before this fix --
    # one SARVA_HOME now redirects all four as a single, coherent
    # profile rather than needing four separate overrides that could
    # disagree.
    home = tmp_path / "sandboxed-home"
    env = dict(os.environ)
    env["SARVA_HOME"] = str(home)
    core_src = str(Path(__file__).resolve().parent.parent.parent / "core")
    env["PYTHONPATH"] = core_src

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sarva.config import DEFAULT_CONFIG_PATH\n"
            "from sarva.memory.vector import DEFAULT_MEMORY_DB_PATH\n"
            "from sarva.memory.longterm import DEFAULT_LONGTERM_MEMORY_DIR\n"
            "print(DEFAULT_CONFIG_PATH)\n"
            "print(DEFAULT_MEMORY_DB_PATH)\n"
            "print(DEFAULT_LONGTERM_MEMORY_DIR)\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines == [
        str(home / "config.json"),
        str(home / "memory.db"),
        str(home / "memory"),
    ]
