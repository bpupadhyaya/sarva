"""Conformance tests for sarva.runtime.run_diagnostics and the `sarva
doctor` CLI command it backs. Named directly in the design doc's own
repo-structure diagram since T0 (`cli.py # ... chat, run, serve, models,
doctor`) but never built until now -- confirmed missing by grep before
starting. The property that actually matters: this report can never
silently drift out of sync with what build_router()/build_providers()
treat as "available", since run_diagnostics() reads the exact same env
vars and calls the exact same helpers."""

from __future__ import annotations

import sarva.runtime as runtime
from sarva.cli import app
from sarva.runtime import DiagnosticCheck, run_diagnostics
from typer.testing import CliRunner

runner = CliRunner()


def _clear_provider_env(monkeypatch) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "SARVA_FOUNDRY_CHECKPOINTS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: False)


def test_run_diagnostics_reports_every_provider_as_unavailable_with_nothing_configured(
    monkeypatch,
):
    _clear_provider_env(monkeypatch)

    checks = run_diagnostics()

    names = {c.name for c in checks}
    assert names == {
        "Anthropic API key",
        "OpenAI API key",
        "Google API key",
        "Ollama (local models)",
        "Foundry (local from-scratch models)",
    }
    assert all(isinstance(c, DiagnosticCheck) for c in checks)
    assert all(c.ok is False for c in checks)
    # Every detail names WHY it's unavailable, not just that it is.
    assert all(c.detail for c in checks)


def test_run_diagnostics_reflects_a_real_anthropic_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    checks = {c.name: c for c in run_diagnostics()}

    assert checks["Anthropic API key"].ok is True
    assert "is set" in checks["Anthropic API key"].detail
    assert checks["OpenAI API key"].ok is False


def test_run_diagnostics_reflects_google_key_from_either_env_var(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    checks = {c.name: c for c in run_diagnostics()}

    assert checks["Google API key"].ok is True


def test_run_diagnostics_reflects_ollama_reachability(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: True)

    checks = {c.name: c for c in run_diagnostics()}

    assert checks["Ollama (local models)"].ok is True
    assert "reachable" in checks["Ollama (local models)"].detail


def test_run_diagnostics_reports_foundry_extra_not_installed(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(runtime, "_foundry_extra_installed", lambda: False)

    checks = {c.name: c for c in run_diagnostics()}

    foundry = checks["Foundry (local from-scratch models)"]
    assert foundry.ok is False
    assert "not installed" in foundry.detail


def test_run_diagnostics_reports_a_real_discovered_checkpoint_bundle(monkeypatch, tmp_path):
    # A real bundle-shaped directory (empty files are enough --
    # discover_checkpoint_bundles only checks the three names exist).
    bundle_dir = tmp_path / "toy"
    bundle_dir.mkdir()
    for name in ("model.pt", "tokenizer.json", "config.json"):
        (bundle_dir / name).write_text("{}")

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SARVA_FOUNDRY_CHECKPOINTS", str(tmp_path))
    monkeypatch.setattr(runtime, "_foundry_extra_installed", lambda: True)

    checks = {c.name: c for c in run_diagnostics()}

    foundry = checks["Foundry (local from-scratch models)"]
    assert foundry.ok is True
    assert "toy" in foundry.detail


def test_doctor_cli_command_runs_and_prints_every_check_name(monkeypatch):
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    for name in (
        "Anthropic API key",
        "OpenAI API key",
        "Google API key",
        "Ollama",
        "Foundry",
        "Web UI",
    ):
        assert name in result.stdout


def test_doctor_cli_never_swallows_bracketed_text_as_rich_markup(monkeypatch):
    # Real regression: a detail string containing literal "[foundry]"
    # was silently eaten by Rich's markup parser (interpreted as an
    # invalid style tag) before `escape()` was added -- this pins that
    # it stays on screen.
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(runtime, "_foundry_extra_installed", lambda: False)

    result = runner.invoke(app, ["doctor"])

    assert "sarva[foundry]" in result.stdout
