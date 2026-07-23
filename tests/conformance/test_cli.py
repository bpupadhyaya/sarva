"""Conformance tests for the `sarva` CLI commands themselves — `chat`,
`run`, `models`, `eval`, `distill`, `sessions list`/`clear`. Until now
only `doctor` had `typer.testing.CliRunner` coverage (confirmed by
`grep -rln "CliRunner" tests/` returning exactly one file) — every other
command was only ever exercised indirectly, through the library
functions it wraps, never through `app` itself the way a real user
invokes it. These tests run the actual Typer `app`, zero-config (Mock
provider only), the same "always works with no API keys" guarantee
`sarva.cli`'s own module docstring makes."""

from __future__ import annotations

import json

import pytest
import sarva.memory.session as session_module
import sarva.runtime as runtime
from sarva.audio import tts_engine_available
from sarva.cli import app
from sarva.memory.session import SessionStore
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
    # "Zero-config" here must mean the same thing regardless of what else
    # happens to be running on the machine executing this suite -- a real
    # local Ollama server (reachable, but without the specific model this
    # test expects pulled) would otherwise make the real router prefer it
    # over falling back to Mock, a real test-isolation bug this session
    # found by actually installing and running Ollama to verify that
    # adapter live (see BUILD-JOURNAL.md).
    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: False)


def _isolate_sessions(monkeypatch, tmp_path) -> None:
    # SessionStore() always resolves the module-level DEFAULT_SESSIONS_DIR
    # at construction time -- patching it here keeps every CLI command
    # under test from touching the real ~/.sarva/sessions on this machine.
    monkeypatch.setattr(session_module, "DEFAULT_SESSIONS_DIR", tmp_path / "sessions")


def test_chat_with_no_provider_configured_routes_to_mock_and_echoes(monkeypatch):
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["chat", "hello there"])

    assert result.exit_code == 0
    assert "[mock] received: hello there" in result.stdout


def test_chat_with_session_persists_the_transcript(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    _isolate_sessions(monkeypatch, tmp_path)

    result = runner.invoke(app, ["chat", "remember this", "--session", "my-convo"])

    assert result.exit_code == 0
    saved = SessionStore(tmp_path / "sessions").load("my-convo")
    assert len(saved) == 2  # user turn + assistant turn
    assert saved[0].text() == "remember this"


def test_chat_with_an_image_of_the_wrong_type_fails_cleanly(tmp_path, monkeypatch):
    _clear_provider_env(monkeypatch)
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("hello")

    result = runner.invoke(app, ["chat", "look at this", "--image", str(not_an_image)])

    assert result.exit_code != 0
    assert "cannot determine an image media type" in result.output


def test_run_with_mock_provider_completes_with_no_tool_calls(monkeypatch, tmp_path):
    # MockProvider's unscripted default turn never issues tool calls, so
    # this proves the CLI wires AgentLoop + BUILTIN_TOOLS end to end
    # without needing a scripted provider to actually exercise a tool.
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["run", "do something", "--workdir", str(tmp_path), "--auto"])

    assert result.exit_code == 0
    assert "[mock] received: do something" in result.stdout


def test_run_with_session_only_persists_on_a_completed_run(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    _isolate_sessions(monkeypatch, tmp_path)

    result = runner.invoke(
        app, ["run", "a task", "--workdir", str(tmp_path), "--auto", "--session", "run-sess"]
    )

    assert result.exit_code == 0
    saved = SessionStore(tmp_path / "sessions").load("run-sess")
    assert len(saved) == 2


def test_models_lists_mock_as_available(monkeypatch):
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert "mock" in result.stdout
    assert "[x]" in result.stdout or "x]" in result.stdout


def test_eval_grades_the_mock_provider_against_the_arithmetic_benchmark(monkeypatch):
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["eval", "--model", "mock"])

    assert result.exit_code == 0
    assert "mock" in result.stdout
    # Mock's echo response is never a correct arithmetic answer -- the
    # honest result is 0%, same no-fabrication discipline the eval
    # harness chapter established elsewhere in this project. Checking
    # "0/10" (the printed correct/total count), not "0%" -- a real bug
    # found while fixing the harness's own grading logic: "0%" is a
    # substring of "30%", "10%", "100%", so this assertion would have
    # silently passed even if the real accuracy weren't 0%, which for a
    # while it genuinely wasn't (see contains_match's own docstring).
    assert "0/10" in result.stdout


def test_distill_writes_a_real_jsonl_file_from_mock_completions(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("what is 2+2?\nwhat is the capital of France?\n")
    out_file = tmp_path / "out.jsonl"

    result = runner.invoke(
        app, ["distill", str(prompts_file), "--model", "mock", "--out", str(out_file)]
    )

    assert result.exit_code == 0
    assert out_file.exists()
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["prompt"] == "what is 2+2?"
    assert records[0]["model"] == "mock"
    assert "[mock] received: what is 2+2?" in records[0]["completion"]


def test_distill_fails_cleanly_for_a_provider_that_is_not_configured(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("hi\n")

    result = runner.invoke(
        app,
        [
            "distill",
            str(prompts_file),
            "--model",
            "claude-opus-4-8",
            "--out",
            str(tmp_path / "out.jsonl"),
        ],
    )

    assert result.exit_code == 1
    assert "not configured" in result.stdout


def test_sessions_list_reports_nothing_saved_when_empty(monkeypatch, tmp_path):
    _isolate_sessions(monkeypatch, tmp_path)

    result = runner.invoke(app, ["sessions", "list"])

    assert result.exit_code == 0
    assert "no saved sessions" in result.stdout


def test_sessions_list_and_clear_reflect_a_real_saved_session(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    _isolate_sessions(monkeypatch, tmp_path)
    runner.invoke(app, ["chat", "hi", "--session", "keepsake"])

    list_result = runner.invoke(app, ["sessions", "list"])
    assert "keepsake" in list_result.stdout

    clear_result = runner.invoke(app, ["sessions", "clear", "keepsake"])
    assert clear_result.exit_code == 0
    assert "cleared session 'keepsake'" in clear_result.stdout

    final_list = runner.invoke(app, ["sessions", "list"])
    assert "keepsake" not in final_list.stdout


@pytest.mark.skipif(not tts_engine_available(), reason="no local TTS engine detected")
def test_speak_writes_a_real_audio_file(tmp_path):
    out_path = tmp_path / "out.wav"

    result = runner.invoke(app, ["speak", "hello from the command line", "--out", str(out_path)])

    assert result.exit_code == 0
    assert out_path.exists()
    assert out_path.read_bytes().startswith(b"RIFF")
    # Rich line-wraps long paths in the captured terminal output, so
    # check the pieces rather than one exact string.
    assert f"wrote {out_path.stat().st_size} bytes to" in result.stdout
    assert out_path.name in result.stdout


def test_speak_fails_cleanly_with_no_engine_available(tmp_path, monkeypatch):
    import sarva.audio as audio_module

    monkeypatch.setattr(audio_module.platform, "system", lambda: "Nonexistent")
    monkeypatch.setattr(audio_module.shutil, "which", lambda *_: None)

    result = runner.invoke(app, ["speak", "hello", "--out", str(tmp_path / "out.wav")])

    assert result.exit_code == 1
    assert "no local text-to-speech engine detected" in result.stdout
    assert not (tmp_path / "out.wav").exists()
