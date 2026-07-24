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
import stat
import sys
from contextlib import asynccontextmanager

import pytest
import sarva.cli as cli_module
import sarva.config as config_module
import sarva.memory.session as session_module
import sarva.runtime as runtime
from sarva.audio import stt_extra_installed, tts_engine_available
from sarva.cli import _parse_mcp_headers, app
from sarva.memory.session import SessionStore
from sarva.providers.base import ToolSpec
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


def test_chat_with_an_invalid_session_name_fails_cleanly_instead_of_a_raw_traceback(
    monkeypatch, tmp_path
):
    # A real bug found by actually running `sarva chat --session "bad
    # name!"`: SessionStore._sanitize() raises a plain ValueError for
    # any name outside [A-Za-z0-9_-], and _chat never caught it -- a
    # raw Python traceback and exit 1, the same "unhandled exception
    # where a clean error belongs" bug class already fixed for
    # eval/distill's --model.
    _clear_provider_env(monkeypatch)
    _isolate_sessions(monkeypatch, tmp_path)

    result = runner.invoke(app, ["chat", "hi", "--session", "bad name!"])

    assert result.exit_code != 0
    assert "invalid session name" in result.stdout
    assert "Traceback" not in result.stdout


def test_run_with_an_invalid_session_name_fails_cleanly_instead_of_a_raw_traceback(
    monkeypatch, tmp_path
):
    _clear_provider_env(monkeypatch)
    _isolate_sessions(monkeypatch, tmp_path)

    result = runner.invoke(
        app, ["run", "hi", "--workdir", str(tmp_path), "--session", "bad name!", "--auto"]
    )

    assert result.exit_code != 0
    assert "invalid session name" in result.stdout
    assert "Traceback" not in result.stdout


def test_sessions_clear_with_an_invalid_name_fails_cleanly(monkeypatch, tmp_path):
    _isolate_sessions(monkeypatch, tmp_path)

    result = runner.invoke(app, ["sessions", "clear", "bad name!"])

    assert result.exit_code != 0
    assert "invalid session name" in result.stdout


def test_chat_with_model_forces_that_exact_model(monkeypatch):
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["chat", "hello", "--model", "mock"])

    assert result.exit_code == 0
    assert "[mock] received: hello" in result.stdout


def test_chat_with_an_unknown_model_fails_cleanly_with_a_clear_message_and_nonzero_exit(
    monkeypatch,
):
    # The real safety property --model exists to guarantee: a typo'd
    # model id must be a clear, visible failure -- never a silent
    # substitution for a different model, and (this pins the CLI half of
    # that guarantee) never a bare Python traceback either.
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["chat", "hello", "--model", "not-a-real-model"])

    assert result.exit_code != 0
    assert "unknown model 'not-a-real-model'" in result.stdout
    assert "run ended: failed" in result.stdout


def test_chat_with_an_image_of_the_wrong_type_fails_cleanly(tmp_path, monkeypatch):
    _clear_provider_env(monkeypatch)
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("hello")

    result = runner.invoke(app, ["chat", "look at this", "--image", str(not_an_image)])

    assert result.exit_code != 0
    assert "cannot determine an image media type" in result.output


def test_run_with_an_image_of_the_wrong_type_fails_cleanly(tmp_path, monkeypatch):
    # sarva run gained --image after ws_chat did (the CLI's own "run"
    # mirrors /ws/chat the way "chat" mirrors /chat) -- mirrors the chat
    # test above so both commands hold the same failure-mode guarantee.
    _clear_provider_env(monkeypatch)
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("hello")

    result = runner.invoke(app, ["run", "look at this", "--image", str(not_an_image), "--auto"])

    assert result.exit_code != 0
    assert "cannot determine an image media type" in result.output


def test_run_with_a_valid_image_completes_successfully(tmp_path, monkeypatch):
    _clear_provider_env(monkeypatch)
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nreal enough bytes for this test")

    result = runner.invoke(
        app, ["run", "what's in this image?", "--image", str(image_path), "--auto"]
    )

    assert result.exit_code == 0
    assert "[mock] received: what's in this image?" in result.stdout


def test_run_with_model_forces_that_exact_model(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)

    result = runner.invoke(
        app, ["run", "do something", "--workdir", str(tmp_path), "--model", "mock", "--auto"]
    )

    assert result.exit_code == 0
    assert "[mock] received: do something" in result.stdout


def test_run_with_an_unknown_model_fails_cleanly_with_a_clear_message_and_nonzero_exit(
    monkeypatch, tmp_path
):
    _clear_provider_env(monkeypatch)

    result = runner.invoke(
        app,
        ["run", "do something", "--workdir", str(tmp_path), "--model", "nope", "--auto"],
    )

    assert result.exit_code != 0
    assert "unknown model 'nope'" in result.stdout
    assert "run ended: failed" in result.stdout


def test_parse_mcp_headers_builds_a_dict_from_name_colon_value_strings():
    assert _parse_mcp_headers(["Authorization: Bearer abc123", "X-Custom:  spaced  "]) == {
        "Authorization": "Bearer abc123",
        "X-Custom": "spaced",
    }


def test_parse_mcp_headers_on_an_empty_list_returns_an_empty_dict():
    assert _parse_mcp_headers([]) == {}


def test_parse_mcp_headers_rejects_an_entry_with_no_colon():
    with pytest.raises(Exception, match="invalid --mcp-header"):
        _parse_mcp_headers(["not-a-header"])


def test_run_mcp_header_actually_reaches_connect_http_mcp_server(monkeypatch, tmp_path):
    # connect_http_mcp_server() has always accepted a headers dict
    # (test_mcp_client_http.py's own test proves that at the library
    # level) -- what was missing, and what this pins, is that
    # --mcp-header on the CLI actually gets parsed and threaded through
    # to that call, not just accepted and silently dropped.
    _clear_provider_env(monkeypatch)
    captured = {}

    class _FakeSession:
        async def list_tools(self):
            class _Result:
                tools = []

            return _Result()

    @asynccontextmanager
    async def fake_connect_http_mcp_server(url, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        yield _FakeSession()

    monkeypatch.setattr(cli_module, "connect_http_mcp_server", fake_connect_http_mcp_server)

    result = runner.invoke(
        app,
        [
            "run",
            "do something",
            "--workdir",
            str(tmp_path),
            "--mcp-server",
            "https://example.invalid/mcp",
            "--mcp-header",
            "Authorization: Bearer test-token",
            "--auto",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["url"] == "https://example.invalid/mcp"
    assert captured["headers"] == {"Authorization": "Bearer test-token"}


def test_run_mcp_tool_names_with_markup_characters_are_escaped_not_swallowed(monkeypatch, tmp_path):
    # Tool names come from the connected MCP server's own response -- for
    # an http(s):// server that's a remote, untrusted source. A real bug
    # found by auditing this file for the same "unescaped external text"
    # class already fixed elsewhere (doctor's detail text, transcribe's
    # error message): this line printed tool names with no escape() at
    # all, so a malicious/buggy server naming a tool with embedded Rich
    # markup could spoof this project's own terminal output. Pinned here
    # without a real MCP round trip -- list_mcp_tools is the one function
    # that turns a session's raw response into Tool objects, so patching
    # it directly is the precise unit for what this test checks.
    _clear_provider_env(monkeypatch)

    @asynccontextmanager
    async def fake_connect_stdio_mcp_server(command, args=None):
        yield object()

    class _FakeTool:
        spec = ToolSpec(
            name="[red]FAKE ERROR[/red] normal_tool",
            description="a fake tool for this test",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, args, ctx):
            raise AssertionError("never actually called in this test")

    async def fake_list_mcp_tools(session):
        return [_FakeTool()]

    monkeypatch.setattr(cli_module, "connect_stdio_mcp_server", fake_connect_stdio_mcp_server)
    monkeypatch.setattr(cli_module, "list_mcp_tools", fake_list_mcp_tools)

    result = runner.invoke(
        app,
        ["run", "do something", "--workdir", str(tmp_path), "--mcp-server", "fake-cmd", "--auto"],
    )

    assert result.exit_code == 0, result.output
    # The raw brackets and their contents must survive verbatim in the
    # captured output -- not interpreted as a real [red]...[/red] tag
    # (which would instead show only "FAKE ERROR normal_tool", styled).
    assert "[red]FAKE ERROR[/red] normal_tool" in result.stdout


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


def test_eval_with_an_unknown_model_fails_cleanly_instead_of_a_raw_traceback(monkeypatch):
    # A real bug found by actually running `sarva eval --model
    # bogus-id`: Registry.get() is called directly here (eval never
    # goes through Router.pick(), since it needs no modality/
    # availability routing), with no error handling at all -- an
    # unknown id crashed with a raw KeyError traceback instead of the
    # same clean message chat/run's --model already give.
    _clear_provider_env(monkeypatch)

    result = runner.invoke(app, ["eval", "--model", "not-a-real-model"])

    assert result.exit_code != 0
    assert "unknown model 'not-a-real-model'" in result.stdout
    assert "KeyError" not in result.stdout


def test_distill_with_an_unknown_model_fails_cleanly_instead_of_a_raw_traceback(
    monkeypatch, tmp_path
):
    _clear_provider_env(monkeypatch)
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("hi\n")

    result = runner.invoke(
        app,
        [
            "distill",
            str(prompts_file),
            "--model",
            "not-a-real-model",
            "--out",
            str(tmp_path / "out.jsonl"),
        ],
    )

    assert result.exit_code != 0
    assert "unknown model 'not-a-real-model'" in result.stdout
    assert "KeyError" not in result.stdout
    assert not (tmp_path / "out.jsonl").exists()


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


def _isolate_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "config.json")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_config_set_writes_and_config_show_reflects_it(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    set_result = runner.invoke(app, ["config", "set", "--anthropic-api-key", "sk-ant-real-test"])
    assert set_result.exit_code == 0
    assert "ANTHROPIC_API_KEY" in set_result.stdout
    # The actual key value must never appear in terminal output.
    assert "sk-ant-real-test" not in set_result.stdout

    show_result = runner.invoke(app, ["config", "show"])
    assert "ANTHROPIC_API_KEY" in show_result.stdout
    assert "set" in show_result.stdout
    assert "saved config file" in show_result.stdout
    assert "sk-ant-real-test" not in show_result.stdout
    # The other three keys were never set.
    assert "OPENAI_API_KEY" in show_result.stdout
    assert "not set" in show_result.stdout


def test_config_show_prefers_a_real_env_var_over_the_saved_file(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    runner.invoke(app, ["config", "set", "--anthropic-api-key", "sk-from-file"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    result = runner.invoke(app, ["config", "show"])

    assert "environment variable" in result.stdout
    assert "saved config file" not in result.stdout


def test_config_set_with_no_keys_fails_cleanly(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    result = runner.invoke(app, ["config", "set"])

    assert result.exit_code != 0
    assert "nothing to save" in result.stdout
    assert not (tmp_path / "config.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="owner-only permissions are POSIX-only")
def test_config_set_writes_with_owner_only_permissions(monkeypatch, tmp_path):
    # sarva config set is a real second caller of save_config() beyond
    # the desktop app's POST /config -- proves the CLI path gets the
    # same real security fix (see sarva.config's own docstring), not
    # just the server one.
    _isolate_config(monkeypatch, tmp_path)

    runner.invoke(app, ["config", "set", "--gemini-api-key", "test-key"])

    mode = stat.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600


def test_config_unset_removes_the_saved_key_and_leaves_others_alone(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    runner.invoke(
        app,
        ["config", "set", "--anthropic-api-key", "sk-a", "--openai-api-key", "sk-b"],
    )

    result = runner.invoke(app, ["config", "unset", "--anthropic-api-key"])

    assert result.exit_code == 0
    assert "removed ANTHROPIC_API_KEY" in result.stdout
    show = runner.invoke(app, ["config", "show"])
    assert "ANTHROPIC_API_KEY    not set" in show.stdout
    assert "OPENAI_API_KEY       set (saved config file)" in show.stdout


def test_config_unset_a_key_that_was_never_saved_is_a_clean_no_op(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    result = runner.invoke(app, ["config", "unset", "--gemini-api-key"])

    assert result.exit_code == 0
    assert "nothing to do" in result.stdout
    assert not (tmp_path / "config.json").exists()


def test_config_unset_never_touches_a_real_environment_variable(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    runner.invoke(app, ["config", "unset", "--anthropic-api-key"])

    # unset only ever edits the saved file -- get_env must still resolve
    # to the real env var exactly as before.
    from sarva.config import get_env

    assert get_env("ANTHROPIC_API_KEY", path=tmp_path / "config.json") == "sk-from-env"


def test_config_unset_with_no_flags_fails_cleanly(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    result = runner.invoke(app, ["config", "unset"])

    assert result.exit_code != 0
    assert "nothing to remove" in result.stdout


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


@pytest.mark.skipif(
    not (tts_engine_available() and stt_extra_installed()),
    reason="needs a local TTS engine and sarva[audio] (faster-whisper)",
)
def test_transcribe_round_trips_real_synthesized_speech(tmp_path):
    # The strongest real proof: synthesize real speech via the local TTS
    # engine, write it to a real file, run it through the actual
    # `transcribe` command (not sarva.audio.transcribe() called
    # directly), and check real words come back -- the same round trip
    # test_audio.py already proves at the library level, now proven
    # through the CLI surface that didn't exist before this command.
    from sarva.audio import synthesize

    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(synthesize("the assistant can now hear and speak"))

    result = runner.invoke(app, ["transcribe", str(audio_path)])

    assert result.exit_code == 0
    lowered = result.stdout.lower()
    assert "hear" in lowered
    assert "speak" in lowered


@pytest.mark.skipif(not stt_extra_installed(), reason="this test needs sarva[audio] installed")
def test_transcribe_uses_the_requested_model_size(tmp_path, monkeypatch):
    # Hermetic proof the --model-size flag actually reaches
    # sarva.audio.transcribe(), without needing a real audio round trip.
    import sarva.audio as audio_module

    captured = {}

    def fake_transcribe(audio_bytes, model_size="tiny"):
        captured["model_size"] = model_size
        return "fake transcript"

    monkeypatch.setattr(audio_module, "transcribe", fake_transcribe)
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"not real audio, never decoded by the fake")

    result = runner.invoke(app, ["transcribe", str(audio_path), "--model-size", "base"])

    assert result.exit_code == 0
    assert "fake transcript" in result.stdout
    assert captured["model_size"] == "base"


def test_transcribe_fails_cleanly_without_the_audio_extra(tmp_path, monkeypatch):
    import sarva.audio as audio_module

    monkeypatch.setattr(audio_module, "stt_extra_installed", lambda: False)
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"irrelevant, never reached")

    result = runner.invoke(app, ["transcribe", str(audio_path)])

    assert result.exit_code == 1
    assert "sarva[audio]" in result.stdout


def test_version_flag_prints_the_real_installed_version_and_exits():
    from importlib.metadata import version

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"sarva {version('sarva')}" in result.stdout
