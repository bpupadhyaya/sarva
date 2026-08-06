"""Conformance tests for the MCP client (spec §3.5's "MCP client support").

`tests/fixtures/mcp_echo_server.py` is a real MCP server, launched as a
real subprocess speaking real stdio JSON-RPC -- not a mock of the
protocol. This is the same bar the rest of the project holds: prove the
real round trip, not a hand-waved approximation of it.
"""

from __future__ import annotations

import shutil
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import sarva.mcp_client as mcp_client_module
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import ToolContext
from sarva.mcp_client import connect_stdio_mcp_server, list_mcp_tools
from sarva.multimodal.content import Modality, ToolCallBlock
from sarva.providers.base import ModelCapabilities, ModelCost, ModelInfo
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass

_ECHO_SERVER = str(Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")


def _connect():
    return connect_stdio_mcp_server(sys.executable, args=[_ECHO_SERVER])


@pytest.mark.asyncio
async def test_list_tools_reflects_the_real_server():
    async with _connect() as session:
        tools = await list_mcp_tools(session)
    names = {t.spec.name for t in tools}
    assert names == {"echo", "structured_only", "fail", "env_var"}
    echo = next(t for t in tools if t.spec.name == "echo")
    assert echo.spec.input_schema["properties"]["text"]["type"] == "string"


@pytest.mark.asyncio
async def test_connect_stdio_mcp_server_times_out_instead_of_hanging_forever(monkeypatch):
    # A real bug found by a fresh-eyes sweep: ClientSession's own
    # read_timeout_seconds constructor parameter defaults to None --
    # unbounded -- and connect_stdio_mcp_server never set it. A real
    # subprocess that spawns successfully (so no FileNotFoundError) but
    # never speaks MCP at all -- a slow-starting or misbehaving server,
    # not an adversarial one -- used to hang session.initialize()
    # forever with no recovery. Confirmed live before this fix with a
    # bare ClientSession over a stream that never delivers a response.
    # A real subprocess that just sleeps, never speaking MCP, stands in
    # for that same unresponsive-server shape through the real public
    # connect_stdio_mcp_server entry point.
    monkeypatch.setattr(mcp_client_module, "_MCP_READ_TIMEOUT", timedelta(seconds=0.5))
    # The decisive property: this must raise (and quickly -- pytest-
    # timeout-free, so a regression here would hang the whole test
    # suite, not just fail one test) rather than hang forever. cli.py's
    # own real call site already unwraps a single-exception
    # ExceptionGroup to surface the real McpError's clear "Timed out
    # while waiting..." message -- checked here directly against
    # whichever shape the SDK raises, exception group or not.
    with pytest.raises((Exception, BaseExceptionGroup)) as exc_info:
        async with connect_stdio_mcp_server(
            sys.executable, args=["-c", "import time; time.sleep(60)"]
        ):
            pass
    raised = exc_info.value
    while isinstance(raised, BaseExceptionGroup) and len(raised.exceptions) == 1:
        raised = raised.exceptions[0]
    assert "time" in str(raised).lower()


@pytest.mark.asyncio
async def test_call_tool_round_trip(tmp_path):
    async with _connect() as session:
        tools = await list_mcp_tools(session)
        echo = next(t for t in tools if t.spec.name == "echo")
        ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
        result = await echo.run({"text": "hello from the real client"}, ctx)

    assert not result.is_error
    assert result.content[0].text == "hello from the real client"


@pytest.mark.asyncio
async def test_structured_content_is_not_silently_dropped(tmp_path):
    # A real bug found by a fresh-eyes sweep: CallToolResult has TWO
    # separate result fields on the wire -- `content` (unstructured) and
    # `structuredContent` (a JSON object validated against the tool's
    # own declared outputSchema) -- and only `content` was ever read
    # here. The MCP spec only requires a server to include
    # structuredContent; duplicating it into `content` as backward-
    # compat text is a SHOULD, not a MUST -- a server built on the
    # lower-level Server.call_tool decorator (a legitimate, documented
    # way to build an MCP server) can legitimately send `content=[]`
    # alongside a real structuredContent, exactly what this fixture's
    # own `structured_only` tool does (returning a real CallToolResult
    # directly, FastMCP's own documented escape hatch, to produce this
    # exact shape against a REAL server, not a mock). Confirmed live
    # before this fix: the model would have seen a clean, empty success
    # with the actual computed answer nowhere in it, no error, no signal
    # anything was lost -- directly contradicting this same module's own
    # _convert_content fallback, whose whole point is never silently
    # dropping content it can't fully translate.
    async with _connect() as session:
        tools = await list_mcp_tools(session)
        structured_only = next(t for t in tools if t.spec.name == "structured_only")
        ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
        result = await structured_only.run({"a": 40, "b": 2}, ctx)

    assert not result.is_error
    assert any('"sum": 42' in b.text for b in result.content)


async def _env_var_result(env: dict[str, str] | None, tmp_path) -> str:
    # A real gap flagged by two separate Explore-agent sweeps: `env` has
    # been part of connect_stdio_mcp_server's signature since MCP
    # support shipped, but nothing proved it actually reached a real
    # spawned subprocess's environment, and no CLI flag threaded one
    # through at all until this milestone (see cli.py's --mcp-env /
    # _parse_mcp_env). Proven here against a real subprocess, not a
    # mock -- this file's own docstring states that bar for everything
    # in it.
    async with connect_stdio_mcp_server(sys.executable, args=[_ECHO_SERVER], env=env) as session:
        tools = await list_mcp_tools(session)
        env_var_tool = next(t for t in tools if t.spec.name == "env_var")
        ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
        result = await env_var_tool.run({"name": "SARVA_TEST_ENV_VAR"}, ctx)
    assert not result.is_error
    return result.content[0].text


@pytest.mark.asyncio
async def test_connect_stdio_mcp_server_env_value_reaches_the_real_subprocess(tmp_path):
    value = await _env_var_result(
        {"SARVA_TEST_ENV_VAR": "reached-the-real-child-process"}, tmp_path
    )
    assert value == "reached-the-real-child-process"


@pytest.mark.asyncio
async def test_connect_stdio_mcp_server_with_no_env_leaves_the_var_unset(tmp_path):
    value = await _env_var_result(None, tmp_path)
    assert value == "MISSING"


@pytest.mark.asyncio
async def test_every_mcp_tool_is_marked_destructive_regardless_of_what_it_does():
    # A real bug found by giving this module its own fresh-eyes sweep:
    # ToolSpec.destructive defaults to False, and McpToolAdapter never
    # set it, so every MCP-provided tool -- arbitrary, remote, unaudited
    # code -- silently bypassed AgentLoop's own confirm-before-
    # destructive-action gate. Every tool here (echo/fail/env_var) is
    # genuinely non-destructive in this fixture server's own
    # implementation, which is exactly the point: MCP's own
    # `destructiveHint` annotation is explicitly documented as untrusted
    # ("clients should never make tool use decisions based on
    # ToolAnnotations received from untrusted servers"), so this
    # deliberately does NOT read that hint -- every MCP tool is
    # destructive unconditionally, regardless of what it claims or
    # actually does.
    async with _connect() as session:
        tools = await list_mcp_tools(session)
    assert tools  # sanity: the fixture server really did report tools
    for tool in tools:
        assert tool.spec.destructive is True, tool.spec.name


@pytest.mark.asyncio
async def test_mcp_tool_call_goes_through_the_destructive_confirmation_gate(tmp_path):
    # The real, end-to-end proof the flag above actually does something:
    # a confirm policy that refuses everything must be consulted before
    # an MCP tool call runs, and the loop must honor a refusal exactly
    # as it would for any other destructive builtin.
    run_root = tmp_path / "runs"
    call = ToolCallBlock(id="a", name="echo", arguments={"text": "should be blocked"})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done")])

    async def refuse_everything(call):
        return False

    async with _connect() as session:
        tools = await list_mcp_tools(session)
        loop = AgentLoop(
            router=_text_only_router(),
            providers={"mock": provider},
            tools=tools,
            confirm=refuse_everything,
            run_root=str(run_root),
        )
        events = [e async for e in loop.run("echo something through MCP")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert len(finished) == 1
    assert finished[0].result.is_error is True
    assert "declined" in finished[0].result.content[0].text.lower()
    shutil.rmtree(run_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_call_tool_error_propagates(tmp_path):
    async with _connect() as session:
        tools = await list_mcp_tools(session)
        fail = next(t for t in tools if t.spec.name == "fail")
        ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
        result = await fail.run({"reason": "deliberate failure"}, ctx)

    assert result.is_error
    assert "deliberate failure" in result.content[0].text


def _text_only_router() -> Router:
    model = ModelInfo(
        id="text-only",
        provider="mock",
        display_name="Text Only Mock",
        capabilities=ModelCapabilities(
            modalities_in={Modality.TEXT},
            modalities_out={Modality.TEXT},
            tool_use=True,
            thinking=False,
            context_window=100_000,
            max_output=8_000,
        ),
        cost=ModelCost(),
    )
    registry = Registry(models={model.id: model})
    return Router(registry, routing={TaskClass.MAIN: ["text-only"]}, available={"text-only"})


@pytest.mark.asyncio
async def test_mcp_tool_reaches_a_real_agent_loop_run(tmp_path):
    """Proves the wrapper is a genuine `Tool` the loop can drive end to
    end -- not just an object with the right shape in isolation."""
    run_root = tmp_path / "runs"
    call = ToolCallBlock(id="a", name="echo", arguments={"text": "via the agent loop"})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done")])

    async with _connect() as session:
        tools = await list_mcp_tools(session)
        loop = AgentLoop(
            router=_text_only_router(),
            providers={"mock": provider},
            tools=tools,
            run_root=str(run_root),
        )
        events = [e async for e in loop.run("echo something through MCP")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert len(finished) == 1
    assert not finished[0].result.is_error
    assert finished[0].result.content[0].text == "via the agent loop"
    shutil.rmtree(run_root, ignore_errors=True)
