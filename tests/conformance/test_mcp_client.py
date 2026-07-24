"""Conformance tests for the MCP client (spec §3.5's "MCP client support").

`tests/fixtures/mcp_echo_server.py` is a real MCP server, launched as a
real subprocess speaking real stdio JSON-RPC -- not a mock of the
protocol. This is the same bar the rest of the project holds: prove the
real round trip, not a hand-waved approximation of it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
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
    assert names == {"echo", "fail", "env_var"}
    echo = next(t for t in tools if t.spec.name == "echo")
    assert echo.spec.input_schema["properties"]["text"]["type"] == "string"


@pytest.mark.asyncio
async def test_call_tool_round_trip(tmp_path):
    async with _connect() as session:
        tools = await list_mcp_tools(session)
        echo = next(t for t in tools if t.spec.name == "echo")
        ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
        result = await echo.run({"text": "hello from the real client"}, ctx)

    assert not result.is_error
    assert result.content[0].text == "hello from the real client"


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
