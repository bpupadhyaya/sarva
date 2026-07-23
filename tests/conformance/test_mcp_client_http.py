"""Conformance tests for the MCP client's Streamable HTTP transport --
closing the gap `mcp_client.py`'s own module docstring used to name as
real, deferred scope.

`tests/fixtures/mcp_http_echo_server.py` is a real MCP server, launched
as a real subprocess speaking real MCP-over-HTTP on a real (locally
bound) port -- not a mock of the protocol, same bar
`test_mcp_client.py`'s stdio tests already hold. Mirrors that file's
test coverage exactly (list tools, a real round trip, real error
propagation, a real AgentLoop run) so the two transports are proven
equivalent from the caller's point of view, not just independently
plausible.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from sarva.agent.loop import AgentLoop
from sarva.agent.tools import ToolContext
from sarva.mcp_client import connect_http_mcp_server, list_mcp_tools
from sarva.multimodal.content import Modality, ToolCallBlock
from sarva.providers.base import ModelCapabilities, ModelCost, ModelInfo
from sarva.providers.mock import MockProvider, ScriptedTurn
from sarva.providers.registry import Registry, Router, TaskClass

_HTTP_ECHO_SERVER = str(Path(__file__).parent.parent / "fixtures" / "mcp_http_echo_server.py")


def _free_port() -> int:
    # Binding to port 0 asks the OS for a genuinely free ephemeral port --
    # avoids CI collisions from a hardcoded port across parallel jobs.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_accepting_connections(port: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"MCP test server never started listening on port {port}")


@pytest.fixture(scope="module")
def mcp_http_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, _HTTP_ECHO_SERVER],
        env={"SARVA_TEST_MCP_PORT": str(port), "PATH": __import__("os").environ["PATH"]},
    )
    try:
        _wait_until_accepting_connections(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _connect(url: str):
    return connect_http_mcp_server(url)


@pytest.mark.asyncio
async def test_list_tools_reflects_the_real_server_over_http(mcp_http_url):
    async with _connect(mcp_http_url) as session:
        tools = await list_mcp_tools(session)
    names = {t.spec.name for t in tools}
    assert names == {"echo", "fail"}
    echo = next(t for t in tools if t.spec.name == "echo")
    assert echo.spec.input_schema["properties"]["text"]["type"] == "string"


@pytest.mark.asyncio
async def test_call_tool_round_trip_over_http(mcp_http_url, tmp_path):
    async with _connect(mcp_http_url) as session:
        tools = await list_mcp_tools(session)
        echo = next(t for t in tools if t.spec.name == "echo")
        ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
        result = await echo.run({"text": "hello over real http"}, ctx)

    assert not result.is_error
    assert result.content[0].text == "hello over real http"


@pytest.mark.asyncio
async def test_call_tool_error_propagates_over_http(mcp_http_url, tmp_path):
    async with _connect(mcp_http_url) as session:
        tools = await list_mcp_tools(session)
        fail = next(t for t in tools if t.spec.name == "fail")
        ctx = ToolContext(workdir=str(tmp_path), run_dir=str(tmp_path / "run"))
        result = await fail.run({"reason": "deliberate http failure"}, ctx)

    assert result.is_error
    assert "deliberate http failure" in result.content[0].text


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
async def test_http_mcp_tool_reaches_a_real_agent_loop_run(mcp_http_url, tmp_path):
    """Proves the HTTP-sourced wrapper is a genuine `Tool` the loop can
    drive end to end, identically to the stdio case -- not just that an
    HTTP round trip happens somewhere in isolation."""
    run_root = tmp_path / "runs"
    call = ToolCallBlock(id="a", name="echo", arguments={"text": "via http mcp"})
    provider = MockProvider(script=[ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="done")])

    async with _connect(mcp_http_url) as session:
        tools = await list_mcp_tools(session)
        loop = AgentLoop(
            router=_text_only_router(),
            providers={"mock": provider},
            tools=tools,
            run_root=str(run_root),
        )
        events = [e async for e in loop.run("echo something through HTTP MCP")]

    finished = [e for e in events if e.type == "tool_finished"]
    assert len(finished) == 1
    assert not finished[0].result.is_error
    assert finished[0].result.content[0].text == "via http mcp"


@pytest.mark.asyncio
async def test_connect_http_mcp_server_passes_through_custom_headers(mcp_http_url):
    # The echo server doesn't check auth, so this can't prove a header is
    # *enforced* -- but it proves the call genuinely accepts headers and
    # completes a real request with them attached, not that the
    # parameter is silently ignored.
    async with connect_http_mcp_server(mcp_http_url, headers={"X-Test": "value"}) as session:
        tools = await list_mcp_tools(session)
    assert {t.spec.name for t in tools} == {"echo", "fail"}


@pytest.mark.asyncio
async def test_http_server_is_reachable_via_plain_http_before_mcp_handshake(mcp_http_url):
    # Confirms the port really is a live HTTP server (not e.g. a process
    # that merely opened the socket but never finished starting) before
    # any MCP-specific assertion runs -- isolates "server didn't start"
    # from "MCP protocol handshake failed" as distinct failure modes.
    base = mcp_http_url.rsplit("/mcp", 1)[0]
    async with httpx.AsyncClient() as client:
        # A bare GET to the MCP endpoint without a session is expected to
        # be REJECTED by the protocol (it requires the Streamable HTTP
        # handshake), not to hang or connection-refuse -- any HTTP
        # response at all proves the server is genuinely listening.
        response = await client.get(f"{base}/mcp", timeout=5.0)
    assert response.status_code < 600
