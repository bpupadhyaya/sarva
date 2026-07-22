# MCP client: plugging in the ecosystem's tools

`sarva.mcp_client` closes the last named gap in §3.5's tool runtime list:
"MCP client support so the ecosystem's tools plug in without
Sarva-specific glue." Any server that speaks the
[Model Context Protocol](https://modelcontextprotocol.io) — filesystem
access, GitHub, databases, whatever a third party ships — becomes a set
of ordinary Sarva `Tool`s with no code written per server.

## Why the official SDK, not a hand-rolled client

Sarva's provider adapters already use the official `anthropic`/`openai`/
`google-genai` SDKs rather than hand-rolled HTTP against each API. The
MCP client follows the same principle: `mcp.ClientSession` from the
official `mcp` Python SDK, not a from-scratch JSON-RPC implementation.
"From scratch" in this project is reserved for the foundry's model math
(tokenizer, transformer, training loop) — commodity protocol clients are
exactly the kind of substrate the provider layer already treats as
commodity.

## What's wired up

Only the **stdio transport**. Most real MCP servers today are local
processes launched with `npx`/`uvx`/a plain command — stdio covers that
majority, and it's the one transport genuinely verifiable offline: spawn
a real local subprocess, speak real MCP over its stdin/stdout, no
network call involved. HTTP/SSE transports are real, named, deferred
scope — not silently assumed covered.

```python
from sarva.mcp_client import connect_stdio_mcp_server, list_mcp_tools

async with connect_stdio_mcp_server("npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as session:
    tools = await list_mcp_tools(session)
    # tools is a list of ready-to-use Sarva `Tool`s — pass straight into AgentLoop
```

Each `McpToolAdapter` implements the same `Tool` protocol as every
built-in tool (`spec` + `async def run(args, ctx)`), so nothing downstream
— the agent loop, the confirmation policy, transcript logging — needs to
know or care that a given tool call is actually a round trip to a
subprocess speaking MCP instead of local Python code.

## CLI usage

```bash
sarva run "list the files in /tmp" \
    --mcp-server "npx -y @modelcontextprotocol/server-filesystem /tmp" \
    --auto
```

`--mcp-server` is repeatable — connect to several servers in one run.
Each server's tools are listed once at startup and merged into the same
tool registry as the built-ins (`read_file`, `write_file`, `remember`,
`recall_memory`, ...); the model sees one flat set of tools, with no way
to tell which ones came from where.

## Content conversion, honestly scoped

An MCP tool result can carry text, images, audio, resource links, or
embedded resources. Text and images convert directly to Sarva's
`TextBlock`/`ImageBlock`. Everything else reports its own declared MCP
content type rather than being silently dropped or raising — the same
"report only what's verifiably known" principle the multimodal degraders
use for content a layer can't fully consume.

## Verification

`tests/conformance/test_mcp_client.py` runs against a real MCP server
(`tests/fixtures/mcp_echo_server.py`, built with the official SDK's
`FastMCP`), launched as a genuine subprocess over genuine stdio — not a
mock of the protocol. It covers tool listing, a successful call, a
failing call (proving MCP error propagation reaches Sarva's
`ToolResultBlock.is_error`), and — the one that actually proves the
integration, not just the wrapper in isolation — a real `AgentLoop.run()`
driven by a `MockProvider` script that calls the MCP-backed tool and
gets back the exact text the real subprocess produced.
