"""sarva.cli — the `sarva` command-line entry point.

Zero-config by default: with no ANTHROPIC_API_KEY set, everything routes to
the offline MockProvider so `sarva chat "hello"` always works.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.markup import escape

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import BUILTIN_TOOLS, always_allow
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva.providers.base import TextDeltaEvent
from sarva.providers.mock import MockProvider
from sarva.providers.ollama_provider import OllamaProvider
from sarva.providers.registry import Registry, Router, load_routing

app = typer.Typer(help="Sarva — an open, all-in-one multimodal AGI tool.")
console = Console()

_DATA_DIR = Path(__file__).parent / "providers" / "data"
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _ollama_reachable() -> bool:
    """Best-effort, fast probe — never blocks CLI startup for more than a beat."""
    try:
        httpx.get(f"{_OLLAMA_HOST}/api/tags", timeout=0.3)
        return True
    except httpx.HTTPError:
        return False


def _build_router() -> Router:
    registry = Registry.load(_DATA_DIR / "models.yaml")
    routing = load_routing(_DATA_DIR / "routing.yaml")
    available = {"mock"}
    if os.environ.get("ANTHROPIC_API_KEY"):
        available |= {m.id for m in registry.all() if m.provider == "anthropic"}
    if _ollama_reachable():
        available |= {m.id for m in registry.all() if m.provider == "ollama"}
    return Router(registry, routing, available)


def _build_providers() -> dict[str, Any]:
    providers: dict[str, Any] = {"mock": MockProvider()}
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["anthropic"] = AnthropicProvider()
    if _ollama_reachable():
        providers["ollama"] = OllamaProvider(host=_OLLAMA_HOST)
    return providers


@app.command()
def chat(message: str = typer.Argument(..., help="Message to send.")) -> None:
    """One-shot chat — no tools, single turn."""
    asyncio.run(_chat(message))


async def _chat(message: str) -> None:
    loop = AgentLoop(
        router=_build_router(), providers=_build_providers(), tools=[], confirm=always_allow
    )
    async for event in loop.run(message):
        # Model output may itself contain "[", e.g. markdown links or
        # citations — never markup-parse text that came from the model.
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            console.print(event.event.text, end="", markup=False)
        if event.type == "run_done":
            console.print()
            if event.state != "done":
                console.print(f"[red]run ended: {event.state}[/red]")


@app.command()
def run(
    task: str = typer.Argument(..., help="Task for the agent to complete."),
    workdir: str = typer.Option(".", help="Working directory for file/shell tools."),
    auto: bool = typer.Option(
        False, "--auto", help="Auto-approve destructive tools (no confirmation prompts)."
    ),
) -> None:
    """Run the agent loop with built-in tools (files, shell)."""
    asyncio.run(_run(task, workdir, auto))


async def _confirm_prompt(call: Any) -> bool:
    return typer.confirm(f"Allow {call.name}({call.arguments})?")


async def _run(task: str, workdir: str, auto: bool) -> None:
    confirm = always_allow if auto else _confirm_prompt
    loop = AgentLoop(
        router=_build_router(),
        providers=_build_providers(),
        tools=BUILTIN_TOOLS,
        confirm=confirm,
        workdir=workdir,
    )
    async for event in loop.run(task):
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            console.print(event.event.text, end="", markup=False)
        elif event.type == "tool_started":
            name = escape(event.call.name)
            args = escape(str(event.call.arguments))
            console.print(f"\n[cyan]-> {name}({args})[/cyan]")
        elif event.type == "tool_finished":
            status = "[red]error[/red]" if event.result.is_error else "[green]ok[/green]"
            console.print(f"  {status}")
        elif event.type == "run_done":
            console.print()
            if event.state != "done":
                console.print(f"[red]run ended: {event.state}[/red]")


@app.command("models")
def models_cmd() -> None:
    """List models known to the registry and whether they're available."""
    router = _build_router()
    for m in router.registry.all():
        mark = "[green]x[/green]" if m.id in router.available else " "
        console.print(f"\\[{mark}] {m.id:20s} {m.display_name}")


if __name__ == "__main__":
    app()
