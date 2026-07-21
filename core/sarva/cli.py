"""sarva.cli — the `sarva` command-line entry point.

Zero-config by default: with no ANTHROPIC_API_KEY set, everything routes to
the offline MockProvider so `sarva chat "hello"` always works.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.markup import escape

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import BUILTIN_TOOLS, always_allow
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ContentBlock, ImageBlock, Message, TextBlock
from sarva.providers.anthropic_provider import AnthropicProvider
from sarva.providers.base import TextDeltaEvent
from sarva.providers.mock import MockProvider
from sarva.providers.ollama_provider import OllamaProvider
from sarva.providers.registry import Registry, Router, load_routing

app = typer.Typer(help="Sarva — an open, all-in-one multimodal AGI tool.")
sessions_app = typer.Typer(help="Manage persisted chat sessions (used by `sarva chat --session`).")
app.add_typer(sessions_app, name="sessions")
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


def _load_image(path: str) -> ImageBlock:
    media_type, _ = mimetypes.guess_type(path)
    if media_type is None or not media_type.startswith("image/"):
        raise typer.BadParameter(f"cannot determine an image media type for {path!r}")
    return ImageBlock(media_type=media_type, data=Path(path).read_bytes())


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send."),
    image: Path | None = typer.Option(
        None, "--image", help="Attach an image file (requires a vision-capable model)."
    ),
    session: str | None = typer.Option(
        None,
        "--session",
        help="Remember this conversation under a name (loads prior history, "
        "saves after the turn). Omit for a one-shot, unremembered chat.",
    ),
) -> None:
    """One-shot chat — no tools, single turn."""
    asyncio.run(_chat(message, image, session))


async def _chat(message: str, image: Path | None, session: str | None) -> None:
    store = SessionStore()
    history = store.load(session) if session else []
    extra_content: list[ContentBlock] = [_load_image(str(image))] if image else []

    loop = AgentLoop(
        router=_build_router(), providers=_build_providers(), tools=[], confirm=always_allow
    )
    final_message: Message | None = None
    async for event in loop.run(message, history=history, extra_content=extra_content):
        # Model output may itself contain "[", e.g. markdown links or
        # citations — never markup-parse text that came from the model.
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            console.print(event.event.text, end="", markup=False)
        if event.type == "run_done":
            console.print()
            final_message = event.final_message
            if event.state != "done":
                console.print(f"[red]run ended: {event.state}[/red]")

    # Tool-free by construction (chat passes tools=[]), so the full turn is
    # exactly [user message, final assistant message] — safe to append as-is.
    # `sarva run` isn't wired for --session yet: reconstructing history across
    # tool-use rounds needs more than this. See BUILD-JOURNAL.md.
    if session and final_message is not None:
        user_message = Message(role="user", content=[TextBlock(text=message), *extra_content])
        store.save(session, [*history, user_message, final_message])


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


@sessions_app.command("list")
def sessions_list() -> None:
    """List saved chat sessions and how many messages each holds."""
    store = SessionStore()
    names = store.list_sessions()
    if not names:
        console.print("no saved sessions")
        return
    for name in names:
        count = len(store.load(name))
        console.print(f"{name}  ({count} messages)")


@sessions_app.command("clear")
def sessions_clear(name: str = typer.Argument(..., help="Session name to delete.")) -> None:
    """Delete a saved session."""
    SessionStore().clear(name)
    console.print(f"cleared session {name!r}")


if __name__ == "__main__":
    app()
