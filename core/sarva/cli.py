"""sarva.cli — the `sarva` command-line entry point.

Zero-config by default: with no ANTHROPIC_API_KEY set, everything routes to
the offline MockProvider so `sarva chat "hello"` always works.
"""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import BUILTIN_TOOLS, always_allow
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ContentBlock, ImageBlock, Message, TextBlock
from sarva.providers.base import TextDeltaEvent
from sarva.runtime import build_providers, build_router

app = typer.Typer(help="Sarva — an open, all-in-one multimodal AGI tool.")
sessions_app = typer.Typer(help="Manage persisted chat sessions (used by `sarva chat --session`).")
app.add_typer(sessions_app, name="sessions")
console = Console()

# Kept as thin aliases so the rest of this file reads the same as before the
# provider-wiring logic moved to sarva.runtime (shared with the server skin).
_build_router = build_router
_build_providers = build_providers


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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8000, help="Port to bind."),
) -> None:
    """Run the REST + WebSocket server — the surface a web UI or desktop app uses."""
    import uvicorn

    from sarva.server.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
