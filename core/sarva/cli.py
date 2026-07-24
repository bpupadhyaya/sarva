"""sarva.cli — the `sarva` command-line entry point.

Zero-config by default: with no ANTHROPIC_API_KEY set, everything routes to
the offline MockProvider so `sarva chat "hello"` always works.
"""

from __future__ import annotations

import asyncio
import mimetypes
import platform
import shlex
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from sarva.agent.loop import AgentLoop
from sarva.agent.tools import BUILTIN_TOOLS, Tool, always_allow
from sarva.mcp_client import connect_http_mcp_server, connect_stdio_mcp_server, list_mcp_tools
from sarva.memory.session import SessionStore
from sarva.multimodal.content import ContentBlock, ImageBlock, Message
from sarva.multimodal.degraders import default_degraders
from sarva.providers.base import TextDeltaEvent
from sarva.runtime import build_providers, build_router, run_diagnostics

app = typer.Typer(help="Sarva — an open, all-in-one multimodal AGI tool.")
sessions_app = typer.Typer(help="Manage persisted chat sessions (used by `sarva chat --session`).")
app.add_typer(sessions_app, name="sessions")
config_app = typer.Typer(help="Manage saved provider API keys (~/.sarva/config.json).")
app.add_typer(config_app, name="config")
console = Console()

# Kept as thin aliases so the rest of this file reads the same as before the
# provider-wiring logic moved to sarva.runtime (shared with the server skin).
_build_router = build_router
_build_providers = build_providers


def _version_callback(show_version: bool) -> None:
    if not show_version:
        return
    from importlib.metadata import version

    console.print(f"sarva {version('sarva')}")
    raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed sarva version and exit.",
    ),
) -> None:
    pass


def _load_image(path: str) -> ImageBlock:
    media_type, _ = mimetypes.guess_type(path)
    if media_type is None or not media_type.startswith("image/"):
        raise typer.BadParameter(f"cannot determine an image media type for {path!r}")
    return ImageBlock(media_type=media_type, data=Path(path).read_bytes())


def _parse_mcp_headers(values: list[str]) -> dict[str, str]:
    # Reject rather than silently skip a malformed entry -- the same
    # "don't guess" discipline session-name validation already applies:
    # a header a user thought they sent but got silently dropped (e.g. a
    # missing ':') is a much worse failure mode than an immediate error.
    headers: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise typer.BadParameter(f"invalid --mcp-header {value!r} -- expected 'Name: Value'")
        name, _, header_value = value.partition(":")
        headers[name.strip()] = header_value.strip()
    return headers


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send."),
    image: Path | None = typer.Option(
        None, "--image", help="Attach an image file (requires a vision-capable model)."
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Force a specific model id (see 'sarva models' for the full list), "
        "bypassing the router's own default candidate selection entirely. "
        "Omit to let Sarva pick automatically.",
    ),
    session: str | None = typer.Option(
        None,
        "--session",
        help="Remember this conversation under a name (loads prior history, "
        "saves after the turn). Omit for a one-shot, unremembered chat.",
    ),
) -> None:
    """One-shot chat — no tools, single turn."""
    asyncio.run(_chat(message, image, model, session))


async def _chat(message: str, image: Path | None, model: str | None, session: str | None) -> None:
    store = SessionStore()
    history = store.load(session) if session else []
    extra_content: list[ContentBlock] = [_load_image(str(image))] if image else []

    loop = AgentLoop(
        router=_build_router(),
        providers=_build_providers(),
        tools=[],
        confirm=always_allow,
        degraders=default_degraders(),
    )
    final_state = None
    last_detail: str | None = None
    transcript: list[Message] = []
    async for event in loop.run(
        message,
        history=history,
        model_override=model,
        extra_content=extra_content,
        transcript_out=transcript,
        session_id=session,
    ):
        # Model output may itself contain "[", e.g. markdown links or
        # citations — never markup-parse text that came from the model.
        if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
            console.print(event.event.text, end="", markup=False)
        elif event.type == "state_changed" and event.detail:
            last_detail = event.detail
        if event.type == "run_done":
            console.print()
            final_state = event.state
            if event.state != "done":
                _print_run_failure(event.state, last_detail)

    if session and final_state == "done":
        store.save(session, transcript)
    if final_state is not None and final_state != "done":
        # A failed/interrupted/budget-exceeded run exiting 0 (true before
        # this fix) meant a script chaining `sarva chat ... || handle_it`
        # could never detect the failure -- the same instinct behind
        # surfacing `detail` above: an error nobody can act on isn't
        # meaningfully different from no error at all.
        raise typer.Exit(code=1)


@app.command()
def run(
    task: str = typer.Argument(..., help="Task for the agent to complete."),
    workdir: str = typer.Option(".", help="Working directory for file/shell tools."),
    image: Path | None = typer.Option(
        None, "--image", help="Attach an image file (requires a vision-capable model)."
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Force a specific model id (see 'sarva models' for the full list), "
        "bypassing the router's own default candidate selection entirely. "
        "Omit to let Sarva pick automatically.",
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Auto-approve destructive tools (no confirmation prompts)."
    ),
    session: str | None = typer.Option(
        None,
        "--session",
        help="Remember this conversation under a name, including tool-use rounds "
        "(loads prior history, saves after the turn). Omit for a one-shot run.",
    ),
    mcp_server: list[str] = typer.Option(
        [],
        "--mcp-server",
        help="Connect an MCP server and add its tools to this run (repeatable). "
        'A shell command connects over stdio, e.g. --mcp-server "npx -y '
        '@modelcontextprotocol/server-filesystem /tmp"; an http:// or https:// '
        "URL connects over Streamable HTTP instead, e.g. "
        "--mcp-server https://example.com/mcp.",
    ),
    mcp_header: list[str] = typer.Option(
        [],
        "--mcp-header",
        help="HTTP header to send to every http(s):// --mcp-server in this run "
        '(repeatable), "Name: Value" -- e.g. --mcp-header "Authorization: Bearer '
        'sk-...". Most real MCP deployments need one of these for auth; '
        "connect_http_mcp_server() has always accepted headers, this is what "
        "actually threads one through from the command line. Applies to every "
        "HTTP server in this invocation alike, not per-server -- a real, named "
        "limit for the (rare) case of multiple HTTP servers needing different "
        "auth in one run.",
    ),
) -> None:
    """Run the agent loop with built-in tools (files, shell) plus any MCP servers."""
    asyncio.run(_run(task, workdir, image, model, auto, session, mcp_server, mcp_header))


async def _confirm_prompt(call: Any) -> bool:
    return typer.confirm(f"Allow {call.name}({call.arguments})?")


def _print_run_failure(state: str, detail: str | None) -> None:
    # Every non-DONE terminal state carries a StateChangedEvent with a
    # `detail` message (an unknown --model, a provider crash, ...) that
    # went completely unsurfaced until now -- only ever visible by
    # digging through .sarva/runs/<id>/transcript.jsonl by hand. `detail`
    # can originate from a provider's own raw error text (StreamErrorEvent),
    # not just this project's own strings -- escape() before rendering,
    # the same discipline `doctor`'s dynamic detail text already uses,
    # rather than assuming it can never contain a stray '[' Rich would
    # try to parse as a markup tag.
    if detail:
        console.print(f"[red]run ended: {state} — {escape(detail)}[/red]")
    else:
        console.print(f"[red]run ended: {state}[/red]")


async def _run(
    task: str,
    workdir: str,
    image: Path | None,
    model: str | None,
    auto: bool,
    session: str | None,
    mcp_servers: list[str],
    mcp_headers: list[str],
) -> None:
    store = SessionStore()
    history = store.load(session) if session else []
    extra_content: list[ContentBlock] = [_load_image(str(image))] if image else []
    confirm = always_allow if auto else _confirm_prompt
    headers = _parse_mcp_headers(mcp_headers)

    async with AsyncExitStack() as stack:
        tools: list[Tool] = list(BUILTIN_TOOLS)
        for server_cmd in mcp_servers:
            if server_cmd.startswith(("http://", "https://")):
                mcp_session = await stack.enter_async_context(
                    connect_http_mcp_server(server_cmd, headers=headers or None)
                )
            else:
                command, *args = shlex.split(server_cmd)
                mcp_session = await stack.enter_async_context(
                    connect_stdio_mcp_server(command, args=args)
                )
            mcp_tools = await list_mcp_tools(mcp_session)
            # escape(): tool names come from the connected MCP server's own
            # response -- for an http(s):// server that's a remote,
            # untrusted source (a malicious/buggy server could name a tool
            # with embedded Rich markup and spoof this project's own
            # terminal output). Same discipline every other externally-
            # sourced string in this file already gets.
            tool_names = ", ".join(escape(t.spec.name) for t in mcp_tools)
            console.print(f"[dim]mcp: {escape(repr(server_cmd))} -> {tool_names}[/dim]")
            tools.extend(mcp_tools)

        loop = AgentLoop(
            router=_build_router(),
            providers=_build_providers(),
            tools=tools,
            confirm=confirm,
            workdir=workdir,
            degraders=default_degraders(),
        )
        final_state = None
        last_detail: str | None = None
        transcript: list[Message] = []
        async for event in loop.run(
            task,
            history=history,
            model_override=model,
            extra_content=extra_content,
            transcript_out=transcript,
            session_id=session,
        ):
            if event.type == "model_stream" and isinstance(event.event, TextDeltaEvent):
                console.print(event.event.text, end="", markup=False)
            elif event.type == "tool_started":
                name = escape(event.call.name)
                args = escape(str(event.call.arguments))
                console.print(f"\n[cyan]-> {name}({args})[/cyan]")
            elif event.type == "tool_finished":
                status = "[red]error[/red]" if event.result.is_error else "[green]ok[/green]"
                console.print(f"  {status}")
            elif event.type == "state_changed" and event.detail:
                last_detail = event.detail
            elif event.type == "run_done":
                console.print()
                final_state = event.state
                if event.state != "done":
                    _print_run_failure(event.state, last_detail)

        if session and final_state == "done":
            store.save(session, transcript)
        if final_state is not None and final_state != "done":
            raise typer.Exit(code=1)


@app.command("models")
def models_cmd() -> None:
    """List models known to the registry and whether they're available."""
    router = _build_router()
    for m in router.registry.all():
        mark = "[green]x[/green]" if m.id in router.available else " "
        console.print(f"\\[{mark}] {m.id:20s} {m.display_name}")


@app.command("doctor")
def doctor_cmd() -> None:
    """Diagnose this local Sarva setup: which providers are configured,
    whether Ollama and any foundry checkpoints are reachable, and whether
    the web UI is built in for `sarva serve` -- the same checks
    `build_router`/`build_providers` use to decide availability, so this
    never drifts out of sync with what actually works."""
    console.print(f"Python {sys.version.split()[0]} on {platform.system()} ({platform.machine()})")
    console.print()

    # `check.detail`/`static_dir` can contain literal square brackets (e.g.
    # "sarva[foundry]") that Rich's markup parser would otherwise silently
    # swallow as an (invalid) style tag -- escape() is what keeps those
    # bytes on screen instead of vanishing, the same reason the rest of
    # this file never prints raw model output without it.
    for check in run_diagnostics():
        mark = "[green]x[/green]" if check.ok else "[dim]-[/dim]"
        console.print(f"\\[{mark}] {check.name:32s} {escape(check.detail)}")

    console.print()
    static_dir = Path(__file__).parent / "server" / "static"
    web_ui_built = static_dir.is_dir() and any(static_dir.iterdir())
    mark = "[green]x[/green]" if web_ui_built else "[dim]-[/dim]"
    detail = (
        f"built web UI present at {static_dir}"
        if web_ui_built
        else f"no built web UI at {static_dir} -- `sarva serve` will be API-only "
        "(run ./scripts/build-web.sh to add it)"
    )
    console.print(f"\\[{mark}] {'Web UI (for sarva serve)':32s} {escape(detail)}")

    console.print(
        "\n[dim]Every unchecked item above is an optional provider or feature -- "
        "a fresh install is expected to fail most of these and still work fine "
        "via the zero-config Mock provider.[/dim]"
    )


@app.command("eval")
def eval_cmd(
    model: str | None = typer.Option(
        None, "--model", help="Only evaluate this model id (default: every available model)."
    ),
) -> None:
    """Grade available models against the bundled benchmark — the same
    yardstick for every model, whichever provider it comes from."""
    asyncio.run(_eval(model))


async def _eval(model_filter: str | None) -> None:
    from sarva.eval import ARITHMETIC, run_benchmark

    router = _build_router()
    providers = _build_providers()
    model_ids = [model_filter] if model_filter else sorted(router.available)

    console.print(f"Benchmark: {ARITHMETIC.name} ({len(ARITHMETIC.cases)} cases)\n")
    for model_id in model_ids:
        info = router.registry.get(model_id)
        provider = providers.get(info.provider)
        if provider is None:
            console.print(
                f"[yellow]skip[/yellow]  {model_id} (provider {info.provider!r} not configured)"
            )
            continue
        report = await run_benchmark(ARITHMETIC, provider, model=model_id)
        correct = sum(r.correct for r in report.results)
        console.print(f"{model_id:25s} {report.accuracy:.0%}  ({correct}/{len(report.results)})")


@app.command("distill")
def distill_cmd(
    prompts_file: Path = typer.Argument(..., help="Text file, one prompt per line."),
    model: str = typer.Option(..., "--model", help="Model id to distill from."),
    out: Path = typer.Option(..., "--out", help="Output JSONL path (prompt/completion/model)."),
    system: str | None = typer.Option(
        None, "--system", help="Optional system prompt applied to every request."
    ),
) -> None:
    """Generate (prompt, completion) pairs from a real model — frontier-
    as-teacher synthetic data (spec §3.6c) for foundry SFT training."""
    asyncio.run(_distill(prompts_file, model, out, system))


async def _distill(prompts_file: Path, model: str, out: Path, system: str | None) -> None:
    from sarva.distill import distill, save_jsonl

    router = _build_router()
    providers = _build_providers()
    info = router.registry.get(model)
    provider = providers.get(info.provider)
    if provider is None:
        console.print(
            f"[red]provider {info.provider!r} for model {model!r} is not configured[/red]"
        )
        raise typer.Exit(1)

    prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]
    console.print(f"Distilling {len(prompts)} prompts from {model}...")
    records = await distill(prompts, provider, model=model, system=system)
    save_jsonl(records, out)
    console.print(f"Wrote {len(records)} records to {out}")


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


@config_app.command("set")
def config_set(
    anthropic_api_key: str | None = typer.Option(None, "--anthropic-api-key"),
    openai_api_key: str | None = typer.Option(None, "--openai-api-key"),
    gemini_api_key: str | None = typer.Option(None, "--gemini-api-key"),
) -> None:
    """Save one or more provider API keys to ~/.sarva/config.json (owner-only
    permissions -- see `sarva.config`'s own docstring). This is the CLI's
    own reachable surface for exactly what the desktop app's first-run
    screen already does via `POST /config` -- `sarva.config.save_config`
    has been callable since that screen shipped, but nothing exposed it
    to a CLI-only user with no desktop app installed at all. A real
    environment variable of the same name always wins over whatever's
    saved here (`sarva.config.get_env`'s own documented precedence,
    unchanged by this command)."""
    from sarva.config import save_config

    values = {
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "GEMINI_API_KEY": gemini_api_key,
    }
    non_empty = {k: v for k, v in values.items() if v}
    if not non_empty:
        console.print(
            "[yellow]nothing to save -- pass at least one of --anthropic-api-key / "
            "--openai-api-key / --gemini-api-key[/yellow]"
        )
        raise typer.Exit(1)
    save_config(non_empty)
    console.print(f"saved {', '.join(sorted(non_empty))} to ~/.sarva/config.json")


@config_app.command("show")
def config_show() -> None:
    """Show which provider keys are configured and where each comes from
    (a real environment variable always wins over a saved config-file
    value) -- never prints the actual key, only whether one is set."""
    import os

    from sarva.config import KNOWN_KEYS, load_config

    saved = load_config()
    for name in KNOWN_KEYS:
        if os.environ.get(name):
            console.print(f"{name:20s} [green]set[/green] (environment variable)")
        elif saved.get(name):
            console.print(f"{name:20s} [green]set[/green] (saved config file)")
        else:
            console.print(f"{name:20s} [dim]not set[/dim]")


@config_app.command("unset")
def config_unset(
    anthropic_api_key: bool = typer.Option(False, "--anthropic-api-key"),
    openai_api_key: bool = typer.Option(False, "--openai-api-key"),
    gemini_api_key: bool = typer.Option(False, "--gemini-api-key"),
) -> None:
    """Remove one or more provider API keys from ~/.sarva/config.json --
    `set`'s missing counterpart. A real environment variable of the same
    name is never touched (this command only ever edits the saved
    file); a key that was never saved is silently a no-op, not an
    error."""
    from sarva.config import unset_config

    requested = {
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "GEMINI_API_KEY": gemini_api_key,
    }
    names = [name for name, wanted in requested.items() if wanted]
    if not names:
        console.print(
            "[yellow]nothing to remove -- pass at least one of --anthropic-api-key / "
            "--openai-api-key / --gemini-api-key[/yellow]"
        )
        raise typer.Exit(1)
    removed = unset_config(names)
    if removed:
        console.print(f"removed {', '.join(sorted(removed))} from ~/.sarva/config.json")
    else:
        console.print("nothing to do -- none of those were saved")


@app.command()
def speak(
    text: str = typer.Argument(..., help="Text to synthesize as speech."),
    out: Path = typer.Option(Path("speech.wav"), "--out", help="Output WAV file path."),
    voice: str | None = typer.Option(
        None, "--voice", help="Voice name (engine-specific; default: a bundled voice)."
    ),
) -> None:
    """Local text-to-speech (macOS `say` / Linux `espeak`) -- no API key,
    no network."""
    from sarva.audio import synthesize

    try:
        audio_bytes = synthesize(text, voice=voice)
    except RuntimeError as e:
        # escape(): the same real bug just found and fixed for
        # transcribe's own error path -- this message has no brackets
        # today, but nothing stops a future edit from adding one, the
        # same reason doctor's dynamic detail text is escaped too.
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1) from e
    out.write_bytes(audio_bytes)
    console.print(f"wrote {len(audio_bytes)} bytes to {out}")


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., help="Audio file to transcribe."),
    model_size: str = typer.Option(
        "tiny",
        "--model-size",
        help="faster-whisper model size (tiny/base/small/medium/large-v3, ...) "
        "-- bigger means more accurate, slower, and a larger one-time download.",
    ),
) -> None:
    """Local speech-to-text via faster-whisper (the `sarva\\[audio]` extra)
    -- no API key, no network. `speak`'s reverse: this project had a real
    TTS command but no STT one, despite `sarva.audio.transcribe()` (used
    internally by `AudioToTextDegrader`) being fully built and tested the
    whole time -- the same "built, unreachable by any real user" shape
    this project keeps finding and closing."""
    from sarva.audio import transcribe as transcribe_audio

    try:
        text = transcribe_audio(audio.read_bytes(), model_size=model_size)
    except ImportError as e:
        # A real bug caught by this file's own test, not assumed safe:
        # the real error message contains a literal "sarva[audio]" --
        # printed unescaped, Rich's markup parser silently swallowed the
        # "[audio]" part, the identical class of bug `doctor`'s dynamic
        # detail text was fixed for earlier.
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1) from e
    # The transcript is externally-derived text (real speech, not this
    # project's own strings) -- never markup-parsed, same discipline
    # chat/run already apply to model output.
    console.print(text, markup=False)


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
