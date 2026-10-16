"""Command line.

optimist                          interactive session
optimist --resume [id]            continue a conversation (picker without id)
optimist -p "question"            one turn, non-interactive; answer on stdout
cat data.csv | optimist -p "..."  stdin is appended to the prompt
optimist ui                       web UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import typer
from rich.console import Console

from .approvals import MODES, Approvals

app = typer.Typer(add_completion=False, help="AI science assistant.")

_MODE_HELP = "Approval mode: ask, edits (default) or auto."


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: str = typer.Option(
        None, "--print", "-p", help="Run one turn non-interactively and print the answer."
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model, provider:name."),
    mode: str = typer.Option(None, "--mode", help=_MODE_HELP),
    resume: str = typer.Option(
        None, "--resume", "-r", help="Continue a conversation by id ('last' for the latest)."
    ),
    output_json: bool = typer.Option(False, "--json", help="With -p: print JSON, not text."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full tool output."),
) -> None:
    """AI science assistant. Without a command, starts an interactive session."""
    if ctx.invoked_subcommand is not None:
        return
    if model:
        os.environ["OPTIMIST_MODEL"] = model
    if mode is not None and mode not in MODES:
        raise typer.BadParameter(f"choose one of {', '.join(MODES)}", param_hint="--mode")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if prompt is not None:
        if not sys.stdin.isatty():
            piped = sys.stdin.read()
            if piped.strip():
                prompt = f"{prompt}\n\n<stdin>\n{piped}\n</stdin>"
        code = asyncio.run(_headless(prompt, mode, resume, output_json, verbose))
        raise typer.Exit(code)
    asyncio.run(_interactive(mode, resume, verbose))


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Model, provider:name."),
    thread: str = typer.Option(None, "--thread", "-t", help="Resume a conversation by id."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive session (same as running optimist with no command)."""
    if model:
        os.environ["OPTIMIST_MODEL"] = model
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_interactive(None, thread, verbose))


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    model: str = typer.Option(None, "--model", "-m", help="Main agent model, provider:name."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Start the web UI."""
    import uvicorn

    if model:
        os.environ["OPTIMIST_MODEL"] = model
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    Console().print(f"[bold]Science assistant UI[/] → http://{host}:{port}")
    uvicorn.run("optimist.server:app", host=host, port=port, reload=reload, log_level="warning")


@asynccontextmanager
async def _session(console: Console, mode: str | None, resume: str | None, verbose: bool):
    from optimistresearch.sources.base import close_http_client

    from ..agent import open_checkpointer
    from ..config import get_settings
    from ..threads import ThreadStore
    from .render import Renderer
    from .session import Session

    settings = get_settings()
    async with open_checkpointer() as saver:
        threads = await ThreadStore.open(settings.db_path)
        try:
            session = Session(
                saver=saver,
                threads=threads,
                renderer=Renderer(console, verbose=verbose),
                approvals=Approvals(mode=mode or ("auto" if settings.auto_approve else "edits")),
            )
            if resume:
                session.new_thread(await _resolve_thread(threads, resume))
            yield session
        finally:
            await threads.close()
            await close_http_client()


async def _resolve_thread(threads, ref: str) -> str:
    known = await threads.list()
    if ref == "last":
        if not known:
            raise typer.BadParameter("no saved conversations", param_hint="--resume")
        return known[0]["id"]
    return next((t["id"] for t in known if t["id"].startswith(ref)), ref)


async def _interactive(mode: str | None, resume: str | None, verbose: bool) -> None:
    from .repl import Repl

    console = Console()
    async with _session(console, mode, resume, verbose) as session:
        await Repl(session, console).loop()


async def _headless(
    prompt: str, mode: str | None, resume: str | None, output_json: bool, verbose: bool
) -> int:
    # Progress goes to stderr so stdout carries only the answer.
    console = Console(stderr=True)
    async with _session(console, mode, resume, verbose) as session:
        try:
            result = await session.run(prompt, ask=None, watch_keys=False)
        except Exception as exc:  # noqa: BLE001 - report and exit non-zero
            console.print(f"[red]Error:[/] {type(exc).__name__}: {exc}")
            return 1
        if output_json:
            usage = result.usage
            payload = {
                "thread_id": session.thread_id,
                "text": result.text,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cost_usd": round(usage.cost, 6) if usage.priced else None,
                },
                "seconds": round(result.seconds, 1),
            }
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(result.text)
        return 0
