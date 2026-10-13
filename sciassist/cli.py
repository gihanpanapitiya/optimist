"""Terminal chat.

    sciassist chat
    sciassist chat --model anthropic:claude-opus-5-5
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import typer
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

app = typer.Typer(add_completion=False, help="AI science assistant.")
console = Console()

TOOL_PREVIEW_CHARS = 300


@app.callback()
def main() -> None:
    """AI science assistant."""


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Main agent model, provider:model."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full tool output."),
    thread: str = typer.Option(
        None, "--thread", "-t", help="Resume a saved conversation by its thread id."
    ),
) -> None:
    """Start an interactive chat session."""
    if model:
        os.environ["SCIASSIST_MODEL"] = model
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_chat(verbose, thread))


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    model: str = typer.Option(None, "--model", "-m", help="Main agent model, provider:model."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Start the web UI."""
    import uvicorn

    if model:
        os.environ["SCIASSIST_MODEL"] = model
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    console.print(f"[bold]Science assistant UI[/] → http://{host}:{port}")
    uvicorn.run("sciassist.server:app", host=host, port=port, reload=reload, log_level="warning")


async def _chat(verbose: bool, thread: str | None) -> None:
    from .agent import persistent_assistant

    async with persistent_assistant() as graph:
        await _chat_loop(graph, verbose, thread)


async def _chat_loop(graph, verbose: bool, thread: str | None) -> None:
    from litresearch.sources.base import close_http_client

    from .config import get_settings

    settings = get_settings()
    thread_id = thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    console.print(f"[bold]Science assistant[/] · model [cyan]{settings.model}[/] · "
                  f"workspace [cyan]{settings.workspace}[/]")
    if thread:
        state = await graph.aget_state(config)
        count = len((state.values or {}).get("messages", []))
        console.print(f"[dim]Resumed thread {thread_id} ({count} messages).[/]")
    console.print("[dim]Type your message. 'exit' to quit.[/]\n")

    try:
        while True:
            try:
                user = await asyncio.to_thread(console.input, "[bold green]you>[/] ")
            except (EOFError, KeyboardInterrupt):
                break
            if user.strip().lower() in {"exit", "quit", ":q"}:
                break
            if not user.strip():
                continue
            try:
                await _turn(graph, user, config, verbose)
            except KeyboardInterrupt:
                console.print("[yellow]interrupted[/]")
            except Exception as exc:  # noqa: BLE001 - keep the session alive
                console.print(f"[red]error:[/] {exc}")
    finally:
        await close_http_client()
        console.print(f"\n[dim]Resume with: sciassist chat --thread {thread_id}[/]")


async def _turn(graph, user: str, config: dict, verbose: bool) -> None:
    graph_input = {"messages": [{"role": "user", "content": user}]}
    # Each pass runs until the agent finishes or stops for approval.
    while graph_input is not None:
        interrupts = None
        async for mode, chunk in graph.astream(
            graph_input, config=config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                console.print(f"  [dim]· {escape(str(chunk.get('message', chunk)))}[/]")
            elif "__interrupt__" in chunk:
                interrupts = chunk["__interrupt__"]
            else:
                for node, update in chunk.items():
                    for message in (update or {}).get("messages", []):
                        # The approval node re-emits the model's message on resume.
                        if node in {"model", "tools"} or isinstance(message, ToolMessage):
                            _render(message, verbose)
        graph_input = await _ask_approval(interrupts) if interrupts else None


async def _ask_approval(interrupts) -> Command:
    """Ask about each pending tool call; the answers resume the run."""
    resume = {}
    for interrupt in interrupts:
        decisions = []
        for action in interrupt.value.get("action_requests", []):
            args = action["args"]
            shown = args.get("command") or args.get("file_path") or args
            console.print(f"[bold yellow]approve {action['name']}?[/] {escape(str(shown))}")
            answer = ""
            while not answer:  # no default: Enter alone must not run a command
                answer = (
                    await asyncio.to_thread(console.input, "[yellow]  [y]es / [n]o / reason> [/]")
                ).strip()
            if answer.lower() in {"y", "yes"}:
                decisions.append({"type": "approve"})
            elif answer.lower() in {"n", "no"}:
                decisions.append({"type": "reject"})
            else:
                decisions.append({"type": "reject", "message": answer})
        resume[interrupt.id] = {"decisions": decisions}
    return Command(resume=resume)


def _render(message, verbose: bool) -> None:
    if isinstance(message, AIMessage):
        if message.text.strip():
            console.print()
            console.print(Markdown(message.text))
            console.print()
        for call in message.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            if len(args) > 200 and not verbose:
                args = args[:200] + "…"
            console.print(f"[magenta]→ {call['name']}[/]({escape(args)})")
    elif isinstance(message, ToolMessage):
        text = message.text
        if not verbose and len(text) > TOOL_PREVIEW_CHARS:
            text = text[:TOOL_PREVIEW_CHARS].rstrip() + f" … [{len(message.text)} chars]"
        style = "red" if message.status == "error" else "dim"
        console.print(f"[{style}]← {message.name}: {escape(text)}[/]")


if __name__ == "__main__":
    app()
