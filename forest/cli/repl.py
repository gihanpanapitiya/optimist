"""The interactive terminal session.

Enter            send          Esc Enter / Ctrl-J / \\ Enter   new line
Shift+Tab        cycle approval mode
Esc / Ctrl-C     interrupt the agent (Ctrl-C at an empty prompt twice: quit)
@path            attach a workspace file      /command   see /help
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import get_settings
from ..tools.files import resolve_in_workspace, to_agent_path
from .approvals import MODE_HELP, MODES
from .render import call_summary
from .session import Session
from .usage import format_cost, format_tokens

MAX_ATTACH_BYTES = 100_000
HIDDEN_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".ipynb_checkpoints"}

COMPACT_PROMPT = (
    "Call the compact_conversation tool now to compact our conversation history, "
    "then confirm in one short line."
)
INIT_PROMPT = (
    "Look through the workspace and write /AGENTS.md: a short guide for future sessions "
    "covering what this workspace contains, the research questions in progress, how code "
    "is organized and run, and conventions to follow. Keep it under 60 lines. If "
    "/AGENTS.md exists, update it instead of replacing what is still true."
)

PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "bottom-toolbar": "noreverse #808080",
        "bottom-toolbar.mode": "noreverse ansimagenta",
        "bottom-toolbar.key": "noreverse #a0a0a0 bold",
    }
)


@dataclass
class Command:
    help: str
    run: Callable[[Repl, str], Awaitable[None]]
    usage: str = ""


class Repl:
    def __init__(self, session: Session, console: Console):
        self.session = session
        self.console = console
        self.quit = False
        self._last_ctrl_c = 0.0
        history = get_settings().db_path.parent / "cli_history"
        self.prompt = PromptSession(
            history=FileHistory(str(history)),
            completer=merge_completers([SlashCompleter(), FileCompleter()]),
            complete_while_typing=True,
            multiline=True,
            key_bindings=self._key_bindings(),
            bottom_toolbar=self._toolbar,
            style=PROMPT_STYLE,
            prompt_continuation=lambda width, *_: "  ".rjust(width),
        )
        self.approval_prompt = PromptSession(style=PROMPT_STYLE)

    # -- main loop ---------------------------------------------------------

    async def loop(self) -> None:
        self._banner()
        if await self.session.message_count():
            await self._show_resumed()
        while not self.quit:
            try:
                text = await self.prompt.prompt_async([("class:prompt", "> ")])
            except KeyboardInterrupt:
                if time.monotonic() - self._last_ctrl_c < 2:
                    break
                self._last_ctrl_c = time.monotonic()
                self.console.print(Text("  Press Ctrl-C again or Ctrl-D to quit.", style="dim"))
                continue
            except EOFError:
                break
            text = text.strip()
            if not text:
                continue
            if text.startswith("/") and not text.startswith("//"):
                await self._command(text)
            else:
                await self.send(text)
        self.console.print(
            Text(f"Resume this chat with: forest --resume {self.session.thread_id}", style="dim")
        )

    async def send(self, text: str, *, title: str | None = None) -> None:
        message = self._attach_files(text)
        try:
            result = await self.session.run(
                message, ask=self._ask, watch_keys=True, title=title or text
            )
        except Exception as exc:  # noqa: BLE001 - keep the session alive
            self.console.print(Text(f"Error: {type(exc).__name__}: {exc}", style="red"))
            return
        self.session.renderer.turn_footer(result.usage, result.seconds)

    # -- approvals ---------------------------------------------------------

    async def _ask(self, action: dict) -> dict | None:
        name = action["name"]
        self.console.print(
            Text.assemble(
                ("? ", "bold yellow"),
                ("Allow ", "bold"),
                (name, "bold yellow"),
                call_summary(name, action.get("args") or {}),
            )
        )
        self.console.print(
            Text.assemble(
                ("  y", "bold green"), " yes   ",
                ("a", "bold green"), f" yes, and don't ask again for {name}   ",
                ("n", "bold red"), " no   ",
                ("or type what to do instead", "dim"),
            )
        )  # fmt: skip
        while True:
            try:
                answer = (
                    await self.approval_prompt.prompt_async([("class:prompt", "  › ")])
                ).strip()
            except (KeyboardInterrupt, EOFError):
                return {"type": "reject", "message": "The user declined."}
            if answer.lower() in {"y", "yes"}:
                return {"type": "approve"}
            if answer.lower() in {"a", "always"}:
                return None
            if answer.lower() in {"n", "no"}:
                return {"type": "reject", "message": "The user declined."}
            if answer:
                return {"type": "reject", "message": f"The user declined and said: {answer}"}

    # -- input -------------------------------------------------------------

    def _key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add(Keys.Enter)
        def _(event):
            buffer = event.current_buffer
            if buffer.complete_state and buffer.complete_state.current_completion:
                buffer.apply_completion(buffer.complete_state.current_completion)
                return
            if buffer.document.text_before_cursor.endswith("\\"):
                buffer.delete_before_cursor()
                buffer.insert_text("\n")
                return
            buffer.validate_and_handle()

        @kb.add(Keys.Escape, Keys.Enter)
        @kb.add(Keys.ControlJ)
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.BackTab)
        def _(event):
            self.session.approvals.cycle()
            event.app.invalidate()

        return kb

    def _toolbar(self) -> FormattedText:
        s = self.session
        usage = s.usage.session
        parts = [
            ("", f" {s.model}  ·  "),
            ("class:bottom-toolbar.mode", f"{s.approvals.mode}"),
            ("", " mode "),
            ("class:bottom-toolbar.key", "⇧⇥"),
            ("", f"  ·  context {format_tokens(s.usage.context_tokens)}"),
            ("", f"  ·  {format_cost(usage)}"),
            ("", "  ·  /help"),
        ]
        return FormattedText(parts)

    def _attach_files(self, text: str) -> str:
        """Inline the files named with @path, as Claude Code does."""
        blocks = []
        for match in re.finditer(r"(?<!\S)@(\S+)", text):
            raw = match.group(1).rstrip(".,;:)")
            try:
                path = resolve_in_workspace(raw)
            except ValueError:
                continue
            if not path.is_file():
                continue
            agent_path = to_agent_path(path)
            size = path.stat().st_size
            if size > MAX_ATTACH_BYTES:
                self.console.print(
                    Text(f"  ⎿  {agent_path} is {size // 1000} kB; the agent will read it", "dim")
                )
                continue
            content = path.read_text(errors="replace")
            blocks.append(f'<file path="{agent_path}">\n{content}\n</file>')
            lines = content.count("\n") + 1
            self.console.print(Text(f"  ⎿  attached {agent_path} ({lines} lines)", style="dim"))
        return text + ("\n\n" + "\n\n".join(blocks) if blocks else "")

    # -- display -----------------------------------------------------------

    def _banner(self) -> None:
        settings = get_settings()
        body = Text.assemble(
            ("forest", "bold cyan"),
            ("  science assistant\n\n", "dim"),
            ("model      ", "dim"), (self.session.model, ""), "\n",
            ("workspace  ", "dim"), (str(settings.workspace), ""), "\n",
            ("approvals  ", "dim"), (self.session.approvals.mode, "magenta"),
            (f" ({MODE_HELP[self.session.approvals.mode]})", "dim"), "\n\n",
            ("/help for commands · @file to attach · shift+tab to change mode · esc to interrupt",
             "dim"),
        )  # fmt: skip
        self.console.print(Panel(body, border_style="cyan", expand=False))

    async def _show_resumed(self) -> None:
        state = await self.session.graph.aget_state(self.session.config)
        messages = (state.values or {}).get("messages", [])
        last = next(
            (m for m in reversed(messages) if m.type == "ai" and m.text.strip()),
            None,
        )
        self.console.print(Text(f"Resumed conversation ({len(messages)} messages).", "dim"))
        if last is not None:
            self.console.print(Markdown(last.text[-3000:]))
        if state.interrupts:
            self.console.print(
                Text("The last run was waiting for approval; send a message to continue.", "yellow")
            )

    # -- slash commands ----------------------------------------------------

    async def _command(self, text: str) -> None:
        name, _, arg = text[1:].partition(" ")
        command = COMMANDS.get(name.lower())
        if command is None:
            self.console.print(Text(f"Unknown command /{name}. Try /help.", style="red"))
            return
        await command.run(self, arg.strip())


async def _help(repl: Repl, _arg: str) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for name, cmd in COMMANDS.items():
        if name in ALIASES:
            continue
        table.add_row(Text(f"/{name} {cmd.usage}".rstrip(), style="cyan"), cmd.help)
    repl.console.print(table)
    repl.console.print(
        Text(
            "\nEnter sends · Esc Enter, Ctrl-J or \\ Enter for a new line · @path attaches a file\n"
            "Shift+Tab cycles approval mode · Esc or Ctrl-C interrupts the agent",
            style="dim",
        )
    )


async def _new(repl: Repl, _arg: str) -> None:
    repl.session.new_thread()
    repl.console.clear()
    repl._banner()
    repl.console.print(Text("Started a new conversation.", style="dim"))


async def _resume(repl: Repl, arg: str) -> None:
    threads = await repl.session.threads.list()
    if not threads:
        repl.console.print(Text("No saved conversations.", style="dim"))
        return
    if not arg:
        table = Table(show_header=False, box=None, padding=(0, 2))
        for i, t in enumerate(threads[:20], 1):
            when = datetime.fromtimestamp(t["updated_at"]).strftime("%b %d %H:%M")
            current = " (current)" if t["id"] == repl.session.thread_id else ""
            table.add_row(Text(str(i), style="cyan"), Text(when, style="dim"), t["title"] + current)
        repl.console.print(table)
        try:
            arg = await repl.approval_prompt.prompt_async("  number to resume (Enter to cancel) › ")
        except (KeyboardInterrupt, EOFError):
            return
        arg = arg.strip()
        if not arg:
            return
    chosen = None
    if arg.isdigit() and 1 <= int(arg) <= len(threads):
        chosen = threads[int(arg) - 1]["id"]
    else:
        chosen = next((t["id"] for t in threads if t["id"].startswith(arg)), None)
    if chosen is None:
        repl.console.print(Text(f"No conversation matches {arg!r}.", style="red"))
        return
    repl.session.new_thread(chosen)
    await repl._show_resumed()


async def _model(repl: Repl, arg: str) -> None:
    if not arg:
        repl.console.print(f"Model: [cyan]{repl.session.model}[/]  (subagents: "
                           f"{repl.session.subagent_model})\nChange with /model provider:name, "
                           "e.g. /model anthropic:claude-opus-5-5")  # fmt: skip
        return
    previous = repl.session.model
    repl.session.model = arg
    try:
        repl.session.build()
    except Exception as exc:  # noqa: BLE001 - a bad model name must not end the session
        repl.session.model = previous
        repl.session.build()
        repl.console.print(Text(f"Could not load {arg}: {exc}", style="red"))
        return
    repl.console.print(Text(f"Model set to {arg}.", style="dim"))


async def _mode(repl: Repl, arg: str) -> None:
    approvals = repl.session.approvals
    if arg:
        if arg not in MODES:
            repl.console.print(Text(f"Modes: {', '.join(MODES)}", style="red"))
            return
        approvals.mode = arg
    else:
        for mode in MODES:
            marker = "●" if mode == approvals.mode else " "
            repl.console.print(f" {marker} [magenta]{mode:6}[/] {MODE_HELP[mode]}")
        if approvals.always:
            repl.console.print(
                f"   always allowed this session: {', '.join(sorted(approvals.always))}"
            )
        return
    repl.console.print(
        Text(f"Approval mode: {approvals.mode} ({MODE_HELP[approvals.mode]})", "dim")
    )


async def _cost(repl: Repl, _arg: str) -> None:
    u = repl.session.usage.session
    repl.console.print(
        f"Session: {format_tokens(u.input_tokens)} input tokens "
        f"({format_tokens(u.cache_read_tokens)} cached) · {format_tokens(u.output_tokens)} "
        f"output · {format_cost(u)}\n"
        f"Current context: {format_tokens(repl.session.usage.context_tokens)} tokens"
    )


async def _compact(repl: Repl, _arg: str) -> None:
    await repl.send(COMPACT_PROMPT, title="/compact")


async def _init(repl: Repl, _arg: str) -> None:
    await repl.send(INIT_PROMPT, title="/init")


async def _verbose(repl: Repl, _arg: str) -> None:
    renderer = repl.session.renderer
    renderer.verbose = not renderer.verbose
    repl.console.print(Text(f"Verbose output {'on' if renderer.verbose else 'off'}.", "dim"))


async def _exit(repl: Repl, _arg: str) -> None:
    repl.quit = True


COMMANDS: dict[str, Command] = {
    "help": Command("show commands and keys", _help),
    "new": Command("start a new conversation", _new),
    "clear": Command("start a new conversation", _new),
    "resume": Command("pick a saved conversation to continue", _resume, "[n|id]"),
    "model": Command("show or change the model", _model, "[provider:name]"),
    "mode": Command("show or set the approval mode (ask, edits, auto)", _mode, "[mode]"),
    "compact": Command("summarize older messages to free up context", _compact),
    "cost": Command("tokens and cost for this session", _cost),
    "init": Command("write /AGENTS.md with notes about this workspace", _init),
    "verbose": Command("toggle full tool output", _verbose),
    "exit": Command("quit", _exit),
    "quit": Command("quit", _exit),
}
ALIASES = {"clear", "quit"}


# -- completion --------------------------------------------------------------


class SlashCompleter(Completer):
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text or "\n" in text:
            return
        for name, cmd in COMMANDS.items():
            if name.startswith(text[1:]) and name not in ALIASES:
                yield Completion("/" + name, start_position=-len(text), display_meta=cmd.help)


class FileCompleter(Completer):
    """Completes @path against workspace files."""

    def __init__(self):
        self._files: list[str] = []
        self._scanned = 0.0

    def _workspace_files(self) -> list[str]:
        if time.monotonic() - self._scanned > 5:
            root = get_settings().workspace
            files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [
                    d for d in dirnames if d not in HIDDEN_DIRS and not d.startswith(".")
                ]
                rel = Path(dirpath).relative_to(root)
                files.extend(str(rel / f) if str(rel) != "." else f for f in filenames)
                if len(files) > 5000:
                    break
            self._files = sorted(files)
            self._scanned = time.monotonic()
        return self._files

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith("@"):
            return
        query = word[1:].lstrip("/").lower()
        matches = [f for f in self._workspace_files() if query in f.lower()]
        # Prefix matches first, then shorter paths.
        matches.sort(key=lambda f: (not f.lower().startswith(query), len(f)))
        for path in matches[:30]:
            yield Completion("@" + path, start_position=-len(word), display=path)
