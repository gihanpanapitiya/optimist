"""How a run looks in the terminal.

Text streams into a live area at the bottom of the screen together with a
status line. Everything final (finished text, tool calls and their results)
is printed above it, so the scrollback reads like a transcript:

    ● execute(python exp/fit.py)
      ⎿  r2 = 0.93  (+12 lines)

Subagent activity is shown indented under the ``task`` call that started it.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from ..tools.files import resolve_in_workspace
from .usage import Usage, format_cost, format_tokens

RESULT_PREVIEW_LINES = 4
_EXECUTE_STATUS = re.compile(
    r"\n*(\[Command (succeeded|failed) with exit code -?\d+\]|Exit code: -?\d+|<no output>)\s*"
)
DIFF_MAX_LINES = 60
NEW_FILE_PREVIEW_LINES = 20


class Renderer:
    """Draws one session's runs. ``verbose`` shows full tool output."""

    def __init__(self, console: Console, *, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self._live: Live | None = None
        self._stream_text = ""
        self._status = ""
        self._started = 0.0
        self._running: dict[str, str] = {}  # tool_call_id -> label
        self._subagent_names: dict[tuple, str] = {}

    # -- live area ---------------------------------------------------------

    def start(self, status: str = "Thinking") -> None:
        self._started = time.monotonic()
        self._status = status
        self._stream_text = ""
        if not self.console.is_terminal:
            return  # piped or headless: no spinner, just the transcript
        self._live = Live(
            self._live_view(),
            console=self.console,
            refresh_per_second=10,
            transient=True,
            get_renderable=self._live_view,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._running.clear()

    def _live_view(self) -> RenderableType:
        parts: list[RenderableType] = []
        if self._stream_text:
            # Only the tail: a live area taller than the terminal can't be redrawn.
            height = max(self.console.size.height - 4, 4)
            lines = self._stream_text.splitlines()[-height:]
            parts.append(Markdown("\n".join(lines)))
        elapsed = int(time.monotonic() - self._started)
        status = self._status
        if self._running:
            status = "Running " + ", ".join(dict.fromkeys(self._running.values()))
        parts.append(
            Spinner(
                "dots",
                text=Text.assemble(
                    (f"{status}… ", "bold cyan"), (f"({elapsed}s · esc to interrupt)", "dim")
                ),
            )
        )
        return Group(*parts)

    # -- events ------------------------------------------------------------

    def token(self, text: str) -> None:
        self._stream_text += text

    def ai_message(self, message: AIMessage, ns: tuple) -> None:
        if ns:
            self._subagent_message(message, ns)
            return
        self._stream_text = ""
        if message.text.strip():
            self.console.print()
            self.console.print(Markdown(message.text))
            self.console.print()
        for call in message.tool_calls:
            self.tool_call(call)

    def tool_call(self, call: Mapping) -> None:
        name, args = call["name"], call.get("args") or {}
        self._running[call["id"]] = name
        self.console.print(
            Text.assemble(("● ", "bold green"), (name, "bold"), call_summary(name, args))
        )
        detail = _call_detail(name, args)
        if detail is not None:
            self.console.print(detail)

    def tool_result(self, message: ToolMessage, ns: tuple) -> None:
        if ns:
            self._subagent_result(message, ns)
            return
        self._running.pop(message.tool_call_id, None)
        failed = message.status == "error"
        text = message.text
        if message.name == "execute":
            code = (
                (message.artifact or {}).get("exit_code")
                if isinstance(message.artifact, dict)
                else None
            )
            failed = failed or code not in (0, None)
            # The status lines are for the model; the terminal shows the code only on failure.
            text = _EXECUTE_STATUS.sub("", text).rstrip() or "(no output)"
            if failed and code is not None:
                text += f"\nexit code {code}"
        summary = _result_summary(message.name, text, self.verbose)
        self.console.print(
            Text("  ⎿  ", style="dim") + Text(summary, style="red" if failed else "dim")
        )

    def progress(self, event: dict) -> None:
        message = str(event.get("message", event))
        self._status = message[:100]
        if event.get("tool") == "deep_literature_research" or self.verbose:
            self.console.print(Text(f"     · {message}", style="dim"))

    def _subagent_message(self, message: AIMessage, ns: tuple) -> None:
        name = message.name or self._subagent_names.get(ns[:1], "subagent")
        self._subagent_names[ns[:1]] = name
        for call in message.tool_calls:
            self.console.print(
                Text.assemble(
                    ("    │ ", "dim"),
                    (f"{name} ", "magenta"),
                    (call["name"], "bold"),
                    call_summary(call["name"], call.get("args") or {}),
                )
            )
        self._status = f"{name} working"

    def _subagent_result(self, message: ToolMessage, ns: tuple) -> None:
        if message.status == "error" or self.verbose:
            summary = _result_summary(message.name, message.text, self.verbose)
            style = "red" if message.status == "error" else "dim"
            self.console.print(Text("    │   ⎿  ", style="dim") + Text(summary, style=style))

    def notice(self, text: str, style: str = "yellow") -> None:
        self.console.print(Text(text, style=style))

    def turn_footer(self, usage: Usage, seconds: float) -> None:
        if not usage.total_tokens:
            return
        self.console.print(
            Text(
                f"  {format_tokens(usage.input_tokens)} in · {format_tokens(usage.output_tokens)} "
                f"out · {format_cost(usage)} · {seconds:.0f}s",
                style="dim",
            )
        )


# -- formatting -------------------------------------------------------------


def call_summary(name: str, args: dict) -> Text:
    key = {
        "execute": "command",
        "read_file": "file_path",
        "write_file": "file_path",
        "edit_file": "file_path",
        "delete": "file_path",
        "ls": "path",
        "glob": "pattern",
        "grep": "pattern",
        "search_literature": "query",
        "deep_literature_research": "question",
        "task": "description",
    }.get(name)
    if key and key in args:
        value = str(args[key])
    else:
        value = json.dumps(args, ensure_ascii=False)
    if name == "task":
        value = f"{args.get('subagent_type', 'general-purpose')}: {value}"
    value = " ".join(value.split())
    if len(value) > 120:
        value = value[:117] + "…"
    return Text(f"({value})", style="default")


def _call_detail(name: str, args: dict) -> RenderableType | None:
    """Extra lines under a call: diffs for file changes, the full shell command."""
    if name == "edit_file":
        return edit_diff(args)
    if name == "write_file":
        return write_preview(args)
    if name == "execute" and (
        "\n" in args.get("command", "") or len(args.get("command", "")) > 120
    ):
        return Panel(
            Syntax(args["command"], "bash", word_wrap=True), border_style="dim", expand=False
        )
    return None


def _result_summary(name: str | None, text: str, verbose: bool) -> str:
    text = text.rstrip()
    lines = text.splitlines()
    if verbose:
        return text
    if name == "read_file":
        return f"Read {len(lines)} lines"
    if name in {"ls", "glob"}:
        try:
            entries = ast.literal_eval(text)  # these tools return a Python list repr
        except (ValueError, SyntaxError):
            entries = lines
        return f"{len(entries)} entries" if isinstance(entries, list) else text[:200]
    if name == "search_literature" and lines:
        return lines[0]
    if name == "task":
        return "Done" + (f": {lines[0][:100]}" if lines else "")
    shown = lines[:RESULT_PREVIEW_LINES]
    out = "\n     ".join(line[:200] for line in shown)
    if len(lines) > len(shown):
        out += f"\n     … +{len(lines) - len(shown)} lines (/verbose to show)"
    return out or "(no output)"


def _read_workspace(file_path: str) -> str | None:
    try:
        path = resolve_in_workspace(file_path)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def edit_diff(args: dict) -> RenderableType | None:
    old_text = _read_workspace(args.get("file_path", ""))
    old, new = args.get("old_string", ""), args.get("new_string", "")
    if old_text is None or old not in old_text:
        return colored_diff(old.splitlines(), new.splitlines(), args.get("file_path", ""))
    count = -1 if args.get("replace_all") else 1
    new_text = old_text.replace(old, new, count)
    return colored_diff(old_text.splitlines(), new_text.splitlines(), args.get("file_path", ""))


def write_preview(args: dict) -> RenderableType | None:
    path, content = args.get("file_path", ""), args.get("content", "")
    existing = _read_workspace(path)
    if existing is not None:
        return colored_diff(existing.splitlines(), content.splitlines(), path)
    lines = content.splitlines()
    shown = "\n".join(lines[:NEW_FILE_PREVIEW_LINES])
    lexer = Syntax.guess_lexer(Path(path).name, shown)
    body: list[RenderableType] = [Syntax(shown, lexer, line_numbers=True, word_wrap=True)]
    if len(lines) > NEW_FILE_PREVIEW_LINES:
        body.append(Text(f"… +{len(lines) - NEW_FILE_PREVIEW_LINES} lines", style="dim"))
    return Panel(
        Group(*body),
        title=f"new file · {len(lines)} lines",
        title_align="left",
        border_style="green",
        expand=False,
    )


def colored_diff(before: list[str], after: list[str], path: str) -> RenderableType:
    out = Text()
    added = removed = shown = 0
    for line in difflib.unified_diff(before, after, lineterm="", n=2):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
        if shown >= DIFF_MAX_LINES:
            continue
        shown += 1
        if line.startswith("@@"):
            out.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            out.append(line + "\n", style="green")
        elif line.startswith("-"):
            out.append(line + "\n", style="red")
        else:
            out.append(line + "\n", style="dim")
    total = added + removed
    if total > shown:
        out.append(f"… {total - shown} more changed lines\n", style="dim")
    title = f"{path} · +{added} -{removed}"
    out.rstrip()
    return Panel(
        out if out else Text("(no change)", style="dim"),
        title=title,
        title_align="left",
        border_style="dim",
        expand=False,
    )
