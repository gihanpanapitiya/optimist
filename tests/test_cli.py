"""The terminal: running turns, approval modes, rendering, attachments, headless mode."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from optimist.cli import app
from optimist.cli.approvals import Approvals
from optimist.cli.render import Renderer

from .conftest import ai, tool_call


def _console() -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(file=out, width=100, force_terminal=False, color_system=None), out


async def _session(mode: str = "edits"):
    from optimist.agent import open_checkpointer
    from optimist.cli.session import Session
    from optimist.config import get_settings
    from optimist.threads import ThreadStore

    console, out = _console()
    saver_cm = open_checkpointer()
    saver = await saver_cm.__aenter__()
    threads = await ThreadStore.open(get_settings().db_path)
    session = Session(
        saver=saver, threads=threads, renderer=Renderer(console), approvals=Approvals(mode=mode)
    )
    return session, out, saver_cm, threads


async def _close(saver_cm, threads):
    await threads.close()
    await saver_cm.__aexit__(None, None, None)


async def test_edits_mode_writes_without_asking_and_asks_for_shell(script, isolated):
    script(
        [
            ai(
                "",
                tool_call("write_file", {"file_path": "/exp/a.py", "content": "print(42)\n"}, "c1"),
            ),
            ai("", tool_call("execute", {"command": "python3 exp/a.py"}, "c2")),
            ai("It printed 42."),
        ]
    )
    session, out, saver_cm, threads = await _session("edits")
    asked = []

    async def ask(action):
        asked.append(action["name"])
        return {"type": "approve"}

    try:
        result = await session.run("run it", ask=ask, watch_keys=False)
    finally:
        await _close(saver_cm, threads)

    assert asked == ["execute"]
    assert (isolated / "exp" / "a.py").read_text() == "print(42)\n"
    assert result.text == "It printed 42."
    text = out.getvalue()
    assert "● write_file(/exp/a.py)" in text
    assert "new file · 1 lines" in text
    assert "42" in text  # shell output preview


async def test_ask_mode_rejection_reaches_the_agent(script, isolated):
    script(
        [
            ai("", tool_call("write_file", {"file_path": "/x.txt", "content": "x"}, "c1")),
            ai("OK, I won't."),
        ]
    )
    session, out, saver_cm, threads = await _session("ask")

    async def ask(action):
        return {"type": "reject", "message": "The user declined and said: not now"}

    try:
        result = await session.run("write x", ask=ask, watch_keys=False)
        state = await session.graph.aget_state(session.config)
    finally:
        await _close(saver_cm, threads)

    assert not (isolated / "x.txt").exists()
    assert result.text == "OK, I won't."
    tool_results = [m for m in state.values["messages"] if m.type == "tool"]
    assert "not now" in tool_results[0].text


async def test_always_stops_asking_for_that_tool(script, isolated):
    script(
        [
            ai("", tool_call("execute", {"command": "echo 1"}, "c1")),
            ai("", tool_call("execute", {"command": "echo 2"}, "c2")),
            ai("done"),
        ]
    )
    session, _, saver_cm, threads = await _session("ask")
    asked = []

    async def ask(action):
        asked.append(action["args"]["command"])
        return None  # "yes, and don't ask again"

    try:
        await session.run("go", ask=ask, watch_keys=False)
    finally:
        await _close(saver_cm, threads)

    assert asked == ["echo 1"]
    assert "execute" in session.approvals.always


async def test_edit_shows_a_diff_against_the_file(script, isolated):
    (isolated / "fit.py").write_text("alpha = 0.1\nbeta = 2\n")
    script(
        [
            ai("", tool_call("read_file", {"file_path": "/fit.py"}, "c0")),
            ai(
                "",
                tool_call(
                    "edit_file",
                    {
                        "file_path": "/fit.py",
                        "old_string": "alpha = 0.1",
                        "new_string": "alpha = 0.5",
                    },
                    "c1",
                ),
            ),
            ai("Changed alpha."),
        ]
    )
    session, out, saver_cm, threads = await _session("auto")
    try:
        await session.run("bump alpha", ask=None, watch_keys=False)
    finally:
        await _close(saver_cm, threads)

    text = out.getvalue()
    assert "/fit.py · +1 -1" in text
    assert "-alpha = 0.1" in text and "+alpha = 0.5" in text
    assert (isolated / "fit.py").read_text() == "alpha = 0.5\nbeta = 2\n"


async def test_subagent_activity_is_shown_and_usage_counted(script, isolated):
    script(
        main=[
            ai("", tool_call("task", {"description": "list", "subagent_type": "coder"}, "c1")),
            ai("The workspace is empty."),
        ],
        sub=[
            ai("", tool_call("ls", {"path": "/"}, "s1"), tokens=(50, 5)),
            ai("nothing there", tokens=(50, 5)),
        ],
    )
    session, out, saver_cm, threads = await _session("edits")
    try:
        result = await session.run("what's there?", ask=None, watch_keys=False)
    finally:
        await _close(saver_cm, threads)

    text = out.getvalue()
    assert "● task(coder: list)" in text
    assert "│ coder ls(/)" in text
    # Two main responses (100 in each) and two subagent responses (50 in each).
    assert result.usage.input_tokens == 300
    assert result.usage.cost > 0
    assert session.usage.context_tokens == 110


async def test_headless_rejects_what_needs_a_person(script, isolated):
    script(
        [
            ai("", tool_call("execute", {"command": "rm -rf /tmp/nothing"}, "c1")),
            ai("I could not run it."),
        ]
    )
    session, _, saver_cm, threads = await _session("edits")
    try:
        result = await session.run("clean up", ask=None, watch_keys=False)
        state = await session.graph.aget_state(session.config)
    finally:
        await _close(saver_cm, threads)

    assert result.text == "I could not run it."
    rejection = [m for m in state.values["messages"] if m.type == "tool"][0]
    assert "non-interactive" in rejection.text


def test_print_mode_outputs_json_and_reads_stdin(script, isolated):
    script([ai("The mean is 2.")])
    result = CliRunner().invoke(
        app, ["-p", "mean of these?", "--json"], input="1\n2\n3\n", catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["text"] == "The mean is 2."
    assert payload["usage"]["input_tokens"] == 100
    assert payload["thread_id"]


def test_bad_mode_is_rejected():
    result = CliRunner().invoke(app, ["-p", "x", "--mode", "yolo"])
    assert result.exit_code != 0


def test_file_mentions_are_attached(isolated):
    from optimist.cli.repl import Repl

    (isolated / "notes.md").write_text("hypothesis: H1\n")
    console, out = _console()
    repl = Repl.__new__(Repl)
    repl.console = console
    message = repl._attach_files("summarize @notes.md please")
    assert '<file path="/notes.md">\nhypothesis: H1\n' in message
    assert "attached /notes.md" in out.getvalue()


@pytest.mark.parametrize(
    ("mode", "tool", "auto"),
    [
        ("ask", "edit_file", False),
        ("edits", "edit_file", True),
        ("edits", "execute", False),
        ("auto", "delete", True),
    ],
)
def test_modes(mode, tool, auto):
    assert Approvals(mode=mode).auto_approves(tool) is auto


def test_mode_cycles():
    approvals = Approvals(mode="ask")
    assert [approvals.cycle() for _ in range(3)] == ["edits", "auto", "ask"]


async def test_interrupting_a_running_tool_leaves_a_usable_thread(script, isolated):
    import asyncio

    script(
        [
            ai("", tool_call("execute", {"command": "sleep 20"}, "c1")),
            ai("Picking up where we left off."),
        ]
    )
    session, _, saver_cm, threads = await _session("auto")
    try:
        run = asyncio.create_task(session.run("long job", ask=None, watch_keys=False))
        await asyncio.sleep(1.5)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        result = await session.run("continue", ask=None, watch_keys=False)
        state = await session.graph.aget_state(session.config)
    finally:
        await _close(saver_cm, threads)

    assert result.text == "Picking up where we left off."
    cancelled = [m for m in state.values["messages"] if m.type == "tool"]
    assert cancelled and cancelled[0].tool_call_id == "c1"
