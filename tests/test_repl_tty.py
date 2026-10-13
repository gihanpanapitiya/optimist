"""The interactive session in a real pseudo-terminal: keys, approvals, interrupts."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pexpect = pytest.importorskip("pexpect")

ROOT = Path(__file__).resolve().parent.parent


def _spawn(scenario: str, tmp_path):
    env = dict(
        os.environ,
        SCENARIO=scenario,
        SCIASSIST_WORKSPACE=str(tmp_path / "ws"),
        SCIASSIST_DB=str(tmp_path / "db" / "chats.sqlite"),
        TERM="xterm-256color",
        PROMPT_TOOLKIT_NO_CPR="1",
    )
    child = pexpect.spawn(
        sys.executable, ["-m", "tests.fake_cli"], cwd=str(ROOT), env=env,
        encoding="utf-8", dimensions=(40, 120), timeout=20,
    )  # fmt: skip
    child.expect("> ")
    return child


def test_mode_cycling_approval_and_commands(tmp_path):
    child = _spawn("approve", tmp_path)
    child.send("\x1b[Z")  # Shift+Tab: edits -> auto
    time.sleep(0.2)
    child.send("\x1b[Z")  # auto -> ask
    time.sleep(0.2)
    child.send("/mode\r")
    child.expect(r"● (\x1b\[[0-9;]*m)*ask")  # the current mode is marked
    child.send("run something\r")
    child.expect("Allow")
    child.expect("› ")
    child.send("y\r")
    child.expect("approved-output")
    child.expect("The command ran.")
    child.expect(r"\d+ in · \d+ out")
    child.send("/cost\r")
    child.expect("Current context")
    child.send("/exit\r")
    child.expect("Resume this chat with: sciassist --resume")
    child.expect(pexpect.EOF)


def test_esc_and_ctrl_c_interrupt_without_quitting(tmp_path):
    child = _spawn("slow", tmp_path)
    for key in ("\x1b", "\x03"):  # Esc, then Ctrl-C
        child.send("think hard\r")
        child.expect("esc to interrupt")
        time.sleep(0.5)
        started = time.monotonic()
        child.send(key)
        child.expect("Interrupted")
        assert time.monotonic() - started < 3
        child.expect("> ")
    child.send("\x03")
    child.expect("Press Ctrl-C again")
    child.send("\x03")
    child.expect(pexpect.EOF)
