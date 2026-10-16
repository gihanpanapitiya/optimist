"""The workspace shell: output format, timeouts, and killing on cancellation."""

from __future__ import annotations

import asyncio

import pytest

from optimist.agent import make_backend


async def test_output_stderr_and_exit_code(isolated):
    result = await make_backend().aexecute("echo out; echo err >&2; exit 3")
    assert result.exit_code == 3
    assert result.output == "out\n\n[stderr] err\n\nExit code: 3"


async def test_runs_in_workspace_without_secrets(isolated, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    result = await make_backend().aexecute("pwd; echo key=$OPENAI_API_KEY")
    assert result.output.splitlines() == [str(isolated), "key="]


async def test_timeout(isolated):
    result = await make_backend().aexecute("sleep 5", timeout=1)
    assert result.exit_code == 124
    assert "timed out after 1 seconds" in result.output


async def test_cancel_kills_the_command_and_its_children(isolated):
    marker = isolated / "finished"
    # The write happens in a child of a child: killing only the shell would miss it.
    task = asyncio.create_task(
        make_backend().aexecute(f"(sleep 2; touch {marker}) & wait", timeout=30)
    )
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(2.5)
    assert not marker.exists()
