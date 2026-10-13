"""Shell execution that stops when the run stops.

deepagents' ``LocalShellBackend`` runs commands with ``subprocess.run`` in a
worker thread, so cancelling the agent (Esc in the terminal, Stop in the web
UI) leaves the command running to completion. This backend runs them as
asyncio subprocesses in their own process group and kills the group on
cancellation or timeout. Output formatting matches the parent class.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse


class WorkspaceShell(LocalShellBackend):
    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.", exit_code=1, truncated=False
            )
        limit = timeout if timeout is not None else self._default_timeout
        if limit <= 0:
            raise ValueError(f"timeout must be positive, got {limit}")

        process = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            cwd=str(self.cwd),
            start_new_session=sys.platform != "win32",
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit)
        except TimeoutError:
            await _kill(process)
            hint = "" if timeout is not None else " For long-running commands, pass a timeout."
            return ExecuteResponse(
                output=f"Error: Command timed out after {limit} seconds.{hint}",
                exit_code=124,
                truncated=False,
            )
        except asyncio.CancelledError:
            await _kill(process)
            raise
        return self._response(
            stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode
        )

    def _response(self, stdout: str, stderr: str, code: int) -> ExecuteResponse:
        parts = [stdout] if stdout else []
        if stderr:
            parts.extend(f"[stderr] {line}" for line in stderr.strip().split("\n"))
        output = "\n".join(parts) if parts else "<no output>"
        truncated = len(output) > self._max_output_bytes
        if truncated:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
        if code != 0:
            output = f"{output.rstrip()}\n\nExit code: {code}"
        return ExecuteResponse(output=output, exit_code=code, truncated=truncated)


async def _kill(process: asyncio.subprocess.Process) -> None:
    """Kill the command and anything it started."""
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), timeout=5)
