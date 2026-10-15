"""Run the real CLI against a scripted model, for terminal tests.

SCENARIO picks the script. Used by test_repl_tty.py through a pseudo-terminal.
"""

from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import AIMessage

import forest.cli.session as session_mod
from forest.cli import app
from tests.conftest import ScriptedModel, ai, tool_call


class SlowModel(ScriptedModel):
    async def _agenerate(self, *args, **kwargs):
        await asyncio.sleep(30)
        return await super()._agenerate(*args, **kwargs)


SCENARIOS = {
    "approve": [
        ai("", tool_call("execute", {"command": "echo approved-output"}, "c1")),
        ai("The command ran."),
    ],
    "slow": [AIMessage("never shown")],
}


def main() -> None:
    scenario = os.environ["SCENARIO"]
    model_cls = SlowModel if scenario == "slow" else ScriptedModel
    replies = iter(SCENARIOS[scenario] + [ai("Second answer.")])
    real_build = session_mod.build_assistant

    def build(*, model, subagents, **kwargs):
        return real_build(model=model_cls(messages=replies), subagents=subagents, **kwargs)

    session_mod.build_assistant = build
    sys.argv = ["forest", *sys.argv[1:]]
    app()


if __name__ == "__main__":
    main()
