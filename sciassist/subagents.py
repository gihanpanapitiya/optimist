"""Subagents, and the ``task`` tool the main agent uses to spawn them.

The shape mirrors langchain ``deepagents`` (a registry of named subagents, a
``task(description, subagent_type)`` tool, and a general-purpose default), so
moving to deepagents later is mostly a matter of passing the same specs to
``create_deep_agent(subagents=...)``.

Each call to ``task`` builds a fresh agent with an empty context. The parent
sees only the subagent's final message, which keeps the parent's context
small. Multiple ``task`` calls in one turn run in parallel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field

from .config import get_settings
from .graph import build_agent_graph
from .prompts import GENERAL_SUBAGENT
from .tools import list_files, read_file, search_literature

logger = logging.getLogger(__name__)

SUBAGENT_RECURSION_LIMIT = 60


@dataclass
class SubAgent:
    name: str
    description: str
    """Shown to the main agent so it can pick the right subagent."""
    system_prompt: str
    tools: list[BaseTool] = field(default_factory=list)
    model: str | None = None
    """Overrides SCIASSIST_SUBAGENT_MODEL for this subagent."""


def default_subagents() -> list[SubAgent]:
    # Subagents deliberately lack `task` (no recursive spawning) and
    # `deep_literature_research` (too slow and costly to launch implicitly).
    return [
        SubAgent(
            name="general-purpose",
            description=(
                "General research worker with quick literature search and workspace "
                "file reading. Use for any self-contained job: checking an idea's "
                "novelty, comparing methods, summarizing a set of papers or files."
            ),
            system_prompt=GENERAL_SUBAGENT,
            tools=[search_literature, read_file, list_files],
        ),
    ]


def make_task_tool(subagents: list[SubAgent]) -> BaseTool:
    registry = {sa.name: sa for sa in subagents}
    compiled: dict = {}
    settings = get_settings()

    def graph_for(name: str):
        if name not in compiled:
            spec = registry[name]
            # checkpointer=False: a subagent run is ephemeral and must not
            # write into the parent conversation's checkpoints.
            compiled[name] = build_agent_graph(
                spec.model or settings.subagent_model,
                spec.tools,
                spec.system_prompt,
                checkpointer=False,
                name=name,
            )
        return compiled[name]

    async def run_task(
        description: Annotated[
            str,
            Field(description="Complete, self-contained instructions, including the goal, "
                              "relevant context, and what the final answer should contain"),
        ],
        subagent_type: Annotated[str, Field(description="Which subagent to use")] = (
            "general-purpose"
        ),
        runtime: ToolRuntime = None,
    ) -> str:
        if subagent_type not in registry:
            return f"Unknown subagent {subagent_type!r}. Available: {', '.join(registry)}"
        _progress(runtime, f"[{subagent_type}] started: {description[:100]}")
        result = await graph_for(subagent_type).ainvoke(
            {"messages": [HumanMessage(description)]},
            # Passing the parent config through keeps tracing nested under
            # the parent run.
            config={**runtime.config, "recursion_limit": SUBAGENT_RECURSION_LIMIT},
        )
        _progress(runtime, f"[{subagent_type}] finished")
        return result["messages"][-1].text

    catalogue = "\n".join(f"- {sa.name}: {sa.description}" for sa in subagents)
    return StructuredTool.from_function(
        coroutine=run_task,
        name="task",
        description=(
            "Delegate a self-contained task to a subagent that works in its own context "
            "and returns only its final answer. Issue several task calls in one turn to "
            "run independent tasks in parallel.\n\nAvailable subagents:\n" + catalogue
        ),
    )


def _progress(runtime: ToolRuntime, message: str) -> None:
    try:
        runtime.stream_writer(
            {"tool": "task", "message": message, "tool_call_id": runtime.tool_call_id}
        )
    except Exception:  # noqa: BLE001 - progress must never break a tool
        pass
