"""Subagents the main agent can spawn through deepagents' ``task`` tool.

Each ``task`` call starts a fresh agent with an empty context; the parent sees
only its final message, which keeps the parent's context small. Several calls
in one turn run in parallel.

Every subagent also gets the deepagents file tools and ``execute`` on the same
workspace, and inherits the main agent's approval rules.
"""

from __future__ import annotations

from deepagents import SubAgent

from .config import get_settings
from .prompts import CODER_SUBAGENT, GENERAL_SUBAGENT
from .tools import search_literature


def default_subagents() -> list[SubAgent]:
    # Subagents deliberately lack `deep_literature_research` (too slow and
    # costly to launch implicitly). Naming one "general-purpose" replaces the
    # deepagents default of the same name.
    model = get_settings().subagent_model
    return [
        SubAgent(
            name="general-purpose",
            description=(
                "General research worker with quick literature search and workspace "
                "file access. Use for any self-contained job: checking an idea's "
                "novelty, comparing methods, summarizing a set of papers or files."
            ),
            system_prompt=GENERAL_SUBAGENT,
            tools=[search_literature],
            model=model,
        ),
        SubAgent(
            name="coder",
            description=(
                "Writes, runs and debugs code in the workspace: data analysis, "
                "simulations, experiment scripts, plots. Give it the goal, the "
                "inputs, and what output to produce; it returns what it changed, "
                "what it ran, and the results."
            ),
            system_prompt=CODER_SUBAGENT,
            tools=[],
            model=model,
        ),
    ]
