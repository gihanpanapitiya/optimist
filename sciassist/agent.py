"""The main science-assistant agent."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .config import get_settings
from .graph import build_agent_graph
from .prompts import MAIN_AGENT
from .subagents import SubAgent, default_subagents, make_task_tool
from .tools import deep_literature_research, list_files, read_file, search_literature


def build_assistant(
    *,
    model: str | None = None,
    subagents: list[SubAgent] | None = None,
    checkpointer=None,
):
    """Compile the main agent with its tools and subagents.

    Pass a checkpointer to keep conversation history across turns (see
    :func:`persistent_assistant`); ``langgraph dev`` supplies its own.
    """
    tools = [
        search_literature,
        deep_literature_research,
        list_files,
        read_file,
        make_task_tool(subagents if subagents is not None else default_subagents()),
    ]
    return build_agent_graph(
        model or get_settings().model,
        tools,
        MAIN_AGENT,
        checkpointer=checkpointer,
        name="science-assistant",
    )


def make_graph():
    """Entry point for ``langgraph.json`` (LangGraph Studio / ``langgraph dev``)."""
    return build_assistant()


@asynccontextmanager
async def persistent_assistant() -> AsyncIterator:
    """The assistant with chat history stored in SQLite, so chats survive restarts.

    Used by both the web UI and the terminal chat; they share one database, so
    a thread id from either can be resumed in the other.
    """
    async with AsyncSqliteSaver.from_conn_string(str(get_settings().db_path)) as saver:
        yield build_assistant(checkpointer=saver)
