"""The main science-assistant agent, built on deepagents.

deepagents supplies the loop and the harness: file tools (``ls``,
``read_file``, ``write_file``, ``edit_file``, ``glob``, ``grep``), shell
``execute``, the ``task`` tool for subagents, approval interrupts,
summarization of long threads, offloading of large tool results, and repair
of tool calls left dangling by a cancelled run. This module adds the
literature tools, the prompt, the subagents and the workspace.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .config import get_settings
from .prompts import MAIN_AGENT
from .subagents import default_subagents
from .tools import deep_literature_research, search_literature

SHELL_TIMEOUT_SECONDS = 300

# Passed through to shell commands. Everything else, API keys included, stays
# out of the agent's environment.
SHELL_ENV_VARS = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TERM", "SHELL")

# Tools that pause for the user's approval unless SCIASSIST_AUTO_APPROVE is
# set. Writes and edits are confined to the workspace; a shell command is not,
# since root_dir only sets its working directory.
APPROVAL_REQUIRED = ("execute", "delete")


def make_backend() -> LocalShellBackend:
    """Files and shell, rooted at the workspace.

    virtual_mode makes the workspace the agent's ``/`` and rejects paths that
    escape it.
    """
    return LocalShellBackend(
        root_dir=get_settings().workspace,
        virtual_mode=True,
        timeout=SHELL_TIMEOUT_SECONDS,
        env={k: os.environ[k] for k in SHELL_ENV_VARS if k in os.environ},
    )


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
    settings = get_settings()
    approvals = {name: {"allowed_decisions": ["approve", "reject"]} for name in APPROVAL_REQUIRED}
    return create_deep_agent(
        model=model or settings.model,
        tools=[search_literature, deep_literature_research],
        system_prompt=MAIN_AGENT,
        subagents=subagents if subagents is not None else default_subagents(),
        backend=make_backend(),
        interrupt_on=None if settings.auto_approve else approvals,
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
