"""Settings for the assistant.

Deliberately small. The deep-research pipeline (the ``optimistresearch``
package) keeps its own configuration in an ``optimistresearch.yaml`` it finds
by walking up from the working directory, or at ``OPTIMISTRESEARCH_CONFIG_FILE``;
this file only covers the conversational agent around it.

    OPTIMIST_MODEL            main agent model, "provider:model"
    OPTIMIST_SUBAGENT_MODEL   model for spawned subagents (default: main model)
    OPTIMIST_WORKSPACE        where research artifacts are written and read
    OPTIMIST_DB               SQLite file holding chat history (checkpoints)
    OPTIMIST_AUTO_APPROVE     "1" runs shell commands and deletions without asking
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model: str
    subagent_model: str
    workspace: Path
    db_path: Path
    auto_approve: bool

    @property
    def literature_dir(self) -> Path:
        """Deep-research corpora live here, one directory per question."""
        return self.workspace / "literature"


@cache
def get_settings() -> Settings:
    # load_dotenv never overrides a variable that is already set. API keys land in
    # os.environ here, which is also where optimistresearch's model clients look.
    load_dotenv(PROJECT_ROOT / ".env")

    model = os.environ.get("OPTIMIST_MODEL", "openai:gpt-5.4")
    workspace = Path(os.environ.get("OPTIMIST_WORKSPACE", PROJECT_ROOT / "workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    # Outside the workspace on purpose: the workspace is what the agent and the
    # UI's file browser see, and the chat database is neither's business.
    db_path = Path(os.environ.get("OPTIMIST_DB", PROJECT_ROOT / "data" / "chats.sqlite"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        model=model,
        subagent_model=os.environ.get("OPTIMIST_SUBAGENT_MODEL", model),
        workspace=workspace.resolve(),
        db_path=db_path.resolve(),
        auto_approve=os.environ.get("OPTIMIST_AUTO_APPROVE", "").lower() in {"1", "true", "yes"},
    )
