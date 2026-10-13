"""Settings for the assistant.

Deliberately small. The deep-research pipeline keeps its own configuration in
``literature/litresearch.yaml``; this file only covers the conversational
agent around it.

    SCIASSIST_MODEL            main agent model, "provider:model"
    SCIASSIST_SUBAGENT_MODEL   model for spawned subagents (default: main model)
    SCIASSIST_WORKSPACE        where research artifacts are written and read
    SCIASSIST_DB               SQLite file holding chat history (checkpoints)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LITERATURE_ROOT = PROJECT_ROOT.parent / "literature"


@dataclass(frozen=True)
class Settings:
    model: str
    subagent_model: str
    workspace: Path
    db_path: Path

    @property
    def literature_dir(self) -> Path:
        """Deep-research corpora live here, one directory per question."""
        return self.workspace / "literature"


@cache
def get_settings() -> Settings:
    # Our own .env first, then litresearch's, so API keys only need to be set
    # once. load_dotenv never overrides a variable that is already set.
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(LITERATURE_ROOT / ".env")

    model = os.environ.get("SCIASSIST_MODEL", "openai:gpt-5.4")
    workspace = Path(os.environ.get("SCIASSIST_WORKSPACE", PROJECT_ROOT / "workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    # Outside the workspace on purpose: the workspace is what the agent and the
    # UI's file browser see, and the chat database is neither's business.
    db_path = Path(os.environ.get("SCIASSIST_DB", PROJECT_ROOT / "data" / "chats.sqlite"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        model=model,
        subagent_model=os.environ.get("SCIASSIST_SUBAGENT_MODEL", model),
        workspace=workspace.resolve(),
        db_path=db_path.resolve(),
    )
