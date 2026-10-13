"""Workspace paths.

The agent's file tools come from deepagents and run in virtual mode: the
workspace is the agent's filesystem root, so ``/literature/x/final_report.md``
means ``<workspace>/literature/x/final_report.md``. Tools that write files
report paths in that form so the agent can pass them straight to
``read_file``. The web UI's file API uses :func:`resolve_in_workspace`.
"""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings


def resolve_in_workspace(path: str) -> Path:
    """Map a relative path, a virtual ``/path`` or a host path into the workspace."""
    workspace = get_settings().workspace
    candidate = Path(path)
    if candidate.is_absolute() and not candidate.is_relative_to(workspace):
        candidate = Path(path.lstrip("/"))
    resolved = (workspace / candidate).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"{path} is outside the workspace ({workspace})")
    return resolved


def to_agent_path(path: str | Path) -> str:
    """A host path inside the workspace, as the agent's file tools see it."""
    workspace = get_settings().workspace
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(workspace):
        return str(path)
    return "/" + resolved.relative_to(workspace).as_posix()
