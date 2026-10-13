"""Read-only access to the workspace.

Deep research writes reports and evidence files to disk instead of into the
conversation. These tools let the agent pull in just the parts it needs.
Paths are confined to the workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from ..config import get_settings

MAX_LINES = 400


def resolve_in_workspace(path: str) -> Path:
    workspace = get_settings().workspace
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"{path} is outside the workspace ({workspace})")
    return resolved


@tool
def list_files(
    path: Annotated[str, Field(description="Directory, relative to the workspace")] = ".",
) -> str:
    """List files under a workspace directory (recursive, up to 200 entries)."""
    try:
        root = resolve_in_workspace(path)
    except ValueError as exc:
        return str(exc)
    if not root.is_dir():
        return f"{path} is not a directory"
    workspace = get_settings().workspace
    entries = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".pdf")
    lines = [str(p.relative_to(workspace)) for p in entries[:200]]
    if len(entries) > 200:
        lines.append(f"... and {len(entries) - 200} more")
    return "\n".join(lines) or "(empty)"


@tool
def read_file(
    path: Annotated[str, Field(description="File path, relative to the workspace or absolute")],
    offset: Annotated[int, Field(description="First line to read, 1-based", ge=1)] = 1,
    limit: Annotated[int, Field(description="Number of lines", ge=1, le=MAX_LINES)] = 200,
) -> str:
    """Read lines from a text file in the workspace, e.g. a deep research report."""
    try:
        target = resolve_in_workspace(path)
    except ValueError as exc:
        return str(exc)
    if not target.is_file():
        return f"{path} does not exist"
    lines = target.read_text(errors="replace").splitlines()
    chunk = lines[offset - 1 : offset - 1 + limit]
    end = offset - 1 + len(chunk)
    header = f"{target.name}: lines {offset}-{end} of {len(lines)}"
    if end < len(lines):
        header += f" (continue with offset={end + 1})"
    return header + "\n" + "\n".join(chunk)
