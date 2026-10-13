"""Web UI backend.

    sciassist ui                # http://127.0.0.1:8000

The chat endpoint streams the agent run as server-sent events, one JSON object
per event:

    {"type": "token", "text": ...}                     main-agent text, as generated
    {"type": "message", "text": ...}                   the finished main-agent message
    {"type": "tool_call", "id", "name", "args"}
    {"type": "progress", "tool", "message"}            custom events from tools
    {"type": "tool_result", "id", "name", "content", "status"}
    {"type": "approval", "interrupts": [{"id", "actions": [{"name", "args"}]}]}
    {"type": "error", "message"} / {"type": "done"}

An ``approval`` event ends the run with tool calls waiting on the user (shell
commands and deletions, see ``agent.APPROVAL_REQUIRED``). The client answers by
posting to ``/api/chat`` again with ``decisions`` instead of a message.

Only the main agent's tokens are streamed. Subagents run inside the ``task``
tool, so their LLM calls show up in the token stream too; they are filtered
out by checkpoint namespace, and the parent sees their work as the tool result.
"""

from __future__ import annotations

import json
import logging
import mimetypes
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from .agent import persistent_assistant
from .config import get_settings
from .threads import ThreadStore
from .tools.files import resolve_in_workspace

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_TEXT_BYTES = 2_000_000
TEXT_SUFFIXES = {
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".py", ".js",
    ".ts", ".html", ".css", ".sh", ".log", ".bib", ".tex", ".xml", ".ipynb", ".r", ".jl",
}
HIDDEN = {".git", "__pycache__", ".DS_Store", ".ipynb_checkpoints"}

_graph = None
_threads: ThreadStore | None = None
# Threads with a run in progress. A second run on the same thread would
# interleave messages in one checkpoint, e.g. from two tabs or a re-sent message.
_active_threads: set[str] = set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _graph, _threads
    from litresearch.sources.base import close_http_client

    async with persistent_assistant() as graph:
        _graph = graph
        # Creates the checkpoint tables on a fresh database, which backfill reads.
        await graph.checkpointer.setup()
        _threads = await ThreadStore.open(get_settings().db_path)
        indexed = await _threads.backfill(graph)
        if indexed:
            logger.info("indexed %d existing conversations", indexed)
        try:
            yield
        finally:
            await _threads.close()
            await close_http_client()


app = FastAPI(title="sciassist", lifespan=lifespan)


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class Decision(BaseModel):
    type: Literal["approve", "reject"]
    message: str | None = None


class ChatRequest(BaseModel):
    thread_id: str
    message: str = ""
    decisions: dict[str, list[Decision]] | None = None
    """Answers to pending approvals: interrupt id -> one decision per action."""


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if req.decisions is not None:
        resume = {
            interrupt_id: {"decisions": [d.model_dump(exclude_none=True) for d in decisions]}
            for interrupt_id, decisions in req.decisions.items()
        }
        graph_input = Command(resume=resume)
    elif req.message.strip():
        graph_input = {"messages": [HumanMessage(req.message)]}
    else:
        raise HTTPException(400, "send a message or decisions")
    return StreamingResponse(
        _sse(_run(req.thread_id, graph_input, req.message)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for event in events:
        yield f"data: {json.dumps(event, default=str)}\n\n"


async def _run(thread_id: str, graph_input, message: str) -> AsyncIterator[dict]:
    if thread_id in _active_threads:
        yield {"type": "error", "message": "This conversation is still running. "
                                           "Stop it or wait for it to finish."}
        yield {"type": "done"}
        return
    _active_threads.add(thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}
    await _threads.touch(thread_id, message)
    yield {"type": "thread", "id": thread_id}
    try:
        async for mode, chunk in _graph.astream(
            graph_input,
            config=config,
            stream_mode=["messages", "updates", "custom"],
        ):
            if mode == "messages":
                msg, meta = chunk
                if (
                    isinstance(msg, AIMessageChunk)
                    and meta.get("langgraph_node") == "model"
                    and _is_main_agent(meta)
                    and msg.text
                ):
                    yield {"type": "token", "text": msg.text}
            elif mode == "custom":
                yield {"type": "progress", **chunk}
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    yield _approval_event(chunk["__interrupt__"])
                    continue
                for node, update in chunk.items():
                    for event in _update_events(node, update):
                        yield event
    except Exception as exc:  # noqa: BLE001 - report to the client, keep the server up
        logger.exception("chat run failed")
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        # Runs on cancellation too (client disconnect / Stop).
        _active_threads.discard(thread_id)
        await _threads.touch(thread_id, message)
    yield {"type": "done"}


def _is_main_agent(meta: dict) -> bool:
    # A subagent's namespace is nested under the parent's tools node,
    # e.g. "tools:<id>|model:<id>".
    return "|" not in (meta.get("langgraph_checkpoint_ns") or "")


def _update_events(node: str, update: dict | None) -> list[dict]:
    """Events for one node's state update.

    Middleware nodes also write messages. The approval node re-emits the
    model's message on resume, so only its tool results (rejections) count.
    """
    messages = (update or {}).get("messages", [])
    if node in {"model", "tools"}:
        return [e for m in messages for e in _message_events(m)]
    if node == "HumanInTheLoopMiddleware.after_model":
        return [e for m in messages if isinstance(m, ToolMessage) for e in _message_events(m)]
    return []


def _approval_event(interrupts) -> dict:
    return {
        "type": "approval",
        "interrupts": [
            {
                "id": i.id,
                "actions": [
                    {"name": a["name"], "args": a["args"]}
                    for a in (i.value or {}).get("action_requests", [])
                ],
            }
            for i in interrupts
        ],
    }


def _message_events(msg) -> list[dict]:
    if isinstance(msg, AIMessage):
        events = [{"type": "message", "text": msg.text}]
        events += [
            {"type": "tool_call", "id": c["id"], "name": c["name"], "args": c["args"]}
            for c in msg.tool_calls
        ]
        return events
    if isinstance(msg, ToolMessage):
        return [
            {
                "type": "tool_result",
                "id": msg.tool_call_id,
                "name": msg.name,
                "content": msg.text,
                "status": msg.status,
            }
        ]
    return []


@app.get("/api/history")
async def history(thread_id: str) -> list[dict]:
    """The thread's messages, in the same event vocabulary the stream uses."""
    state = await _graph.aget_state({"configurable": {"thread_id": thread_id}})
    out: list[dict] = []
    for msg in (state.values or {}).get("messages", []):
        if isinstance(msg, HumanMessage):
            out.append({"type": "user", "text": msg.text})
        else:
            out.extend(_message_events(msg))
    if state.interrupts and thread_id not in _active_threads:
        out.append(_approval_event(state.interrupts))
    return out


@app.get("/api/threads")
async def list_threads() -> list[dict]:
    threads = await _threads.list()
    for thread in threads:
        thread["running"] = thread["id"] in _active_threads
    return threads


class RenameRequest(BaseModel):
    title: str


@app.patch("/api/threads/{thread_id}")
async def rename_thread(thread_id: str, req: RenameRequest) -> dict:
    if not req.title.strip():
        raise HTTPException(400, "title must not be empty")
    if not await _threads.rename(thread_id, req.title):
        raise HTTPException(404, "no such conversation")
    return {"ok": True}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    if thread_id in _active_threads:
        raise HTTPException(409, "this conversation is still running")
    await _graph.checkpointer.adelete_thread(thread_id)
    await _threads.delete(thread_id)
    return {"ok": True}


@app.get("/api/info")
async def info() -> dict:
    settings = get_settings()
    return {"model": settings.model, "workspace": str(settings.workspace)}


# --------------------------------------------------------------------------
# Workspace files
# --------------------------------------------------------------------------


@app.get("/api/tree")
async def tree() -> dict:
    workspace = get_settings().workspace
    return _node(workspace, workspace)


def _node(path: Path, root: Path) -> dict:
    rel = "" if path == root else str(path.relative_to(root))
    if path.is_dir():
        children = sorted(
            (p for p in path.iterdir() if p.name not in HIDDEN),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        return {
            "name": path.name,
            "path": rel,
            "type": "dir",
            "children": [_node(c, root) for c in children],
        }
    stat = path.stat()
    return {
        "name": path.name,
        "path": rel,
        "type": "file",
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


@app.get("/api/file")
async def file(path: str = Query(...)) -> dict:
    target = _workspace_file(path)
    workspace = get_settings().workspace
    kind = _kind(target)
    result = {
        "path": str(target.relative_to(workspace)),
        "name": target.name,
        "kind": kind,
        "size": target.stat().st_size,
    }
    if kind in {"markdown", "json", "text"}:
        data = target.read_bytes()
        result["truncated"] = len(data) > MAX_TEXT_BYTES
        result["content"] = data[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
    return result


@app.get("/api/raw")
async def raw(path: str = Query(...)) -> FileResponse:
    target = _workspace_file(path)
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type)


def _workspace_file(path: str) -> Path:
    try:
        target = resolve_in_workspace(path)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not target.is_file():
        raise HTTPException(404, f"{path} not found")
    return target


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix == ".json":
        return "json"
    if suffix == ".pdf":
        return "pdf"
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime.startswith("image/"):
        return "image"
    if suffix in TEXT_SUFFIXES or mime.startswith("text/"):
        return "text"
    return "binary"


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
