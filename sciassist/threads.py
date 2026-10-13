"""The conversation index behind the UI's chat list.

The checkpointer stores each thread's full state but has no cheap way to list
threads with a title and a last-active time, so that lives in a small table in
the same SQLite file. The checkpoints stay the source of truth: deleting a
thread removes both, and :meth:`ThreadStore.backfill` indexes any thread the
table doesn't know about (e.g. chats from before this table existed).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import aiosqlite
from langchain_core.messages import HumanMessage

TITLE_CHARS = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id  TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""


def title_from(message: str) -> str:
    text = " ".join(message.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1].rstrip() + "…"


class ThreadStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: Path) -> ThreadStore:
        conn = await aiosqlite.connect(path)
        conn.row_factory = aiosqlite.Row
        await conn.execute(SCHEMA)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    async def list(self) -> list[dict]:
        async with self._conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads "
            "ORDER BY updated_at DESC"
        ) as cursor:
            return [
                {"id": r["thread_id"], "title": r["title"],
                 "created_at": r["created_at"], "updated_at": r["updated_at"]}
                for r in await cursor.fetchall()
            ]

    async def touch(self, thread_id: str, first_message: str) -> None:
        """Record activity; a new thread is titled from its first message."""
        now = time.time()
        await self._conn.execute(
            "INSERT INTO threads (thread_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at",
            (thread_id, title_from(first_message), now, now),
        )
        await self._conn.commit()

    async def rename(self, thread_id: str, title: str) -> bool:
        cursor = await self._conn.execute(
            "UPDATE threads SET title = ? WHERE thread_id = ?", (title.strip(), thread_id)
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def delete(self, thread_id: str) -> None:
        await self._conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        await self._conn.commit()

    async def backfill(self, graph) -> int:
        """Index checkpointed threads that have no row yet. Returns how many."""
        async with self._conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints "
            "WHERE thread_id NOT IN (SELECT thread_id FROM threads)"
        ) as cursor:
            missing = [r["thread_id"] for r in await cursor.fetchall()]

        for thread_id in missing:
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            messages = (state.values or {}).get("messages", [])
            first = next((m.text for m in messages if isinstance(m, HumanMessage)), "")
            if not first:
                continue
            # The latest checkpoint's timestamp is the thread's last activity.
            try:
                updated = datetime.fromisoformat(state.created_at).timestamp()
            except (TypeError, ValueError):
                updated = time.time()
            await self._conn.execute(
                "INSERT INTO threads (thread_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (thread_id, title_from(first), updated, updated),
            )
        await self._conn.commit()
        return len(missing)
