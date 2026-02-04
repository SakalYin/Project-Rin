"""
Async SQLite database for persisting chat history with session support.

Each session stores messages as a JSON array, allowing users to continue
previous conversations or start fresh.
"""

from __future__ import annotations

import aiosqlite
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from src.core.config import AppConfig


class Message(TypedDict):
    """Single message in a session."""
    role: str
    content: str
    time: float


class SessionInfo(TypedDict):
    """Session metadata for display."""
    id: int
    name: str
    message_count: int
    created_at: float
    updated_at: float


class ChatDatabase:
    """Manages persistent chat history with session support."""

    def __init__(self, config: AppConfig) -> None:
        self._db_path = config.database.path
        self._history_limit = config.database.history_limit
        self._db: aiosqlite.Connection | None = None
        self._current_session_id: int | None = None

    async def initialize(self) -> None:
        """Create the database and tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                messages    TEXT    NOT NULL DEFAULT '[]',
                created_at  REAL    NOT NULL,
                updated_at  REAL    NOT NULL
            )
        """)
        await self._db.commit()

    async def create_session(self, name: str | None = None) -> int:
        """Create a new session and return its ID."""
        if self._db is None:
            raise RuntimeError("Database not initialized — call initialize() first")

        now = time.time()
        if name is None:
            name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor = await self._db.execute(
            "INSERT INTO sessions (name, messages, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, "[]", now, now),
        )
        await self._db.commit()
        self._current_session_id = cursor.lastrowid
        return self._current_session_id

    async def load_session(self, session_id: int) -> bool:
        """Load an existing session. Returns True if found."""
        if self._db is None:
            raise RuntimeError("Database not initialized — call initialize() first")

        cursor = await self._db.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row:
            self._current_session_id = session_id
            return True
        return False

    async def get_sessions(self, limit: int = 20) -> list[SessionInfo]:
        """Get list of recent sessions for selection UI."""
        if self._db is None:
            raise RuntimeError("Database not initialized — call initialize() first")

        cursor = await self._db.execute(
            """
            SELECT id, name, messages, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()

        sessions: list[SessionInfo] = []
        for row in rows:
            messages = json.loads(row[2])
            sessions.append({
                "id": row[0],
                "name": row[1],
                "message_count": len(messages),
                "created_at": row[3],
                "updated_at": row[4],
            })
        return sessions

    async def save_message(self, role: str, content: str) -> None:
        """Persist a single message to the current session."""
        if self._db is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        if self._current_session_id is None:
            raise RuntimeError("No session loaded — call create_session() or load_session() first")

        # Get current messages
        cursor = await self._db.execute(
            "SELECT messages FROM sessions WHERE id = ?",
            (self._current_session_id,),
        )
        row = await cursor.fetchone()
        messages: list[Message] = json.loads(row[0]) if row else []

        # Append new message
        messages.append({
            "role": role,
            "content": content,
            "time": time.time(),
        })

        # Update session
        await self._db.execute(
            "UPDATE sessions SET messages = ?, updated_at = ? WHERE id = ?",
            (json.dumps(messages), time.time(), self._current_session_id),
        )
        await self._db.commit()

    async def get_history(self, limit: int | None = None) -> list[dict[str, str]]:
        """Return the last *limit* messages as OpenAI-style dicts."""
        if self._db is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        if self._current_session_id is None:
            return []

        limit = limit or self._history_limit

        cursor = await self._db.execute(
            "SELECT messages FROM sessions WHERE id = ?",
            (self._current_session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return []

        messages: list[Message] = json.loads(row[0])
        # Return last `limit` messages in OpenAI format
        recent = messages[-limit:] if limit else messages
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    async def get_session_name(self) -> str | None:
        """Get the current session's name."""
        if self._db is None or self._current_session_id is None:
            return None

        cursor = await self._db.execute(
            "SELECT name FROM sessions WHERE id = ?",
            (self._current_session_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def close(self) -> None:
        """Cleanly close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
