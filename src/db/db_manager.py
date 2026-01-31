"""
Async SQLite database for persisting chat history.
"""

from __future__ import annotations

import aiosqlite
import json
import time
from pathlib import Path

from src.core.config import AppConfig


class ChatDatabase:
    """Manages persistent chat history in SQLite."""

    def __init__(self, config: AppConfig) -> None:
        self._db_path = config.database.path
        self._history_limit = config.database.history_limit
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the database and tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                timestamp   REAL    NOT NULL
            )
        """)
        await self._db.commit()

    async def save_message(self, role: str, content: str) -> None:
        """Persist a single message (role = 'user' | 'assistant')."""
        if self._db is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        await self._db.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, time.time()),
        )
        await self._db.commit()

    async def get_history(self, limit: int | None = None) -> list[dict[str, str]]:
        """Return the last *limit* messages as OpenAI-style dicts."""
        if self._db is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        limit = limit or self._history_limit
        cursor = await self._db.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        # Rows come newest-first; reverse so oldest is first.
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    async def close(self) -> None:
        """Cleanly close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
