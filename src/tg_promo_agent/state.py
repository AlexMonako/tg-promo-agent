"""SQLite-backed state: cooldowns, daily counters, decision log."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    tool TEXT NOT NULL,
    target TEXT,
    payload TEXT,
    result TEXT,
    dry_run INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_actions_ts ON actions(ts);
CREATE INDEX IF NOT EXISTS idx_actions_tool_ts ON actions(tool, ts);
CREATE INDEX IF NOT EXISTS idx_actions_target_ts ON actions(target, ts);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dms (
    owner_username TEXT PRIMARY KEY,
    channel_username TEXT,
    ts INTEGER NOT NULL,
    result TEXT
);
"""


class StateStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with self._conn() as db:
            await db.executescript(SCHEMA)
            await db.commit()

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        db = await aiosqlite.connect(self.path)
        try:
            db.row_factory = aiosqlite.Row
            yield db
        finally:
            await db.close()

    async def log_action(
        self,
        tool: str,
        target: str | None,
        payload: dict[str, Any],
        result: dict[str, Any],
        dry_run: bool,
    ) -> None:
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO actions (ts, tool, target, payload, result, dry_run) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(time.time()),
                    tool,
                    target,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    1 if dry_run else 0,
                ),
            )
            await db.commit()

    async def count_actions(self, tool: str, since_ts: int) -> int:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT COUNT(*) AS n FROM actions WHERE tool = ? AND ts >= ?",
                (tool, since_ts),
            )
            row = await cur.fetchone()
            return int(row["n"]) if row else 0

    async def last_action_ts(self, tool: str, target: str | None = None) -> int | None:
        async with self._conn() as db:
            if target is None:
                cur = await db.execute(
                    "SELECT ts FROM actions WHERE tool = ? ORDER BY ts DESC LIMIT 1",
                    (tool,),
                )
            else:
                cur = await db.execute(
                    "SELECT ts FROM actions WHERE tool = ? AND target = ? "
                    "ORDER BY ts DESC LIMIT 1",
                    (tool, target),
                )
            row = await cur.fetchone()
            return int(row["ts"]) if row else None

    async def record_dm(self, owner_username: str, channel_username: str, result: str) -> None:
        async with self._conn() as db:
            await db.execute(
                "INSERT OR REPLACE INTO dms (owner_username, channel_username, ts, result) "
                "VALUES (?, ?, ?, ?)",
                (owner_username, channel_username, int(time.time()), result),
            )
            await db.commit()

    async def dm_sent_within(self, owner_username: str, seconds: int) -> bool:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT ts FROM dms WHERE owner_username = ?",
                (owner_username,),
            )
            row = await cur.fetchone()
            if not row:
                return False
            return (int(time.time()) - int(row["ts"])) < seconds

    async def recent_actions(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT ts, tool, target, payload, result, dry_run "
                "FROM actions ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "ts": int(r["ts"]),
                    "tool": r["tool"],
                    "target": r["target"],
                    "payload": json.loads(r["payload"] or "{}"),
                    "result": json.loads(r["result"] or "{}"),
                    "dry_run": bool(r["dry_run"]),
                }
            )
        return out
