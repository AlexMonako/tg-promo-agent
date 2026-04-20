"""Telegram MTProto tools using Telethon."""

from __future__ import annotations

from typing import Any

import structlog
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
from telethon.sessions import StringSession
from telethon.tl.types import Channel, Chat, User

log = structlog.get_logger(__name__)


class TelegramTools:
    def __init__(self, api_id: int, api_hash: str, session_string: str) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session = StringSession(session_string) if session_string else StringSession()
        self._client: TelegramClient | None = None

    async def connect(self) -> None:
        if self._client is not None and self._client.is_connected():
            return
        if not self._api_id or not self._api_hash:
            log.warning("telegram.no_credentials")
            return
        self._client = TelegramClient(self._session, self._api_id, self._api_hash)
        await self._client.connect()
        if not await self._client.is_user_authorized():
            log.error("telegram.not_authorized",
                      hint="run scripts/login.py locally and set TELEGRAM_SESSION_STRING")
            await self._client.disconnect()
            self._client = None

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    def _require(self) -> TelegramClient:
        if self._client is None:
            raise RuntimeError("Telegram client is not connected / authorized")
        return self._client

    async def post_to_channel(self, channel: str, text: str) -> dict[str, Any]:
        client = self._require()
        try:
            entity = await client.get_entity(channel)
            msg = await client.send_message(entity, text, link_preview=False)
            return {"ok": True, "message_id": msg.id, "channel": channel}
        except FloodWaitError as e:
            return {"ok": False, "error": "flood_wait", "seconds": int(e.seconds)}
        except RPCError as e:
            return {"ok": False, "error": str(e)}

    async def comment_in_channel(
        self, channel: str, message_id: int, text: str
    ) -> dict[str, Any]:
        client = self._require()
        try:
            entity = await client.get_entity(channel)
            msg = await client.send_message(
                entity, text, comment_to=message_id, link_preview=False
            )
            return {"ok": True, "message_id": msg.id, "channel": channel, "reply_to": message_id}
        except FloodWaitError as e:
            return {"ok": False, "error": "flood_wait", "seconds": int(e.seconds)}
        except RPCError as e:
            return {"ok": False, "error": str(e)}

    async def dm_user(self, username: str, text: str) -> dict[str, Any]:
        client = self._require()
        try:
            entity = await client.get_entity(username)
            if not isinstance(entity, User):
                return {"ok": False, "error": f"{username} is not a user"}
            msg = await client.send_message(entity, text, link_preview=False)
            return {"ok": True, "message_id": msg.id, "to": username}
        except FloodWaitError as e:
            return {"ok": False, "error": "flood_wait", "seconds": int(e.seconds)}
        except RPCError as e:
            return {"ok": False, "error": str(e)}

    async def fetch_recent_messages(self, channel: str, limit: int = 10) -> list[dict[str, Any]]:
        client = self._require()
        try:
            entity = await client.get_entity(channel)
            out: list[dict[str, Any]] = []
            async for m in client.iter_messages(entity, limit=limit):
                out.append(
                    {
                        "id": m.id,
                        "date": m.date.isoformat() if m.date else None,
                        "text": (m.message or "")[:400],
                    }
                )
            return out
        except RPCError as e:
            log.warning("telegram.fetch_failed", channel=channel, error=str(e))
            return []

    async def channel_stats(self, channel: str) -> dict[str, Any]:
        client = self._require()
        try:
            entity = await client.get_entity(channel)
            if isinstance(entity, (Channel, Chat)):
                full = await client.get_participants(entity, limit=0)
                return {
                    "channel": channel,
                    "title": getattr(entity, "title", None),
                    "participants_count": getattr(full, "total", None),
                }
            return {"channel": channel, "error": "not a channel"}
        except RPCError as e:
            return {"channel": channel, "error": str(e)}
