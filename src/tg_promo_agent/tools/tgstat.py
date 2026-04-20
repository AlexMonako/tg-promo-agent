"""TGStat API client: search similar channels, analytics, reports.

Docs: https://api.tgstat.ru/docs
"""

from __future__ import annotations

from typing import Any

import aiohttp
import structlog

log = structlog.get_logger(__name__)

BASE_URL = "https://api.tgstat.ru"


class TGStatTools:
    def __init__(self, token: str) -> None:
        self._token = token
        self._session: aiohttp.ClientSession | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    async def _ensure(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._token:
            return {"status": "error", "error": "no_tgstat_token"}
        session = await self._ensure()
        params = {**params, "token": self._token}
        try:
            async with session.get(f"{BASE_URL}{path}", params=params) as resp:
                data = await resp.json(content_type=None)
                return data if isinstance(data, dict) else {"status": "error", "raw": data}
        except aiohttp.ClientError as e:
            log.warning("tgstat.error", path=path, error=str(e))
            return {"status": "error", "error": str(e)}

    async def channel_info(self, channel: str) -> dict[str, Any]:
        """https://api.tgstat.ru/docs/ru/channels/get.html"""
        return await self._get("/channels/get", {"channelId": channel})

    async def channel_stat(self, channel: str) -> dict[str, Any]:
        """https://api.tgstat.ru/docs/ru/channels/stat.html"""
        return await self._get("/channels/stat", {"channelId": channel})

    async def search_similar(
        self,
        channel: str,
        min_subscribers: int = 1000,
        max_subscribers: int = 50000,
        languages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find channels with overlapping audience.

        Uses /channels/search with similarity heuristics. Real TGStat exposes more
        endpoints (e.g. /channels/similar) under paid plans — this is the common-denominator
        path that works on the free tier.
        """
        params: dict[str, Any] = {
            "q": channel,
            "country": "ru" if (languages and "ru" in languages) else "",
            "limit": 20,
        }
        data = await self._get("/channels/search", params)
        # Post-filter by subscriber range
        if data.get("status") == "ok":
            items = data.get("response", {}).get("items", [])
            filtered = [
                c
                for c in items
                if min_subscribers <= int(c.get("participants_count") or 0) <= max_subscribers
            ]
            data["response"]["items"] = filtered
        return data

    async def channel_posts(self, channel: str, limit: int = 20) -> dict[str, Any]:
        """https://api.tgstat.ru/docs/ru/channels/posts.html"""
        return await self._get("/channels/posts", {"channelId": channel, "limit": limit})
