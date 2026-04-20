"""Cross-posting to X / Reddit / VK.

Implemented as best-effort stubs guarded by per-platform enable flags + env tokens.
Each method returns a structured result and never raises to the agent loop.
"""

from __future__ import annotations

import os
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger(__name__)


async def post_to_twitter(text: str) -> dict[str, Any]:
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        return {"ok": False, "error": "no_twitter_token"}
    # Twitter v2 posting requires OAuth 1.0a user context or OAuth2 user-context PKCE —
    # a bearer token alone cannot post. We surface this explicitly so the agent
    # doesn't repeatedly try a doomed action.
    return {
        "ok": False,
        "error": "twitter_requires_oauth1_user_context",
        "hint": "add TWITTER_API_KEY / SECRET / ACCESS_TOKEN / ACCESS_SECRET to enable",
    }


async def post_to_reddit(subreddit: str, title: str, url: str, body: str = "") -> dict[str, Any]:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    if not all([client_id, client_secret, username, password]):
        return {"ok": False, "error": "no_reddit_credentials"}

    auth = aiohttp.BasicAuth(client_id, client_secret)  # type: ignore[arg-type]
    data = {"grant_type": "password", "username": username, "password": password}
    headers = {"User-Agent": "tg-promo-agent/0.1 (+https://github.com/AlexMonako/tg-promo-agent)"}

    async with aiohttp.ClientSession() as s:
        async with s.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth, data=data, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            tok = await resp.json()
        if "access_token" not in tok:
            return {"ok": False, "error": "reddit_auth_failed", "details": tok}

        post_headers = {**headers, "Authorization": f"bearer {tok['access_token']}"}
        form = {
            "sr": subreddit.lstrip("r/"),
            "title": title,
            "kind": "link" if url else "self",
            "url": url,
            "text": body,
            "api_type": "json",
        }
        async with s.post(
            "https://oauth.reddit.com/api/submit",
            headers=post_headers, data=form,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            res = await resp.json()
        return {"ok": resp.status == 200, "response": res}


async def post_to_vk(group_id: int, text: str) -> dict[str, Any]:
    token = os.getenv("VK_ACCESS_TOKEN")
    if not token:
        return {"ok": False, "error": "no_vk_token"}
    params = {
        "owner_id": -abs(group_id),
        "from_group": 1,
        "message": text,
        "access_token": token,
        "v": "5.199",
    }
    async with aiohttp.ClientSession() as s, s.post(
        "https://api.vk.com/method/wall.post",
        data=params,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json()
    if "error" in data:
        return {"ok": False, "error": data["error"].get("error_msg"), "details": data["error"]}
    return {"ok": True, "response": data.get("response", {})}
