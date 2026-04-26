"""Groq tool-calling wrapper used by the autonomy loop."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from groq import AsyncGroq, RateLimitError

log = structlog.get_logger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 1.0  # seconds; doubles on each attempt


class GroqPlanner:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncGroq(api_key=api_key) if api_key else None
        self._model = model

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def _create(self, **kwargs: Any) -> Any:
        """Call chat.completions.create with exponential-backoff retry on HTTP 429."""
        delay = _BASE_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
            except RateLimitError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                # Honour Retry-After header when present
                retry_after: float | None = None
                try:
                    header = exc.response.headers.get("retry-after")
                    if header:
                        retry_after = float(header)
                except Exception:  # noqa: BLE001
                    pass
                wait = retry_after if retry_after is not None else delay
                log.warning(
                    "groq.rate_limited",
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    retry_in=wait,
                )
                await asyncio.sleep(wait)
                delay *= 2

    async def plan(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ask the model to pick zero or more tool calls. Returns the raw tool_calls list."""
        if self._client is None:
            log.warning("groq.disabled", reason="no GROQ_API_KEY configured")
            return []
        resp = await self._create(
            model=self._model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
        )
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        out: list[dict[str, Any]] = []
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                log.warning("groq.bad_args", raw=c.function.arguments)
                continue
            out.append({"name": c.function.name, "args": args})
        if not out and msg.content:
            log.info("groq.no_tool_calls", content=msg.content[:200])
        return out

    async def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
        if self._client is None:
            return ""
        resp = await self._create(
            model=self._model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
