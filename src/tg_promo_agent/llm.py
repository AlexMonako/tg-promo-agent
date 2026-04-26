"""Groq tool-calling wrapper used by the autonomy loop."""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import structlog
from groq import AsyncGroq, RateLimitError

log = structlog.get_logger(__name__)


# Per-call retry budget on top of the SDK's built-in retries. We deliberately
# keep this conservative — Groq's free tier RPM is low, and re-hammering only
# delays the inevitable. After this many 429s in one tick we give up and let
# the next tick (≥ AGENT_TICK_SECONDS later) try again.
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_BASE_DELAY_S = 4.0


class GroqPlanner:
    """Wraps Groq with two models: a small planner (tool-call) and a larger writer."""

    def __init__(
        self,
        api_key: str,
        model: str,
        planner_model: str | None = None,
    ) -> None:
        # Larger SDK retry budget than the default (2) so transient 429 spikes
        # at the edge of TPM/RPM don't drop a whole tick.
        self._client = AsyncGroq(api_key=api_key, max_retries=4) if api_key else None
        self._writer_model = model
        self._planner_model = planner_model or model

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def _call_with_backoff(self, **kwargs: Any) -> Any:
        """Run a chat.completions.create with extra 429 backoff on top of the SDK."""
        assert self._client is not None
        last_err: Exception | None = None
        for attempt in range(_RATE_LIMIT_RETRIES):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                last_err = e
                if attempt == _RATE_LIMIT_RETRIES - 1:
                    break
                delay = _RATE_LIMIT_BASE_DELAY_S * (2**attempt) + random.uniform(0, 1.5)
                log.warning(
                    "groq.rate_limited",
                    attempt=attempt + 1,
                    sleep_s=round(delay, 1),
                    model=kwargs.get("model"),
                )
                await asyncio.sleep(delay)
        log.error("groq.rate_limited_giving_up", error=str(last_err))
        raise last_err if last_err else RuntimeError("groq rate limited")

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
        try:
            resp = await self._call_with_backoff(
                model=self._planner_model,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=1024,
            )
        except RateLimitError:
            return []
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
        try:
            resp = await self._call_with_backoff(
                model=self._writer_model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
        except RateLimitError:
            return ""
        return (resp.choices[0].message.content or "").strip()
