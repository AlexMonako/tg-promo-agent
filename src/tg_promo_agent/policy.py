"""Anti-spam policy: enforces hard caps and cooldowns before tools fire."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .state import StateStore

# All tools known to the agent. Kept here so snapshot() can report on each.
ALL_TOOLS: tuple[str, ...] = (
    "tg_post_own_channel",
    "tg_comment",
    "tg_dm_channel_owner",
    "tgstat_find_similar",
    "tgstat_report",
    "cross_post",
    "noop",
)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""


class PolicyEngine:
    def __init__(self, cfg: AppConfig, store: StateStore) -> None:
        self.cfg = cfg
        self.store = store

    async def snapshot(self) -> dict[str, Any]:
        """Summarize current rate-limit state for the planner."""
        now = int(time.time())
        day_ago = now - 24 * 3600
        cooldown_s = self.cfg.policy.min_minutes_between_same_tool * 60

        cooldowns_min: dict[str, int] = {}
        daily_usage: dict[str, int] = {}
        for tool in ALL_TOOLS:
            # noop is exempt from the global cooldown — always available.
            if tool == "noop":
                cooldowns_min[tool] = 0
            else:
                last = await self.store.last_action_ts(tool)
                if last and cooldown_s:
                    remaining = max(0, cooldown_s - (now - last))
                    cooldowns_min[tool] = remaining // 60
                else:
                    cooldowns_min[tool] = 0
            daily_usage[tool] = await self.store.count_actions(tool, day_ago)

        p = self.cfg.policy
        n_channels = max(len(self.cfg.own_channels), 1)
        daily_caps = {
            "tg_post_own_channel": p.max_own_channel_posts_per_day * n_channels,
            "tg_comment": p.max_comments_per_day,
            "tg_dm_channel_owner": p.max_dms_per_day,
        }
        return {
            "cooldowns_min": cooldowns_min,
            "daily_usage": daily_usage,
            "daily_caps": daily_caps,
        }

    async def check(self, tool: str, target: str | None) -> PolicyDecision:
        p = self.cfg.policy
        now = int(time.time())
        day_ago = now - 24 * 3600

        if tool == "tg_post_own_channel":
            if target is not None:
                count = await self.store.count_actions(tool, day_ago)
                if count >= p.max_own_channel_posts_per_day * max(len(self.cfg.own_channels), 1):
                    return PolicyDecision(False, f"daily post cap reached ({count})")
        elif tool == "tg_comment":
            count = await self.store.count_actions(tool, day_ago)
            if count >= p.max_comments_per_day:
                return PolicyDecision(False, f"daily comment cap reached ({count})")
            if target and target not in self.cfg.allowed_foreign_channels:
                return PolicyDecision(False, f"{target} not in allowed_foreign_channels")
            if target:
                last = await self.store.last_action_ts(tool, target)
                if last and (now - last) < p.min_minutes_between_same_foreign_channel * 60:
                    return PolicyDecision(
                        False,
                        f"cooldown on {target}: "
                        f"{(now - last) // 60}m < {p.min_minutes_between_same_foreign_channel}m",
                    )
        elif tool == "tg_dm_channel_owner":
            count = await self.store.count_actions(tool, day_ago)
            if count >= p.max_dms_per_day:
                return PolicyDecision(False, f"daily DM cap reached ({count})")
            if target:
                cooldown_s = self.cfg.similar_search.dm_cooldown_days * 24 * 3600
                if await self.store.dm_sent_within(target, cooldown_s):
                    return PolicyDecision(False, f"already DM'd {target} within cooldown")
            if (
                "tg_dm_channel_owner" in p.require_human_approval_for
                or "dm_channel_owner" in p.require_human_approval_for
            ):
                return PolicyDecision(False, "tg_dm_channel_owner requires human approval")

        # Global per-tool cooldown. noop is exempt — it's the fallback action
        # and must always be available when every other tool is blocked.
        if tool != "noop":
            last_any = await self.store.last_action_ts(tool)
            if last_any and (now - last_any) < p.min_minutes_between_same_tool * 60:
                return PolicyDecision(
                    False,
                    f"global cooldown on {tool}: "
                    f"{(now - last_any) // 60}m < {p.min_minutes_between_same_tool}m",
                )

        return PolicyDecision(True)

    def sanitize_content(self, text: str) -> str:
        cleaned = text
        for bad in self.cfg.policy.forbidden_phrases_in_generated_content:
            cleaned = cleaned.replace(bad, "")
        return cleaned.strip()
