"""Autonomy loop: Groq plans next tool call, policy checks, execute, log."""

from __future__ import annotations

import asyncio
import contextlib
import random
from typing import Any

import structlog

from .config import AppConfig
from .llm import GroqPlanner
from .policy import PolicyEngine
from .state import StateStore
from .tools import cross_post
from .tools.telegram import TelegramTools
from .tools.tgstat import TGStatTools

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a pragmatic growth agent for a set of Telegram channels owned by the user.
Your goal: grow subscribers and engagement of the user's own channels over time, while strictly
respecting platform rules and the user's anti-spam policy.

Rules you must follow:
1. Never post, comment, or DM outside the whitelists in the config. If a target is not whitelisted, don't use that tool on it.
2. Favor quality over quantity. One good, context-aware comment or post beats ten generic ones.
3. For every action, briefly explain (in the `rationale` argument) WHY it's the best next move now.
4. If no good action is available, call `noop` — this is always acceptable.
5. Never generate forbidden phrases, unsubstantiated claims, or pyramid-scheme language.
6. Cross-promotion DMs must be personal: reference the target channel's topic and propose a concrete mutual value.
7. Obey the policy caps. If you hit a cap, use `noop` and wait.

You have a persistent memory (recent action log) and may use analytics tools to decide what to do.
"""


def _enabled_cross_post_platforms(cfg: AppConfig) -> list[str]:
    out: list[str] = []
    if cfg.cross_post.twitter.enabled:
        out.append("twitter")
    if cfg.cross_post.reddit.enabled:
        out.append("reddit")
    if cfg.cross_post.vk.enabled:
        out.append("vk")
    return out


def build_tool_schemas(cfg: AppConfig) -> list[dict[str, Any]]:
    """Return only the tools that make sense given current config.

    Tools are HIDDEN (not just blocked) when they can't succeed, so the LLM
    never wastes a tick choosing them. The agent always keeps `tg_post_own_channel`
    and `noop` available as long as there is at least one postable channel.
    """
    own_usernames = [c.username for c in cfg.own_channels]
    postable = [c.username for c in cfg.own_channels if c.can_post]
    allowed_foreign = cfg.allowed_foreign_channels
    cp_platforms = _enabled_cross_post_platforms(cfg)
    have_tgstat = bool(cfg.tgstat_token)

    schemas: list[dict[str, Any]] = []

    if postable:
        schemas.append({
            "type": "function",
            "function": {
                "name": "tg_post_own_channel",
                "description": "Publish an original post to one of the user's own Telegram channels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "enum": postable},
                        "content_brief": {
                            "type": "string",
                            "description": "What the post should cover. The agent will generate the final text.",
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["channel", "content_brief", "rationale"],
                },
            },
        })

    if allowed_foreign:
        schemas.append({
            "type": "function",
            "function": {
                "name": "tg_comment",
                "description": "Post a thoughtful comment under a recent message of a whitelisted foreign channel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "enum": allowed_foreign},
                        "message_id": {"type": "integer"},
                        "comment_brief": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["channel", "message_id", "comment_brief", "rationale"],
                },
            },
        })

    if own_usernames:
        schemas.append({
            "type": "function",
            "function": {
                "name": "tg_dm_channel_owner",
                "description": "Send a personalized mutual-PR proposal to a channel owner via DM.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "owner_username": {"type": "string"},
                        "their_channel": {"type": "string"},
                        "our_channel": {"type": "string", "enum": own_usernames},
                        "message_brief": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "owner_username",
                        "their_channel",
                        "our_channel",
                        "message_brief",
                        "rationale",
                    ],
                },
            },
        })

    if have_tgstat and own_usernames:
        schemas.append({
            "type": "function",
            "function": {
                "name": "tgstat_find_similar",
                "description": "Find channels similar to one of ours for potential mutual promotion.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "our_channel": {"type": "string", "enum": own_usernames},
                        "rationale": {"type": "string"},
                    },
                    "required": ["our_channel", "rationale"],
                },
            },
        })
        schemas.append({
            "type": "function",
            "function": {
                "name": "tgstat_report",
                "description": "Fetch TGStat analytics for one of our channels and store a short report.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "our_channel": {"type": "string", "enum": own_usernames},
                        "rationale": {"type": "string"},
                    },
                    "required": ["our_channel", "rationale"],
                },
            },
        })

    if cp_platforms and own_usernames:
        schemas.append({
            "type": "function",
            "function": {
                "name": "cross_post",
                "description": "Generate and publish a cross-platform teaser (X / Reddit / VK) that drives traffic to one of our TG channels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "platform": {"type": "string", "enum": cp_platforms},
                        "our_channel": {"type": "string", "enum": own_usernames},
                        "content_brief": {"type": "string"},
                        "extra": {
                            "type": "object",
                            "description": "Platform-specific fields, e.g. {'subreddit': 'r/foo'} or {'group_id': 12345}.",
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["platform", "our_channel", "content_brief", "rationale"],
                },
            },
        })

    schemas.append({
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Do nothing this tick. Use when cooldowns are active or no good action is available.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    })

    return schemas


class Agent:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.store = StateStore(cfg.state_db_path)
        self.policy = PolicyEngine(cfg, self.store)
        self.llm = GroqPlanner(cfg.groq_api_key, cfg.groq_model)
        self.tg = TelegramTools(cfg.telegram_api_id, cfg.telegram_api_hash, cfg.telegram_session_string)
        self.tgstat = TGStatTools(cfg.tgstat_token)
        self._stop = asyncio.Event()
        self._last_status: dict[str, Any] = {"state": "starting"}

    @property
    def status(self) -> dict[str, Any]:
        return self._last_status

    async def start(self) -> None:
        await self.store.init()
        await self.tg.connect()
        self._last_status = {"state": "running", "dry_run": self.cfg.dry_run}

    async def stop(self) -> None:
        self._stop.set()
        await self.tg.disconnect()
        await self.tgstat.close()
        self._last_status = {"state": "stopped"}

    async def run_forever(self) -> None:
        await self.start()
        try:
            while not self._stop.is_set():
                try:
                    await self.tick()
                except Exception as e:  # noqa: BLE001
                    log.exception("agent.tick_error", error=str(e))
                # Jitter prevents thundering herd against APIs.
                sleep_for = self.cfg.agent_tick_seconds * random.uniform(0.8, 1.2)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=sleep_for)
        finally:
            await self.stop()

    async def tick(self) -> None:
        recent = await self.store.recent_actions(limit=20)
        memory = "\n".join(
            f"- [{a['tool']}] target={a['target']} dry_run={a['dry_run']} "
            f"ok={a['result'].get('ok')} blocked={a['result'].get('blocked') or ''}"
            for a in recent
        ) or "(no prior actions)"

        channel_lines = [
            f"- @{c.username} | topic: {c.topic} | lang: {c.language} | "
            f"goals: {','.join(c.goals)} | can_post: {c.can_post}"
            for c in self.cfg.own_channels
        ] or ["(no channels configured yet — use `noop`)"]

        snap = await self.policy.snapshot()
        cd_lines = [
            f"  - {tool}: cooldown {m}m remaining, used {snap['daily_usage'][tool]} today"
            f"{' (cap ' + str(snap['daily_caps'][tool]) + ')' if tool in snap['daily_caps'] else ''}"
            for tool, m in snap["cooldowns_min"].items()
        ]

        schemas = build_tool_schemas(self.cfg)
        # Hide tools that are currently on cooldown, over daily cap, or gated
        # behind `require_human_approval_for` so the planner can't waste a
        # tick choosing them (noop always stays).
        on_cooldown = {t for t, m in snap["cooldowns_min"].items() if m > 0}
        over_cap = {
            t for t, cap in snap["daily_caps"].items()
            if snap["daily_usage"].get(t, 0) >= cap
        }
        gated = set(self.cfg.policy.require_human_approval_for)
        # Legacy alias: allow `dm_channel_owner` to gate `tg_dm_channel_owner`.
        if "dm_channel_owner" in gated:
            gated.add("tg_dm_channel_owner")
        blocked_tools = (on_cooldown | over_cap | gated) - {"noop"}
        schemas = [
            s for s in schemas
            if s["function"]["name"] not in blocked_tools
            or s["function"]["name"] == "noop"
        ]
        available_tool_names = [s["function"]["name"] for s in schemas]

        user_prompt = (
            "Own channels:\n" + "\n".join(channel_lines) + "\n\n"
            f"Allowed foreign channels for commenting: "
            f"{self.cfg.allowed_foreign_channels or '[] (none allowed)'}\n\n"
            f"Available tools this tick: {available_tool_names}\n"
            "Tools NOT in this list are disabled (missing config/credentials) — "
            "don't try to use them.\n\n"
            "Current rate-limit state:\n"
            + "\n".join(cd_lines)
            + "\n(if a tool shows cooldown > 0m it will be blocked — pick a different tool)\n\n"
            f"Recent action log (most recent first):\n{memory}\n\n"
            f"Dry-run mode: {self.cfg.dry_run}.\n"
            "Pick ONE best next action now. Prefer producing real content "
            "(`tg_post_own_channel`) over analytics when channels have few recent posts. "
            "Use `noop` only if every useful tool is on cooldown or nothing is safe."
        )

        tool_calls = await self.llm.plan(SYSTEM_PROMPT, user_prompt, schemas)
        if not tool_calls:
            log.info("agent.tick_noop", reason="no tool calls from LLM")
            self._last_status = {"state": "idle", "reason": "no_plan"}
            return

        call = tool_calls[0]
        await asyncio.sleep(1.0)  # brief pause to avoid back-to-back 429s
        await self._dispatch(call["name"], call["args"])

    @staticmethod
    def _target_for_tool(name: str, args: dict[str, Any]) -> str | None:
        """Pick the arg that identifies the *subject* of this tool call.

        For DMs this must be `owner_username` (the recipient) — not `our_channel`
        — otherwise the per-owner DM cooldown looks up the wrong key and we can
        spam the same owner. For cross-posts the subject is the platform.
        """
        if name == "tg_dm_channel_owner":
            return args.get("owner_username")
        if name == "cross_post":
            return args.get("platform")
        return (
            args.get("channel")
            or args.get("our_channel")
            or args.get("owner_username")
            or args.get("platform")
        )

    async def _dispatch(self, name: str, args: dict[str, Any]) -> None:
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            log.warning("agent.unknown_tool", tool=name, args=args)
            return

        target = self._target_for_tool(name, args)
        decision = await self.policy.check(name, target)
        if not decision.allowed:
            log.info("agent.blocked_by_policy", tool=name, reason=decision.reason)
            await self.store.log_action(
                name, target, args, {"ok": False, "blocked": decision.reason}, self.cfg.dry_run
            )
            self._last_status = {"state": "blocked", "tool": name, "reason": decision.reason}
            return

        if self.cfg.dry_run:
            log.info("agent.dry_run", tool=name, args=args)
            await self.store.log_action(
                name, target, args, {"ok": True, "dry_run": True}, True
            )
            self._last_status = {"state": "dry_run", "tool": name}
            return

        result = await handler(args)
        await self.store.log_action(name, target, args, result, False)
        self._last_status = {"state": "acted", "tool": name, "result": result.get("ok")}

    # --- Handlers ---

    async def _do_noop(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "reason": args.get("reason", "")}

    async def _do_tg_post_own_channel(self, args: dict[str, Any]) -> dict[str, Any]:
        channel = args["channel"]
        brief = args["content_brief"]
        own = next((c for c in self.cfg.own_channels if c.username == channel), None)
        sys = (
            "You are the editor-in-chief of a Telegram channel. Write a single post, plain text, "
            "no hashtags spam, no clickbait. Match the channel's language and tone."
        )
        usr = (
            f"Channel: @{channel}\n"
            f"Topic: {own.topic if own else 'unknown'}\n"
            f"Language: {own.language if own else 'en'}\n"
            f"Audience: {own.target_audience if own else 'general'}\n"
            f"Brief: {brief}\n"
            f"Write the final post text now."
        )
        text = self.policy.sanitize_content(await self.llm.generate_text(sys, usr, max_tokens=700))
        if not text:
            return {"ok": False, "error": "empty_generation"}
        return await self.tg.post_to_channel(channel, text)

    async def _do_tg_comment(self, args: dict[str, Any]) -> dict[str, Any]:
        channel = args["channel"]
        message_id = int(args["message_id"])
        brief = args["comment_brief"]
        sys = "Write a single, respectful, value-adding comment. Max 2 sentences. No links."
        text = self.policy.sanitize_content(
            await self.llm.generate_text(sys, brief, max_tokens=200)
        )
        if not text:
            return {"ok": False, "error": "empty_generation"}
        return await self.tg.comment_in_channel(channel, message_id, text)

    async def _do_tg_dm_channel_owner(self, args: dict[str, Any]) -> dict[str, Any]:
        owner = args["owner_username"]
        their = args["their_channel"]
        ours = args["our_channel"]
        brief = args["message_brief"]
        sys = (
            "Write a polite, concise DM proposing mutual cross-promotion between two Telegram channels. "
            "2–4 sentences. Be specific about what you offer and ask. No emojis."
        )
        usr = f"Our channel: @{ours}\nTheir channel: @{their}\nContext: {brief}"
        text = self.policy.sanitize_content(
            await self.llm.generate_text(sys, usr, max_tokens=250)
        )
        if not text:
            return {"ok": False, "error": "empty_generation"}
        res = await self.tg.dm_user(owner, text)
        if res.get("ok"):
            await self.store.record_dm(owner, their, "sent")
        return res

    async def _do_tgstat_find_similar(self, args: dict[str, Any]) -> dict[str, Any]:
        our = args["our_channel"]
        ss = self.cfg.similar_search
        data = await self.tgstat.search_similar(
            our,
            min_subscribers=ss.min_subscribers,
            max_subscribers=ss.max_subscribers,
            languages=ss.languages,
        )
        return {"ok": data.get("status") == "ok", "data": data}

    async def _do_tgstat_report(self, args: dict[str, Any]) -> dict[str, Any]:
        our = args["our_channel"]
        stat = await self.tgstat.channel_stat(our)
        info = await self.tgstat.channel_info(our)
        return {"ok": stat.get("status") == "ok", "stat": stat, "info": info}

    async def _do_cross_post(self, args: dict[str, Any]) -> dict[str, Any]:
        platform = args["platform"]
        our = args["our_channel"]
        brief = args["content_brief"]
        extra = args.get("extra") or {}
        sys = (
            "Write a short promotional teaser (<=280 chars for twitter; <=300 for others). "
            "Mention the value the reader gets, not the channel owner. No hashtag stuffing."
        )
        usr = f"Target platform: {platform}\nPromote TG channel: @{our}\nBrief: {brief}"
        text = self.policy.sanitize_content(
            await self.llm.generate_text(sys, usr, max_tokens=200)
        )
        link = f"https://t.me/{our}"
        if platform == "twitter":
            return await cross_post.post_to_twitter(f"{text}\n{link}")
        if platform == "reddit":
            subreddit = extra.get("subreddit") or (
                self.cfg.cross_post.reddit.subreddits[:1] or [""]
            )[0]
            if not subreddit:
                return {"ok": False, "error": "no_subreddit"}
            return await cross_post.post_to_reddit(subreddit, text[:280], link, body=text)
        if platform == "vk":
            group_id = extra.get("group_id") or (
                self.cfg.cross_post.vk.group_ids[:1] or [0]
            )[0]
            if not group_id:
                return {"ok": False, "error": "no_vk_group"}
            return await cross_post.post_to_vk(int(group_id), f"{text}\n{link}")
        return {"ok": False, "error": f"unknown_platform:{platform}"}
