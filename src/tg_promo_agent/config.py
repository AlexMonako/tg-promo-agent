"""Configuration loader: merges environment variables with a YAML policy file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OwnChannel(BaseModel):
    username: str
    topic: str
    language: str = "en"
    target_audience: str = ""
    goals: list[str] = Field(default_factory=list)
    discussion_group: str | None = None
    # If False, the agent cannot use tg_post_own_channel on this channel
    # (e.g. we don't own it; we only mention it via cross-posts / mutual PR).
    can_post: bool = True


class Policy(BaseModel):
    max_own_channel_posts_per_day: int = 6
    max_comments_per_day: int = 20
    max_dms_per_day: int = 10
    min_minutes_between_same_tool: int = 15
    min_minutes_between_same_foreign_channel: int = 240
    comment_blacklist_keywords: list[str] = Field(default_factory=list)
    forbidden_phrases_in_generated_content: list[str] = Field(default_factory=list)
    require_human_approval_for: list[str] = Field(default_factory=list)


class CrossPostTwitter(BaseModel):
    enabled: bool = False


class CrossPostReddit(BaseModel):
    enabled: bool = False
    subreddits: list[str] = Field(default_factory=list)


class CrossPostVK(BaseModel):
    enabled: bool = False
    group_ids: list[int] = Field(default_factory=list)


class CrossPost(BaseModel):
    twitter: CrossPostTwitter = Field(default_factory=CrossPostTwitter)
    reddit: CrossPostReddit = Field(default_factory=CrossPostReddit)
    vk: CrossPostVK = Field(default_factory=CrossPostVK)


class SimilarSearch(BaseModel):
    min_subscribers: int = 1000
    max_subscribers: int = 50000
    languages: list[str] = Field(default_factory=lambda: ["ru", "en"])
    dm_cooldown_days: int = 30


class AppConfig(BaseModel):
    own_channels: list[OwnChannel] = Field(default_factory=list)
    policy: Policy = Field(default_factory=Policy)
    allowed_foreign_channels: list[str] = Field(default_factory=list)
    cross_post: CrossPost = Field(default_factory=CrossPost)
    similar_search: SimilarSearch = Field(default_factory=SimilarSearch)

    # Populated from env
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_session_string: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    tgstat_token: str = ""
    agent_tick_seconds: int = 600
    dry_run: bool = True
    state_db_path: str = "data/state.sqlite3"
    port: int = 8080


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> AppConfig:
    config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    raw["telegram_api_id"] = int(os.getenv("TELEGRAM_API_ID") or 0)
    raw["telegram_api_hash"] = os.getenv("TELEGRAM_API_HASH") or ""
    raw["telegram_session_string"] = os.getenv("TELEGRAM_SESSION_STRING") or ""
    raw["groq_api_key"] = os.getenv("GROQ_API_KEY") or ""
    raw["groq_model"] = os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
    raw["tgstat_token"] = os.getenv("TGSTAT_TOKEN") or ""
    raw["agent_tick_seconds"] = int(os.getenv("AGENT_TICK_SECONDS") or 600)
    raw["dry_run"] = _env_bool("DRY_RUN", True)
    raw["state_db_path"] = os.getenv("STATE_DB_PATH") or "data/state.sqlite3"
    raw["port"] = int(os.getenv("PORT") or 8080)

    return AppConfig(**raw)
