from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tg_promo_agent.config import AppConfig, OwnChannel, Policy
from tg_promo_agent.policy import PolicyEngine
from tg_promo_agent.state import StateStore


@pytest.fixture()
async def engine(tmp_path: Path):
    cfg = AppConfig(
        own_channels=[OwnChannel(username="mine", topic="x")],
        allowed_foreign_channels=["@foo"],
        policy=Policy(
            max_own_channel_posts_per_day=2,
            max_comments_per_day=2,
            max_dms_per_day=2,
            min_minutes_between_same_tool=0,
            min_minutes_between_same_foreign_channel=0,
            forbidden_phrases_in_generated_content=["bad phrase"],
        ),
    )
    store = StateStore(str(tmp_path / "s.sqlite"))
    await store.init()
    return PolicyEngine(cfg, store), store


async def test_sanitize_removes_forbidden():
    cfg = AppConfig(
        policy=Policy(forbidden_phrases_in_generated_content=["click here now"])
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = StateStore(f"{tmp}/s.sqlite")
        await store.init()
        eng = PolicyEngine(cfg, store)
        assert eng.sanitize_content("hello click here now world") == "hello  world"


async def test_foreign_channel_whitelist(engine):
    eng, _ = engine
    res = await eng.check("tg_comment", "@not_allowed")
    assert not res.allowed
    assert "not in allowed_foreign_channels" in res.reason


async def test_comment_cooldown(engine):
    eng, store = engine
    # No cap tripped yet
    res = await eng.check("tg_comment", "@foo")
    assert res.allowed
    # Log action and assert we still pass (cooldowns are 0 in fixture)
    await store.log_action("tg_comment", "@foo", {}, {"ok": True}, False)
    res2 = await eng.check("tg_comment", "@foo")
    assert res2.allowed
