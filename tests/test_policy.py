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


async def test_snapshot_reports_usage_and_caps(engine):
    eng, store = engine
    snap = await eng.snapshot()
    assert snap["daily_usage"]["tg_post_own_channel"] == 0
    assert snap["daily_caps"]["tg_post_own_channel"] == 2  # per-channel cap (2) × 1 channel
    assert snap["cooldowns_min"]["tg_post_own_channel"] == 0

    await store.log_action("tg_post_own_channel", "mine", {}, {"ok": True}, False)
    snap2 = await eng.snapshot()
    assert snap2["daily_usage"]["tg_post_own_channel"] == 1


def test_build_tool_schemas_hides_disabled_tools():
    from tg_promo_agent.agent import build_tool_schemas
    cfg = AppConfig(
        own_channels=[OwnChannel(username="mine", topic="x")],
        allowed_foreign_channels=[],       # tg_comment disabled
        tgstat_token="",                   # tgstat_* disabled
    )
    names = [s["function"]["name"] for s in build_tool_schemas(cfg)]
    assert "tg_post_own_channel" in names
    assert "noop" in names
    assert "tg_comment" not in names
    assert "tgstat_find_similar" not in names
    assert "tgstat_report" not in names
    assert "cross_post" not in names
    # But dm_channel_owner is kept because only own_channels is required
    assert "tg_dm_channel_owner" in names


def test_build_tool_schemas_hides_post_when_no_postable():
    from tg_promo_agent.agent import build_tool_schemas
    cfg = AppConfig(
        own_channels=[OwnChannel(username="readonly", topic="x", can_post=False)],
    )
    names = [s["function"]["name"] for s in build_tool_schemas(cfg)]
    assert "tg_post_own_channel" not in names
    # Agent can still dm and noop
    assert "tg_dm_channel_owner" in names
    assert "noop" in names


def test_target_for_tool_dm_uses_owner_username():
    from tg_promo_agent.agent import Agent
    args = {
        "owner_username": "alice",
        "their_channel": "alice_channel",
        "our_channel": "ours",
        "message_brief": "hi",
        "rationale": "r",
    }
    # DM target must be the recipient, not our own channel.
    assert Agent._target_for_tool("tg_dm_channel_owner", args) == "alice"


def test_target_for_tool_cross_post_uses_platform():
    from tg_promo_agent.agent import Agent
    args = {"platform": "reddit", "our_channel": "ours", "content_brief": "x", "rationale": "r"}
    assert Agent._target_for_tool("cross_post", args) == "reddit"


def test_target_for_tool_post_uses_channel():
    from tg_promo_agent.agent import Agent
    args = {"channel": "mine", "content_brief": "x", "rationale": "r"}
    assert Agent._target_for_tool("tg_post_own_channel", args) == "mine"


# --- message_id hallucination guard ----------------------------------------


def test_validate_comment_msg_id_no_cache_passes():
    from tg_promo_agent.agent import Agent
    # No cache yet for this channel — don't reject; let TG report the real
    # error if the id turns out to be invalid.
    assert Agent._validate_comment_message_id(None, 12345) is None


def test_validate_comment_msg_id_rejects_when_cache_empty():
    from tg_promo_agent.agent import Agent
    # We tried to fetch and got nothing → channel inaccessible. Reject so we
    # don't waste a cooldown slot trying to comment somewhere we can't read.
    res = Agent._validate_comment_message_id((123.0, []), 12345)
    assert res is not None
    assert res["error"] == "no_recent_posts_known"
    assert res["rejected_id"] == 12345
    assert res["ok"] is False


def test_validate_comment_msg_id_rejects_hallucination():
    from tg_promo_agent.agent import Agent
    posts = [{"id": 4421, "text": "..."}, {"id": 4420, "text": "..."}]
    res = Agent._validate_comment_message_id((123.0, posts), 12345)
    assert res is not None
    assert res["error"] == "hallucinated_message_id"
    assert res["rejected_id"] == 12345
    assert res["valid_recent_ids"] == [4420, 4421]


def test_validate_comment_msg_id_accepts_known_id():
    from tg_promo_agent.agent import Agent
    posts = [{"id": 4421, "text": "..."}, {"id": 4420, "text": "..."}]
    assert Agent._validate_comment_message_id((123.0, posts), 4420) is None


def test_format_foreign_block_renders_real_ids():
    from tg_promo_agent.agent import Agent

    cfg = AppConfig(allowed_foreign_channels=["foo"])
    agent = Agent.__new__(Agent)  # bypass full __init__; we only need the method
    agent.cfg = cfg
    block = agent._format_foreign_block(
        {"foo": [{"id": 42, "text": "VR news drop\nApple Vision Pro 2"}]}
    )
    assert "@foo" in block
    assert "[message_id=42]" in block
    assert "VR news drop Apple Vision Pro 2" in block
    # Newlines in body are flattened so the prompt stays one-line-per-msg.
    assert "VR news drop\nApple Vision Pro" not in block


def test_format_foreign_block_marks_dead_channel():
    from tg_promo_agent.agent import Agent

    agent = Agent.__new__(Agent)
    agent.cfg = AppConfig(allowed_foreign_channels=["dead"])
    block = agent._format_foreign_block({"dead": []})
    assert "@dead" in block
    assert "inaccessible" in block.lower() or "skip" in block.lower()


async def test_refresh_foreign_messages_uses_cache(monkeypatch):
    """Second call within TTL must NOT hit Telegram again."""
    from tg_promo_agent.agent import Agent

    cfg = AppConfig(allowed_foreign_channels=["alpha", "beta"])
    agent = Agent.__new__(Agent)
    agent.cfg = cfg
    agent._foreign_msg_cache = {}

    calls: list[str] = []

    class FakeTG:
        async def fetch_recent_messages(self, channel: str, limit: int):
            calls.append(channel)
            return [{"id": 1, "text": f"hi from {channel}"}]

    agent.tg = FakeTG()  # type: ignore[assignment]

    first = await agent._refresh_foreign_messages()
    assert sorted(first.keys()) == ["alpha", "beta"]
    assert calls == ["alpha", "beta"]

    # Second call within TTL — should be served entirely from cache.
    second = await agent._refresh_foreign_messages()
    assert second == first
    assert calls == ["alpha", "beta"]  # unchanged — no new TG calls


async def test_refresh_foreign_messages_soft_fails_on_runtime_error():
    """A disconnected TG client must not crash the tick — failures cache empty."""
    from tg_promo_agent.agent import Agent

    cfg = AppConfig(allowed_foreign_channels=["alpha"])
    agent = Agent.__new__(Agent)
    agent.cfg = cfg
    agent._foreign_msg_cache = {}

    class BrokenTG:
        async def fetch_recent_messages(self, channel: str, limit: int):
            raise RuntimeError("not connected")

    agent.tg = BrokenTG()  # type: ignore[assignment]
    out = await agent._refresh_foreign_messages()
    assert out == {"alpha": []}
    # Empty result is still cached so we don't retry every tick.
    assert "alpha" in agent._foreign_msg_cache


# --- AuthKeyDuplicatedError graceful handling -----------------------------


async def test_telegram_connect_handles_authkey_duplicated():
    """connect() must not raise on AuthKeyDuplicatedError — sets a flag instead.

    Otherwise the exception propagates up and crashes the whole uvicorn
    process, taking /health down with it (which is exactly what hid the
    problem from the operator the first time we hit it).
    """
    from telethon.errors import AuthKeyDuplicatedError

    from tg_promo_agent.tools.telegram import TelegramTools

    tools = TelegramTools(api_id=12345, api_hash="hash", session_string="")

    # Replace the TelegramClient ctor with a fake whose .connect() raises
    # the exception we care about. We patch on the module the import lives
    # on so `TelegramClient(...)` inside connect() returns our fake.
    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def connect(self):
            raise AuthKeyDuplicatedError(request=None)

        async def disconnect(self):
            pass

    import tg_promo_agent.tools.telegram as tg_mod

    real = tg_mod.TelegramClient
    tg_mod.TelegramClient = FakeClient  # type: ignore[assignment]
    try:
        await tools.connect()  # must NOT raise
    finally:
        tg_mod.TelegramClient = real  # type: ignore[assignment]

    assert tools.is_session_revoked is True
    # _client must remain None so callers gating on it short-circuit.
    assert tools._client is None


async def test_agent_tick_short_circuits_on_revoked_session():
    """Tick must not call LLM / fetch / log when session is revoked."""
    from tg_promo_agent.agent import Agent

    cfg = AppConfig(allowed_foreign_channels=["alpha"])
    agent = Agent.__new__(Agent)
    agent.cfg = cfg
    agent._last_status = {}

    class RevokedTG:
        is_session_revoked = True

        async def fetch_recent_messages(self, channel: str, limit: int):
            raise AssertionError("must not be called when session revoked")

    class ExplodingLLM:
        async def plan(self, *a, **kw):
            raise AssertionError("must not be called when session revoked")

    agent.tg = RevokedTG()  # type: ignore[assignment]
    agent.llm = ExplodingLLM()  # type: ignore[assignment]
    # No store / policy / cache assigned — tick must early-return before
    # touching them.
    await agent.tick()
    assert agent._last_status["state"] == "tg_session_revoked"
