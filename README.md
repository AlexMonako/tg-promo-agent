# tg-promo-agent

Autonomous AI agent that grows a set of Telegram channels you own. Makes its own decisions each tick
(what to post, where to comment, who to DM, when to fetch analytics) using Groq's LLM as the planner
and Telegram MTProto + TGStat + optional cross-post platforms as its toolbox.

Designed to run continuously on Railway.

## How it works

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        autonomy loop (every N sec)                   │
  │                                                                      │
  │   memory (SQLite action log)  ──►  Groq planner (tool-calling)       │
  │            ▲                              │                          │
  │            │                              ▼                          │
  │   policy engine (caps/cooldowns)   pick one tool + args              │
  │            │                              │                          │
  │            ▼                              ▼                          │
  │   execute tool:  tg_post / tg_comment / tg_dm / tgstat_* / cross_*   │
  │            │                                                         │
  │            ▼                                                         │
  │   log action → update cooldowns → idle for tick                      │
  └──────────────────────────────────────────────────────────────────────┘
```

Tools available to the planner:

| Tool | What it does |
|------|--------------|
| `tg_post_own_channel`     | Generate and publish a post in one of your channels (warm-up, content) |
| `tg_comment`              | Post a thoughtful comment in a whitelisted foreign channel |
| `tg_dm_channel_owner`     | Send a personalized mutual-PR proposal via DM |
| `tgstat_find_similar`     | Find channels with overlapping audience for outreach |
| `tgstat_report`           | Pull analytics (subs growth, ER, best time) for one of your channels |
| `cross_post`              | Teaser on X / Reddit / VK that drives traffic back to TG |
| `noop`                    | Do nothing this tick (cooldowns active / nothing useful to do) |

The same tools are also exposed over **MCP** (`python -m tg_promo_agent.mcp_server`) so you can drive
them manually from Claude Desktop, Cursor, or any other MCP client.

## Setup

1. Get `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` from <https://my.telegram.org/apps>.
2. Generate a `TELEGRAM_SESSION_STRING` **once** on your own machine:
   ```
   pip install -e .
   python -m tg_promo_agent.scripts.login
   ```
   Paste the printed string into Railway as the `TELEGRAM_SESSION_STRING` env var.
3. Set `GROQ_API_KEY` (required) and `TGSTAT_TOKEN` (optional, enables analytics + similar-channel search).
4. Copy `config.example.yaml` to `config.yaml` and fill in your channels + policy.

## Run locally (dry-run)

```bash
cp .env.example .env        # fill in values
cp config.example.yaml config.yaml
export $(grep -v '^#' .env | xargs) && DRY_RUN=true python -m tg_promo_agent.main
```

`DRY_RUN=true` makes the agent decide but never actually post / DM / comment. The decisions are written
to `data/state.sqlite3` and visible at `GET /actions`.

## Deploy to Railway

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**.
3. Add a Persistent Volume and mount it at `/data` (for SQLite state).
4. Set env vars from `.env.example`. Keep `DRY_RUN=true` for the first day, then flip it off.
5. Railway will use the provided `Dockerfile` and `railway.toml`.

The agent exposes `GET /health`, `GET /config`, `GET /actions?limit=N` on `$PORT`.

## Anti-spam policy

Every action is checked against the policy in `config.yaml` *before* it fires. Hard caps enforced:

- Per-day caps for own-channel posts, comments, DMs.
- Global per-tool cooldown.
- Per-foreign-channel comment cooldown.
- DM deduplication with configurable cooldown in days.
- Foreign-channel whitelist — the agent **cannot** comment anywhere not explicitly listed.
- Forbidden phrases stripped from generated content.

## Safety caveats

Aggressive unsolicited DMs / generic spammy comments get accounts banned. This agent is built to do
the opposite: targeted, low-volume, human-sounding outreach with strict rate limits. Keep
`DRY_RUN=true` until you've reviewed a few days of decisions in `GET /actions` and you're happy with
them.

## License

MIT
