"""FastAPI health + status endpoint for Railway's healthcheck."""

from __future__ import annotations

from fastapi import FastAPI

from .agent import Agent


def build_app(agent: Agent) -> FastAPI:
    app = FastAPI(title="tg-promo-agent", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "status": agent.status}

    @app.get("/actions")
    async def actions(limit: int = 50) -> dict[str, object]:
        rows = await agent.store.recent_actions(limit=min(max(limit, 1), 200))
        return {"count": len(rows), "actions": rows}

    @app.get("/config")
    async def config_summary() -> dict[str, object]:
        cfg = agent.cfg
        return {
            "dry_run": cfg.dry_run,
            "own_channels": [c.username for c in cfg.own_channels],
            "allowed_foreign_channels": cfg.allowed_foreign_channels,
            "agent_tick_seconds": cfg.agent_tick_seconds,
            "groq_model": cfg.groq_model,
            "groq_planner_model": cfg.groq_planner_model,
            "tgstat_enabled": bool(cfg.tgstat_token),
        }

    return app
