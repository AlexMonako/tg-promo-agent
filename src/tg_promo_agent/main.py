"""Entrypoint: boots the FastAPI health server and the agent's autonomy loop concurrently."""

from __future__ import annotations

import asyncio
import logging

import structlog
import uvicorn

from .agent import Agent
from .config import load_config
from .health import build_app


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


async def _serve(agent: Agent) -> None:
    app = build_app(agent)
    cfg = uvicorn.Config(
        app, host="0.0.0.0", port=agent.cfg.port, log_level="info", access_log=False
    )
    server = uvicorn.Server(cfg)
    await server.serve()


async def amain() -> None:
    _configure_logging()
    cfg = load_config()
    log = structlog.get_logger("main")
    log.info(
        "boot",
        dry_run=cfg.dry_run,
        own_channels=[c.username for c in cfg.own_channels],
        tick_s=cfg.agent_tick_seconds,
    )
    agent = Agent(cfg)
    tasks = [asyncio.create_task(agent.run_forever()), asyncio.create_task(_serve(agent))]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
        for t in done:
            t.result()
    finally:
        await agent.stop()


def run() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    run()
