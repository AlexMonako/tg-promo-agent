"""Optional stdio MCP server that exposes the agent's tools to external MCP clients.

Run with: `python -m tg_promo_agent.mcp_server`

This lets Claude Desktop / Cursor / any MCP client drive the same toolset manually,
without running the autonomous loop.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .agent import Agent, build_tool_schemas
from .config import load_config


async def amain() -> None:
    cfg = load_config()
    agent = Agent(cfg)
    await agent.start()

    server: Server = Server("tg-promo-agent")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        schemas = build_tool_schemas(cfg)
        return [
            Tool(
                name=s["function"]["name"],
                description=s["function"]["description"],
                inputSchema=s["function"]["parameters"],
            )
            for s in schemas
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        await agent._dispatch(name, arguments)  # noqa: SLF001 — internal use is fine here
        return [TextContent(type="text", text=json.dumps(agent.status, ensure_ascii=False))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
