"""Interactive helper: log in once locally and print a StringSession.

Usage:
    python -m tg_promo_agent.scripts.login

After it finishes, copy the printed string into TELEGRAM_SESSION_STRING on Railway.
"""

from __future__ import annotations

import asyncio
import os
import sys

from telethon import TelegramClient
from telethon.sessions import StringSession


def _prompt(key: str) -> str:
    return (os.getenv(key) or input(f"{key}: ")).strip()


async def amain() -> None:
    api_id = int(_prompt("TELEGRAM_API_ID"))
    api_hash = _prompt("TELEGRAM_API_HASH")

    async with TelegramClient(StringSession(), api_id, api_hash) as client:
        await client.start()  # will prompt for phone + code + 2FA if needed
        session_string = client.session.save()
        print("\n=== TELEGRAM_SESSION_STRING (copy this into Railway) ===\n")
        print(session_string)
        print("\n=== keep this secret; anyone with it can act as you on Telegram ===")


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
