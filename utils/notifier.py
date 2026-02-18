import os
from utils.logger import setup_logger

logger = setup_logger("utils.notifier")


class Notifier:
    """Trade notification dispatcher.

    Currently logs notifications. Extend with Telegram/Slack/webhook
    by implementing additional notify methods.
    """

    def __init__(self):
        self._telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    def notify(self, message: str):
        """Send a notification via all configured channels."""
        logger.info(f"[NOTIFY] {message}")

        if self._telegram_token and self._telegram_chat_id:
            self._send_telegram(message)

    def _send_telegram(self, message: str):
        try:
            import aiohttp
            import asyncio

            async def _send():
                url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
                async with aiohttp.ClientSession() as session:
                    await session.post(url, json={
                        "chat_id": self._telegram_chat_id,
                        "text": message,
                        "parse_mode": "Markdown",
                    })

            asyncio.get_event_loop().run_until_complete(_send())
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
