"""Aiogram router wiring for Telegram Business updates and owner direct commands."""

from __future__ import annotations

from typing import List

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import BusinessConnection, Message

from assistant.service import Assistant

ALLOWED_UPDATES: List[str] = [
    "message",
    "business_connection",
    "business_message",
]

STRANGER_START_TEXT = (
    "Salom! Bu shaxsiy yordamchi bot va faqat o'z egasi uchun ishlaydi. "
    "Bu yerda suhbatlashib bo'lmaydi — iltimos, egasining shaxsiy akkauntiga to'g'ridan-to'g'ri yozing."
)


def _is_owner_direct_message(message: Message, owner_id: int) -> bool:
    if getattr(message, "business_connection_id", None):
        return False
    from_user = getattr(message, "from_user", None)
    chat = getattr(message, "chat", None)
    if from_user is None or chat is None:
        return False
    return from_user.id == owner_id and chat.id == owner_id


def build_router(assistant: Assistant) -> Router:
    """Create an aiogram Router connected to the given Assistant instance."""
    router = Router(name="personal_business_assistant")
    owner_id = assistant.settings.owner_id

    @router.business_connection()
    async def handle_business_connection(connection: BusinessConnection) -> None:
        await assistant.on_business_connection(connection)

    @router.business_message()
    async def handle_business_message(message: Message, bot: Bot) -> None:
        await assistant.on_business_message(bot, message)

    @router.message(Command("start"))
    async def handle_start(message: Message) -> None:
        if getattr(message, "business_connection_id", None):
            return
        if _is_owner_direct_message(message, owner_id):
            await message.answer(assistant.start_text())
        else:
            await message.answer(STRANGER_START_TEXT)

    @router.message(Command("pause"))
    async def handle_pause(message: Message) -> None:
        if not _is_owner_direct_message(message, owner_id):
            return
        assistant.pause()
        await message.answer("Assistant paused. Send /resume to continue.")

    @router.message(Command("resume"))
    async def handle_resume(message: Message) -> None:
        if not _is_owner_direct_message(message, owner_id):
            return
        assistant.resume()
        await message.answer("Assistant resumed.")

    @router.message(Command("status"))
    async def handle_status(message: Message) -> None:
        if not _is_owner_direct_message(message, owner_id):
            return
        await message.answer(assistant.status_text())

    @router.message()
    async def handle_other_direct_messages(message: Message) -> None:
        # Silently ignore all non-command direct messages from anyone.
        return

    return router
