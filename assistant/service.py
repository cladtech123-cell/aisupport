"""Core business message handling, state management, and preview/auto-send pipeline."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BusinessConnection, Message

from assistant.config import Settings
from assistant.gemini_client import AIError, AILimitError, GeminiClient
from assistant.language import detect_language
from assistant.prompts import SKIP_TOKEN, build_system_prompt, build_user_prompt
from assistant.store import HistoryStore, RuntimeState, SeenMessages, ensure_private_dir

logger = logging.getLogger(__name__)

MAX_TELEGRAM_MESSAGE_LENGTH = 4000
ERROR_NOTIFICATION_COOLDOWN_SECONDS = 600.0

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class Outcome(str, Enum):
    """Result of processing a single Telegram Business message."""

    previewed = "previewed"
    sent = "sent"
    duplicate = "duplicate"
    ignored_bot = "ignored_bot"
    no_connection = "no_connection"
    foreign_connection = "foreign_connection"
    connection_disabled = "connection_disabled"
    unsupported = "unsupported"
    owner_message = "owner_message"
    owner_active = "owner_active"
    paused = "paused"
    limited = "limited"
    skipped = "skipped"
    permission_denied = "permission_denied"
    error = "error"


@dataclass
class Preview:
    """Captured preview of a generated reply."""

    connection_id: str
    chat_id: int
    chat_name: str
    language: Optional[str]
    incoming: str
    reply: str
    note: Optional[str] = None


def print_preview(preview: Preview) -> None:
    """Log a human-readable preview block when a message is not sent."""
    header = "PREVIEW, not sent"
    if preview.note:
        header = f"{header} ({preview.note})"
    lang_tag = preview.language or "unknown"
    logger.info(
        "[%s] chat=%s (%s) lang=%s\n  incoming: %s\n  reply:    %s",
        header,
        preview.chat_name,
        preview.chat_id,
        lang_tag,
        preview.incoming,
        preview.reply,
    )


def describe(message: Message) -> Optional[str]:
    """Return a textual representation of a message, or None for service messages."""
    if getattr(message, "text", None):
        return str(message.text)

    caption = (getattr(message, "caption", None) or "").strip()

    if getattr(message, "voice", None) is not None:
        return f"[voice message] {caption}".strip()
    if getattr(message, "video_note", None) is not None:
        return "[video message]"
    if getattr(message, "sticker", None) is not None:
        emoji = getattr(message.sticker, "emoji", None) or ""
        return f"[sticker] {emoji}".strip()
    if getattr(message, "photo", None):
        return f"[photo] {caption}".strip()
    if getattr(message, "video", None) is not None:
        return f"[video] {caption}".strip()
    if getattr(message, "audio", None) is not None:
        return f"[audio] {caption}".strip()
    if getattr(message, "document", None) is not None:
        return f"[document] {caption}".strip()
    if getattr(message, "animation", None) is not None:
        return f"[GIF] {caption}".strip()
    if getattr(message, "location", None) is not None:
        return "[location]"
    if getattr(message, "contact", None) is not None:
        return "[contact]"
    if getattr(message, "poll", None) is not None:
        question = getattr(message.poll, "question", None) or ""
        return f"[poll] {question}".strip()

    return None


def can_reply(connection: Any) -> bool:
    """Check whether the business connection has permission to send replies."""
    rights = getattr(connection, "rights", None)
    if rights is not None and getattr(rights, "can_reply", None) is not None:
        return bool(rights.can_reply)
    return bool(getattr(connection, "can_reply", False))


def read_profile(path: Union[str, Path]) -> str:
    """Read profile.md and strip HTML comments."""
    target = Path(path)
    if not target.exists():
        return ""
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return ""
    cleaned = _HTML_COMMENT_RE.sub("", raw)
    return cleaned.strip()


def is_profile_unfilled(path: Union[str, Path]) -> bool:
    """Return True if profile.md does not exist or still only contains empty template fields."""
    content = read_profile(path)
    if not content:
        return True
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and ":" in stripped:
            _, after = stripped.split(":", 1)
            if after.strip():
                return False
            continue
        return False
    return True


def _chat_display_name(message: Message) -> str:
    chat = getattr(message, "chat", None)
    if chat is not None:
        title = getattr(chat, "title", None)
        if title:
            return str(title)
        first = getattr(chat, "first_name", None) or ""
        last = getattr(chat, "last_name", None) or ""
        full = f"{first} {last}".strip()
        if full:
            return full
        username = getattr(chat, "username", None)
        if username:
            return f"@{username}"
    user = getattr(message, "from_user", None)
    if user is not None:
        first = getattr(user, "first_name", None) or ""
        last = getattr(user, "last_name", None) or ""
        full = f"{first} {last}".strip()
        if full:
            return full
    return str(getattr(chat, "id", "unknown"))


def _sender_display_name(message: Message) -> str:
    user = getattr(message, "from_user", None)
    if user is not None:
        first = getattr(user, "first_name", None) or ""
        last = getattr(user, "last_name", None) or ""
        full = f"{first} {last}".strip()
        if full:
            return full
        username = getattr(user, "username", None)
        if username:
            return f"@{username}"
    return _chat_display_name(message)


def _split_chunks(text: str, max_length: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> List[str]:
    if len(text) <= max_length:
        return [text]
    chunks: List[str] = []
    remaining = text
    while remaining:
        chunks.append(remaining[:max_length])
        remaining = remaining[max_length:]
    return chunks


class Assistant:
    """Coordinates Telegram Business updates, conversation history, and AI reply generation."""

    def __init__(
        self,
        settings: Settings,
        generator: Any,
        *,
        clock: Callable[[], float] = time.time,
        preview_sink: Callable[[Preview], None] = print_preview,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.clock = clock
        self.preview_sink = preview_sink

        ensure_private_dir(settings.data_dir)
        self.history = HistoryStore(
            settings.data_dir / "history.json",
            limit=settings.history_limit,
        )
        self.seen = SeenMessages(
            settings.data_dir / "seen.json",
            capacity=5000,
        )
        self.state = RuntimeState(settings.data_dir / "state.json")
        self.previews_path = settings.data_dir / "previews.jsonl"

        self.connections: Dict[str, Any] = {}
        self.outcomes: Deque[Outcome] = deque(maxlen=100)
        self._takeover_until: Dict[Tuple[str, int], float] = {}
        self._chat_locks: Dict[Tuple[str, int], asyncio.Lock] = {}
        self._last_error_notify_at: float = 0.0

    def _record(self, outcome: Outcome) -> Outcome:
        self.outcomes.append(outcome)
        return outcome

    def _get_lock(self, connection_id: str, chat_id: int) -> asyncio.Lock:
        key = (connection_id, chat_id)
        lock = self._chat_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[key] = lock
        return lock

    def _is_takeover_active(self, connection_id: str, chat_id: int) -> bool:
        key = (connection_id, chat_id)
        until = self._takeover_until.get(key, 0.0)
        if until <= 0.0:
            return False
        if self.clock() < until:
            return True
        self._takeover_until.pop(key, None)
        return False

    def _check_block_reason(self, connection_id: str, chat_id: int) -> Optional[Outcome]:
        now = self.clock()
        if self.state.paused:
            return Outcome.paused
        if self.state.limit_until > now:
            return Outcome.limited
        if self._is_takeover_active(connection_id, chat_id):
            return Outcome.owner_active
        return None

    def _write_preview(self, preview: Preview) -> None:
        self.preview_sink(preview)
        ensure_private_dir(self.previews_path.parent)
        record = {
            **asdict(preview),
            "ts": self.clock(),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self.previews_path), flags, 0o600)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line)
            try:
                os.chmod(self.previews_path, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    async def _notify_owner(self, bot: Bot, text: str) -> None:
        if not self.settings.owner_id:
            return
        try:
            await bot.send_message(chat_id=self.settings.owner_id, text=text)
        except Exception as exc:
            logger.warning("Failed to send notification to owner: %s", exc)

    async def on_business_connection(self, connection: BusinessConnection) -> None:
        """Cache and log a Telegram Business connection update."""
        self.connections[connection.id] = connection
        user_id = getattr(getattr(connection, "user", None), "id", None)
        if user_id != self.settings.owner_id:
            logger.warning(
                "Ignored foreign business connection %s from user=%s (expected owner=%s)",
                connection.id,
                user_id,
                self.settings.owner_id,
            )
            return
        logger.info(
            "Business connection %s: user=%s enabled=%s can_reply=%s",
            connection.id,
            user_id,
            connection.is_enabled,
            can_reply(connection),
        )

    async def on_business_message(self, bot: Bot, message: Message) -> Outcome:
        """Process an incoming Telegram Business message in strict 12-step order."""
        # 1. Require business_connection_id
        conn_id = getattr(message, "business_connection_id", None)
        if not conn_id:
            return self._record(Outcome.no_connection)

        chat_id = message.chat.id

        # 2. Deduplicate BEFORE any await
        dup_key = f"{conn_id}:{chat_id}:{message.message_id}"
        if not self.seen.first_time(dup_key):
            return self._record(Outcome.duplicate)

        # 3. Ignore messages sent by bots or via a business bot
        if getattr(message, "sender_business_bot", None) is not None:
            return self._record(Outcome.ignored_bot)
        from_user = getattr(message, "from_user", None)
        if from_user is not None and getattr(from_user, "is_bot", False):
            return self._record(Outcome.ignored_bot)

        # 4. Resolve business connection (from cache, or fetch from Telegram)
        connection = self.connections.get(conn_id)
        if connection is None:
            try:
                connection = await bot.get_business_connection(conn_id)
                self.connections[conn_id] = connection
            except Exception as exc:
                logger.warning(
                    "Failed to look up business connection %s: %s", conn_id, exc
                )
                return self._record(Outcome.no_connection)

        conn_user_id = getattr(getattr(connection, "user", None), "id", None)
        if conn_user_id != self.settings.owner_id:
            logger.warning(
                "Business message on foreign connection %s (user=%s)",
                conn_id,
                conn_user_id,
            )
            return self._record(Outcome.foreign_connection)

        if not getattr(connection, "is_enabled", False):
            return self._record(Outcome.connection_disabled)

        # 5. Describe message content (None for service messages)
        incoming_text = describe(message)
        if incoming_text is None:
            return self._record(Outcome.unsupported)

        now = self.clock()

        # 6. Owner's own message in the business chat -> record + activate takeover timer
        if from_user is not None and from_user.id == self.settings.owner_id:
            owner_lang = detect_language(incoming_text)
            self.history.add(
                conn_id,
                chat_id,
                role="owner",
                name="You",
                text=incoming_text,
                lang=owner_lang,
                ts=now,
            )
            if self.settings.owner_takeover_minutes > 0:
                self._takeover_until[(conn_id, chat_id)] = (
                    now + self.settings.owner_takeover_minutes * 60.0
                )
            return self._record(Outcome.owner_message)

        # 7. Detect language (fall back to chat's last known language) and record incoming
        detected_lang = detect_language(incoming_text)
        lang = detected_lang or self.history.last_language(conn_id, chat_id)
        sender_name = _sender_display_name(message)
        chat_name = _chat_display_name(message)
        self.history.add(
            conn_id,
            chat_id,
            role="other",
            name=sender_name,
            text=incoming_text,
            lang=lang,
            ts=now,
        )

        # 8. Inside per-chat lock, check pause / rate-limit / owner takeover
        lock = self._get_lock(conn_id, chat_id)
        async with lock:
            block_outcome = self._check_block_reason(conn_id, chat_id)
            if block_outcome is not None:
                return self._record(block_outcome)

            # 9. Generate reply
            profile_text = read_profile(self.settings.profile_path)
            now_dt = datetime.fromtimestamp(self.clock(), tz=timezone.utc)
            system_prompt = build_system_prompt(profile_text, now=now_dt)
            chat_history = self.history.get(conn_id, chat_id)
            user_prompt = build_user_prompt(chat_history, language=lang)

            try:
                raw_reply = await self.generator.generate(system_prompt, user_prompt)
            except AILimitError as exc:
                err_now = self.clock()
                if exc.resets_at is not None and exc.resets_at > err_now:
                    limit_until = float(exc.resets_at)
                else:
                    limit_until = (
                        err_now + float(self.settings.limit_cooldown_minutes) * 60.0
                    )
                self.state.set_limit_until(limit_until)
                until_iso = datetime.fromtimestamp(
                    limit_until, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC")
                logger.warning(
                    "AI usage limit reached (paused until %s): %s",
                    until_iso,
                    exc,
                )
                if self.settings.auto_send:
                    await self._notify_owner(
                        bot,
                        f"⚠️ AI usage limit reached. Replies are paused until {until_iso} "
                        "(or send /resume to clear early).",
                    )
                return self._record(Outcome.limited)
            except AIError as exc:
                err_now = self.clock()
                logger.error("AI generation failed for chat %s: %s", chat_id, exc)
                if self.settings.auto_send and (
                    self._last_error_notify_at == 0.0
                    or (err_now - self._last_error_notify_at)
                    >= ERROR_NOTIFICATION_COOLDOWN_SECONDS
                ):
                    self._last_error_notify_at = err_now
                    await self._notify_owner(
                        bot,
                        f"⚠️ Assistant error while generating a reply: {exc}",
                    )
                return self._record(Outcome.error)

            reply = (raw_reply or "").strip()

            # 10. Empty or [SKIP]
            if not reply or reply == SKIP_TOKEN:
                return self._record(Outcome.skipped)

            # 11. Preview mode (AUTO_SEND=false)
            if not self.settings.auto_send:
                preview = Preview(
                    connection_id=conn_id,
                    chat_id=chat_id,
                    chat_name=chat_name,
                    language=lang,
                    incoming=incoming_text,
                    reply=reply,
                )
                self._write_preview(preview)
                return self._record(Outcome.previewed)

            # 12. Auto-send mode
            post_block = self._check_block_reason(conn_id, chat_id)
            if post_block is not None:
                self._write_preview(
                    Preview(
                        connection_id=conn_id,
                        chat_id=chat_id,
                        chat_name=chat_name,
                        language=lang,
                        incoming=incoming_text,
                        reply=reply,
                        note=f"not sent: {post_block.value}",
                    )
                )
                return self._record(post_block)

            # Re-fetch business connection directly from Telegram (never from cache)
            try:
                fresh_conn = await bot.get_business_connection(conn_id)
                self.connections[conn_id] = fresh_conn
            except Exception as exc:
                logger.warning(
                    "Failed to re-verify business connection %s before send: %s",
                    conn_id,
                    exc,
                )
                self._write_preview(
                    Preview(
                        connection_id=conn_id,
                        chat_id=chat_id,
                        chat_name=chat_name,
                        language=lang,
                        incoming=incoming_text,
                        reply=reply,
                        note="permission denied: connection lookup failed",
                    )
                )
                return self._record(Outcome.permission_denied)

            fresh_user_id = getattr(getattr(fresh_conn, "user", None), "id", None)
            if fresh_user_id != self.settings.owner_id:
                self._write_preview(
                    Preview(
                        connection_id=conn_id,
                        chat_id=chat_id,
                        chat_name=chat_name,
                        language=lang,
                        incoming=incoming_text,
                        reply=reply,
                        note="permission denied: foreign connection owner",
                    )
                )
                return self._record(Outcome.permission_denied)

            if not getattr(fresh_conn, "is_enabled", False):
                self._write_preview(
                    Preview(
                        connection_id=conn_id,
                        chat_id=chat_id,
                        chat_name=chat_name,
                        language=lang,
                        incoming=incoming_text,
                        reply=reply,
                        note="permission denied: connection disabled",
                    )
                )
                return self._record(Outcome.permission_denied)

            if not can_reply(fresh_conn):
                self._write_preview(
                    Preview(
                        connection_id=conn_id,
                        chat_id=chat_id,
                        chat_name=chat_name,
                        language=lang,
                        incoming=incoming_text,
                        reply=reply,
                        note="permission denied: can_reply is false",
                    )
                )
                return self._record(Outcome.permission_denied)

            chunks = _split_chunks(reply, MAX_TELEGRAM_MESSAGE_LENGTH)
            try:
                for chunk in chunks:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        business_connection_id=conn_id,
                    )
            except TelegramAPIError as exc:
                logger.error("Failed to send business reply to %s: %s", chat_id, exc)
                return self._record(Outcome.error)

            self.history.add(
                conn_id,
                chat_id,
                role="owner",
                name="You",
                text=reply,
                lang=lang,
                ts=self.clock(),
            )
            return self._record(Outcome.sent)

    def pause(self) -> None:
        """Pause assistant replies persistently."""
        self.state.set_paused(True)

    def resume(self) -> None:
        """Resume assistant replies and clear rate-limit and owner-takeover timers."""
        self.state.paused = False
        self.state.limit_until = 0.0
        self.state.save()
        self._takeover_until.clear()

    def status_text(self) -> str:
        """Build an English status summary for the owner."""
        if self.settings.auto_send:
            mode_line = "Mode: AUTO-SEND (replies are sent to chats)"
        else:
            mode_line = "Mode: PREVIEW (replies are only shown in the logs)"

        state_line = "State: PAUSED" if self.state.paused else "State: ACTIVE"

        now = self.clock()
        if self.state.limit_until > now:
            until_str = datetime.fromtimestamp(
                self.state.limit_until, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
            limit_line = f"Rate limit cooldown: active until {until_str}"
        else:
            limit_line = "Rate limit cooldown: none"

        chats_line = f"Tracked chats in history: {self.history.chat_count()}"
        model_line = f"AI model: {self.settings.gemini_model}"
        return "\n".join([mode_line, state_line, limit_line, model_line, chats_line])

    def start_text(self) -> str:
        """Build an Uzbek onboarding and status guide for the bot owner."""
        mode_uz = (
            "AUTO-SEND (javoblar chatlarga avtomatik yuboriladi)"
            if self.settings.auto_send
            else "PREVIEW (javoblar yuborilmaydi, faqat log va previews.jsonl da ko'rinadi)"
        )
        holat_uz = (
            "PAUZADA (to'xtatilgan)"
            if self.state.paused
            else "ISHLAYAPTI (faol)"
        )
        takeover_min = self.settings.owner_takeover_minutes
        takeover_note = (
            f"Chatda o'zingiz xabar yozsangiz, bot o'sha chatda {takeover_min} daqiqa jim turadi."
            if takeover_min > 0
            else "Owner takeover taymeri o'chirilgan (0 daqiqa)."
        )

        lines = [
            "Assalomu alaykum! Bu sizning shaxsiy Telegram Business assistent botingiz.",
            "Menga ulangan chatlarda kimdir sizga yozsa, `profile.md` dagi ma'lumotlar va uslubingiz asosida sizning nomingizdan javob tayyorlayman.",
            "",
            "📊 Hozirgi holat:",
            f"• Rejim: {mode_uz}",
            f"• Holat: {holat_uz}",
            "",
            "🔗 Akkauntga ulash (Telegram Premium / Business kerak):",
            "1. Telegram → Settings → Telegram Business → Chatbots bo'limiga kiring.",
            "2. Shu botning username'ini kiriting.",
            "3. Bot qaysi chatlarni ko'rishini tanlang va javob yozish (Reply to messages) ruxsatini yoqing.",
            "",
            "⚙️ Buyruqlar:",
            "• /status — hozirgi rejim, limit va chatlar sonini ko'rish",
            "• /pause — botni vaqtincha to'xtatish",
            "• /resume — botni qayta yoqish (pauza, limit va taymerlarni tozalaydi)",
            "",
            "💡 Bilib qo'ying:",
            f"• {takeover_note}",
            "• Pul, qarz, uchrashuv va va'dalar bo'yicha bot hech narsani tasdiqlamaydi va maxfiy ma'lumot bermaydi.",
            "• O'zingiz haqingizdagi ma'lumot va namuna xabarlarni `profile.md` fayliga yozib to'ldiring.",
        ]
        if not self.settings.auto_send:
            lines.append(
                "• Hozir PREVIEW rejimi yoqilgan — javoblar haqiqatan yuborilishi uchun sozlamalarda `AUTO_SEND=true` qiling."
            )
        return "\n".join(lines)


def make_ai_client(settings: Settings) -> GeminiClient:
    """Create a configured GeminiClient from Settings."""
    return GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout=settings.gemini_timeout_seconds,
        auth_label=settings.ai_auth_mode,
    )


def build_assistant(settings: Settings, **kwargs: Any) -> Assistant:
    """Create an Assistant instance with the default GeminiClient unless a generator is passed."""
    generator = kwargs.pop("generator", None)
    if generator is None:
        generator = make_ai_client(settings)
    return Assistant(settings, generator, **kwargs)
