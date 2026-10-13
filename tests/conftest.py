"""Shared fixtures and offline Telegram harness for testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Sequence, Union

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import GetBusinessConnection, SendMessage, TelegramMethod
from aiogram.methods.base import TelegramType
from aiogram.types import BusinessConnection, Message, Update
import pytest

from assistant.config import Settings, load_settings
from assistant.handlers import build_router
from assistant.service import Assistant, Preview

OWNER_ID = 111222333
DEFAULT_CONN_ID = "bc-main"
TEST_BOT_TOKEN = "123456789:AAH_test_token_value_12345"


class Clock:
    """Deterministic controllable clock for tests."""

    def __init__(self, initial: float = 1_760_000_000.0) -> None:
        self.current = float(initial)

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> float:
        self.current += float(seconds)
        return self.current


class FakeGenerator:
    """Controllable fake AI generator recording all calls."""

    def __init__(
        self,
        responses: Optional[Sequence[Union[str, Exception]]] = None,
        *,
        default_reply: str = "Yaxshi, keyinroq yozaman.",
    ) -> None:
        self.queue: List[Union[str, Exception]] = list(responses or [])
        self.default_reply = default_reply
        self.calls: List[Dict[str, str]] = []
        self.hook: Optional[Callable[[], Awaitable[None]]] = None

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if self.hook is not None:
            hook_fn = self.hook
            self.hook = None
            await hook_fn()
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return str(item)
        return self.default_reply


class RecordingSession(BaseSession):
    """Offline aiogram BaseSession that records requests and never touches the network."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: List[TelegramMethod[Any]] = []
        self.connections: Dict[str, Dict[str, Any]] = {}
        self._next_msg_id = 9000

    async def close(self) -> None:
        return None

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: Optional[int] = None,
    ) -> TelegramType:
        self.requests.append(method)

        if isinstance(method, GetBusinessConnection):
            conn_id = method.business_connection_id
            payload = self.connections.get(conn_id)
            if payload is None:
                raise TelegramBadRequest(
                    method=method,
                    message=f"Bad Request: business connection {conn_id} not found",
                )
            return BusinessConnection.model_validate(payload, context={"bot": bot})  # type: ignore[return-value]

        if isinstance(method, SendMessage):
            self._next_msg_id += 1
            chat_id = int(method.chat_id)
            msg_payload: Dict[str, Any] = {
                "message_id": self._next_msg_id,
                "date": 1_760_000_000,
                "chat": {
                    "id": chat_id,
                    "type": "private",
                    "first_name": "Recipient",
                },
                "from": {
                    "id": 999888777,
                    "is_bot": True,
                    "first_name": "AssistantBot",
                    "username": "assistant_bot",
                },
                "text": method.text,
            }
            if getattr(method, "business_connection_id", None):
                msg_payload["business_connection_id"] = method.business_connection_id
            return Message.model_validate(msg_payload, context={"bot": bot})  # type: ignore[return-value]

        return True  # type: ignore[return-value]

    async def stream_content(
        self,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        raise AssertionError("Network streaming is forbidden in offline tests")
        yield b""  # pragma: no cover


def _connection_payload(
    *,
    connection_id: str = DEFAULT_CONN_ID,
    user_id: int = OWNER_ID,
    is_enabled: bool = True,
    can_reply: bool = True,
) -> Dict[str, Any]:
    return {
        "id": connection_id,
        "user": {
            "id": user_id,
            "is_bot": False,
            "first_name": "Owner",
            "username": "owner_user",
        },
        "user_chat_id": user_id,
        "date": 1_760_000_000,
        "can_reply": can_reply,
        "rights": {
            "can_reply": can_reply,
        },
        "is_enabled": is_enabled,
    }


class Harness:
    """End-to-end offline test harness wiring Dispatcher, Router, Assistant, and RecordingSession."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        overrides: Optional[Dict[str, Any]] = None,
        responses: Optional[Sequence[Union[str, Exception]]] = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / "data"
        self.profile_path = tmp_path / "profile.md"
        if not self.profile_path.exists():
            self.profile_path.write_text(
                "# Men haqimda\n- Ism: Sardor\n",
                encoding="utf-8",
            )

        merged_overrides: Dict[str, Any] = {
            "TELEGRAM_BOT_TOKEN": TEST_BOT_TOKEN,
            "OWNER_ID": str(OWNER_ID),
            "AUTO_SEND": "false",
            "GEMINI_API_KEY": "AIzaSyTestKey1234567890",
            "DATA_DIR": str(self.data_dir),
            "PROFILE_PATH": str(self.profile_path),
        }
        if overrides:
            merged_overrides.update(overrides)

        self.settings: Settings = load_settings(
            tmp_path / ".env",
            overrides=merged_overrides,
        )
        self.clock = Clock()
        self.generator = FakeGenerator(responses=responses)
        self.previews: List[Preview] = []
        self.session = RecordingSession()
        self.bot = Bot(token=self.settings.bot_token, session=self.session)
        self._update_id = 1
        self._msg_id = 100

        self._build_runtime()

    def _build_runtime(self) -> None:
        self.assistant = Assistant(
            self.settings,
            self.generator,
            clock=self.clock,
            preview_sink=self.previews.append,
        )
        self.dp = Dispatcher()
        self.dp.include_router(build_router(self.assistant))

    def restart(self) -> None:
        """Simulate process restart using the same persistent data directory."""
        self._build_runtime()

    def set_connection(
        self,
        *,
        connection_id: str = DEFAULT_CONN_ID,
        user_id: int = OWNER_ID,
        is_enabled: bool = True,
        can_reply: bool = True,
    ) -> Dict[str, Any]:
        """Set the connection returned by Telegram GetBusinessConnection without sending an update."""
        payload = _connection_payload(
            connection_id=connection_id,
            user_id=user_id,
            is_enabled=is_enabled,
            can_reply=can_reply,
        )
        self.session.connections[connection_id] = payload
        return payload

    async def connect(
        self,
        *,
        connection_id: str = DEFAULT_CONN_ID,
        user_id: int = OWNER_ID,
        is_enabled: bool = True,
        can_reply: bool = True,
    ) -> None:
        """Register the connection in Fake Telegram AND feed a business_connection update."""
        payload = self.set_connection(
            connection_id=connection_id,
            user_id=user_id,
            is_enabled=is_enabled,
            can_reply=can_reply,
        )
        await self._feed({"business_connection": payload})

    async def _feed(self, body: Dict[str, Any]) -> None:
        self._update_id += 1
        raw = {"update_id": self._update_id, **body}
        update = Update.model_validate(raw, context={"bot": self.bot})
        await self.dp.feed_update(self.bot, update)

    async def business_message(
        self,
        text: Optional[str] = "Salom!",
        *,
        from_id: int = 555666,
        from_name: str = "Ali",
        chat_id: int = 555666,
        chat_name: str = "Ali",
        message_id: Optional[int] = None,
        connection_id: str = DEFAULT_CONN_ID,
        is_bot: bool = False,
        via_business_bot: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if message_id is None:
            self._msg_id += 1
            message_id = self._msg_id

        msg: Dict[str, Any] = {
            "message_id": message_id,
            "date": int(self.clock()),
            "business_connection_id": connection_id,
            "chat": {
                "id": chat_id,
                "type": "private",
                "first_name": chat_name,
            },
            "from": {
                "id": from_id,
                "is_bot": is_bot,
                "first_name": from_name,
            },
        }
        if text is not None:
            msg["text"] = text
        if via_business_bot:
            msg["sender_business_bot"] = {
                "id": 999888777,
                "is_bot": True,
                "first_name": "HelperBot",
                "username": "helper_bot",
            }
        if extra:
            msg.update(extra)

        await self._feed({"business_message": msg})

    async def direct_message(
        self,
        text: str,
        *,
        from_id: int = OWNER_ID,
        chat_id: Optional[int] = None,
    ) -> None:
        if chat_id is None:
            chat_id = from_id
        self._msg_id += 1
        msg: Dict[str, Any] = {
            "message_id": self._msg_id,
            "date": int(self.clock()),
            "chat": {
                "id": chat_id,
                "type": "private",
                "first_name": "User",
            },
            "from": {
                "id": from_id,
                "is_bot": False,
                "first_name": "User",
            },
            "text": text,
        }
        if text.startswith("/"):
            cmd_token = text.split()[0]
            msg["entities"] = [
                {
                    "offset": 0,
                    "length": len(cmd_token),
                    "type": "bot_command",
                }
            ]
        await self._feed({"message": msg})

    @property
    def sent_messages(self) -> List[SendMessage]:
        return [r for r in self.session.requests if isinstance(r, SendMessage)]

    @property
    def connection_lookups(self) -> List[GetBusinessConnection]:
        return [
            r for r in self.session.requests if isinstance(r, GetBusinessConnection)
        ]


@pytest.fixture
def make_harness(tmp_path: Path) -> Callable[..., Harness]:
    def _factory(**kwargs: Any) -> Harness:
        return Harness(tmp_path, **kwargs)

    return _factory
