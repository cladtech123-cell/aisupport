"""Tests for Telegram Business updates, owner controls, rate limits, and auto-send pipeline."""

from __future__ import annotations

import asyncio
import json

from assistant.gemini_client import AIAuthError, AILimitError, AITimeout
from assistant.prompts import SKIP_TOKEN
from assistant.service import Outcome
from tests.conftest import DEFAULT_CONN_ID, OWNER_ID


async def test_preview_mode_records_preview_without_sending(make_harness) -> None:
    h = make_harness(responses=["Va alaykum assalom! Ishlar yaxshi."])
    await h.connect()

    await h.business_message("Assalomu alaykum, qalesan?", chat_id=701, from_id=701)

    assert list(h.assistant.outcomes) == [Outcome.previewed]
    assert len(h.sent_messages) == 0
    assert len(h.previews) == 1
    assert h.previews[0].reply == "Va alaykum assalom! Ishlar yaxshi."
    assert h.previews[0].language == "uz"

    # Unsent preview reply must NOT be stored in chat history
    history_items = h.assistant.history.get(DEFAULT_CONN_ID, 701)
    assert len(history_items) == 1
    assert history_items[0]["role"] == "other"

    # previews.jsonl must be written
    previews_file = h.data_dir / "previews.jsonl"
    assert previews_file.exists()
    lines = previews_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["reply"] == "Va alaykum assalom! Ishlar yaxshi."


async def test_known_connection_makes_zero_telegram_requests_in_preview(make_harness) -> None:
    h = make_harness(responses=["Salom!"])
    await h.connect()
    h.session.requests.clear()

    await h.business_message("Salom, bormisan?")
    assert list(h.assistant.outcomes) == [Outcome.previewed]
    assert len(h.session.requests) == 0


async def test_unknown_connection_lookup_and_cache_and_missing_error(make_harness) -> None:
    h = make_harness(responses=["Birinchi", "Ikkinchi"])
    # Set connection in Telegram without feeding a business_connection update
    h.set_connection(connection_id="bc-lazy")

    await h.business_message("Salom 1", connection_id="bc-lazy", message_id=1)
    await h.business_message("Salom 2", connection_id="bc-lazy", message_id=2)

    assert list(h.assistant.outcomes) == [Outcome.previewed, Outcome.previewed]
    # Only 1 lookup because the second message hits the connection cache
    assert len(h.connection_lookups) == 1

    # Lookup failure for nonexistent connection -> no_connection
    await h.business_message("Salom 3", connection_id="bc-missing", message_id=3)
    assert h.assistant.outcomes[-1] == Outcome.no_connection


async def test_foreign_and_disabled_connections_do_not_call_ai(make_harness) -> None:
    h = make_harness()
    await h.connect(connection_id="bc-foreign", user_id=999999999)
    await h.connect(connection_id="bc-disabled", user_id=OWNER_ID, is_enabled=False)

    await h.business_message("Salom", connection_id="bc-foreign", message_id=10)
    await h.business_message("Salom", connection_id="bc-disabled", message_id=11)

    assert list(h.assistant.outcomes) == [
        Outcome.foreign_connection,
        Outcome.connection_disabled,
    ]
    assert len(h.generator.calls) == 0


async def test_any_chat_receives_reply_without_allowlist(make_harness) -> None:
    h = make_harness(responses=["Javob 1", "Javob 2", "Javob 3"])
    await h.connect()

    for idx, chat_id in enumerate([1001, 2002, 3003], start=1):
        await h.business_message(
            "Salom!",
            chat_id=chat_id,
            from_id=chat_id,
            message_id=idx,
        )

    assert list(h.assistant.outcomes) == [
        Outcome.previewed,
        Outcome.previewed,
        Outcome.previewed,
    ]


async def test_owner_message_activates_takeover_and_expires(make_harness) -> None:
    h = make_harness(
        overrides={"OWNER_TAKEOVER_MINUTES": "15"},
        responses=["Vaqt o'tgach javob"],
    )
    await h.connect()

    # Owner writes in chat 800
    await h.business_message(
        "Hozir o'zim yozaman",
        from_id=OWNER_ID,
        chat_id=800,
        message_id=1,
    )
    assert h.assistant.outcomes[-1] == Outcome.owner_message
    assert len(h.generator.calls) == 0

    # Contact replies while takeover is active (5 min later)
    h.clock.advance(5 * 60)
    await h.business_message(
        "Xo'p, kutaman",
        from_id=800,
        chat_id=800,
        message_id=2,
    )
    assert h.assistant.outcomes[-1] == Outcome.owner_active
    assert len(h.generator.calls) == 0

    # Advance past 15 minutes total -> assistant replies again
    h.clock.advance(11 * 60)
    await h.business_message(
        "Bo'ldimi?",
        from_id=800,
        chat_id=800,
        message_id=3,
    )
    assert h.assistant.outcomes[-1] == Outcome.previewed
    assert len(h.generator.calls) == 1
    # Verify owner's earlier message is in the prompt as [You]
    assert "[You]: Hozir o'zim yozaman" in h.generator.calls[0]["user_prompt"]


async def test_bot_and_via_business_bot_messages_ignored(make_harness) -> None:
    h = make_harness()
    await h.connect()

    await h.business_message("Bot text", is_bot=True, message_id=1)
    await h.business_message("Via business bot", via_business_bot=True, message_id=2)

    assert list(h.assistant.outcomes) == [Outcome.ignored_bot, Outcome.ignored_bot]
    assert len(h.generator.calls) == 0


async def test_deduplication_parallel_sequential_and_across_restart(make_harness) -> None:
    h = make_harness(responses=["Yagona javob"])
    await h.connect()

    await asyncio.gather(
        h.business_message("Salom!", chat_id=500, message_id=42),
        h.business_message("Salom!", chat_id=500, message_id=42),
    )
    assert sorted(o.value for o in h.assistant.outcomes) == ["duplicate", "previewed"]
    assert len(h.generator.calls) == 1

    # Restart assistant and send the same message_id again
    h.restart()
    await h.connect()
    await h.business_message("Salom!", chat_id=500, message_id=42)
    assert h.assistant.outcomes[-1] == Outcome.duplicate
    assert len(h.generator.calls) == 1


async def test_service_message_unsupported_and_media_described(make_harness) -> None:
    h = make_harness(responses=["Stiker javobi", "Ovoz javobi"])
    await h.connect()

    # Service message with no text or supported media
    await h.business_message(None, message_id=1)
    assert h.assistant.outcomes[-1] == Outcome.unsupported

    # Sticker
    await h.business_message(
        None,
        message_id=2,
        extra={
            "sticker": {
                "file_id": "stk1",
                "file_unique_id": "ustk1",
                "type": "regular",
                "width": 512,
                "height": 512,
                "is_animated": False,
                "is_video": False,
                "emoji": "😀",
            }
        },
    )
    assert h.assistant.outcomes[-1] == Outcome.previewed
    assert "[sticker] 😀" in h.generator.calls[-1]["user_prompt"]

    # Voice message
    await h.business_message(
        None,
        message_id=3,
        extra={
            "voice": {
                "file_id": "vc1",
                "file_unique_id": "uvc1",
                "duration": 5,
            }
        },
    )
    assert h.assistant.outcomes[-1] == Outcome.previewed
    assert "[voice message]" in h.generator.calls[-1]["user_prompt"]


async def test_history_isolated_by_connection_and_chat(make_harness) -> None:
    h = make_harness(responses=["R1", "R2", "R3"])
    await h.connect(connection_id="conn-A")
    await h.connect(connection_id="conn-B")

    await h.business_message(
        "Maxfiy xabar A1",
        connection_id="conn-A",
        chat_id=10,
        from_id=10,
        message_id=1,
    )
    await h.business_message(
        "Maxfiy xabar A2",
        connection_id="conn-A",
        chat_id=20,
        from_id=20,
        message_id=2,
    )
    await h.business_message(
        "Boshqa connection xabari",
        connection_id="conn-B",
        chat_id=10,
        from_id=10,
        message_id=3,
    )

    prompt_a2 = h.generator.calls[1]["user_prompt"]
    prompt_b1 = h.generator.calls[2]["user_prompt"]

    assert "Maxfiy xabar A1" not in prompt_a2
    assert "Maxfiy xabar A1" not in prompt_b1


async def test_language_hints_and_emoji_fallback(make_harness) -> None:
    h = make_harness(responses=["R1", "R2", "R3", "R4"])
    await h.connect()

    await h.business_message("Привет, как дела?", chat_id=90, from_id=90, message_id=1)
    assert "Their latest message is in Russian." in h.generator.calls[-1]["user_prompt"]

    # Emoji-only follow-up in the same chat inherits Russian
    await h.business_message("👍", chat_id=90, from_id=90, message_id=2)
    assert "Their latest message is in Russian." in h.generator.calls[-1]["user_prompt"]

    await h.business_message("Hi, are you free tonight?", chat_id=91, from_id=91, message_id=3)
    assert "Their latest message is in English." in h.generator.calls[-1]["user_prompt"]

    await h.business_message("Salom, qalesan?", chat_id=92, from_id=92, message_id=4)
    assert "Their latest message is in Uzbek." in h.generator.calls[-1]["user_prompt"]


async def test_skip_token_is_not_previewed_or_sent(make_harness) -> None:
    h = make_harness(
        overrides={"AUTO_SEND": "true"},
        responses=[SKIP_TOKEN, "   "],
    )
    await h.connect()

    await h.business_message("Ok", message_id=1)
    await h.business_message("👍", message_id=2)

    assert list(h.assistant.outcomes) == [Outcome.skipped, Outcome.skipped]
    assert len(h.previews) == 0
    assert len(h.sent_messages) == 0


async def test_auto_send_verifies_connection_splits_long_reply_and_saves_history(
    make_harness,
) -> None:
    long_reply = "A" * 4500
    h = make_harness(
        overrides={"AUTO_SEND": "true"},
        responses=[long_reply],
    )
    await h.connect()
    h.session.requests.clear()

    await h.business_message("Uzun javob yoz", chat_id=333, from_id=333, message_id=1)

    assert list(h.assistant.outcomes) == [Outcome.sent]
    # First request must be GetBusinessConnection, followed by two SendMessage chunks (4000 + 500)
    assert len(h.connection_lookups) == 1
    assert len(h.sent_messages) == 2
    assert len(h.sent_messages[0].text) == 4000
    assert len(h.sent_messages[1].text) == 500
    assert h.sent_messages[0].business_connection_id == DEFAULT_CONN_ID

    items = h.assistant.history.get(DEFAULT_CONN_ID, 333)
    assert len(items) == 2
    assert items[1]["role"] == "owner"
    assert items[1]["text"] == long_reply


async def test_auto_send_aborts_if_can_reply_revoked_or_disabled(make_harness) -> None:
    h = make_harness(
        overrides={"AUTO_SEND": "true"},
        responses=["Javob 1", "Javob 2"],
    )
    await h.connect()

    # Revoke can_reply on Telegram side without an update
    h.set_connection(can_reply=False)
    await h.business_message("Salom 1", message_id=1)
    assert h.assistant.outcomes[-1] == Outcome.permission_denied
    assert len(h.sent_messages) == 0

    # Disable connection on Telegram side
    h.set_connection(can_reply=True, is_enabled=False)
    await h.business_message("Salom 2", message_id=2)
    assert h.assistant.outcomes[-1] == Outcome.permission_denied
    assert len(h.sent_messages) == 0


async def test_auto_send_aborts_if_owner_writes_during_generation(make_harness) -> None:
    h = make_harness(
        overrides={"AUTO_SEND": "true"},
        responses=["Tayyorlangan javob"],
    )
    await h.connect()

    async def owner_intervenes() -> None:
        await h.business_message(
            "O'zim javob berdim!",
            from_id=OWNER_ID,
            chat_id=444,
            message_id=99,
        )

    h.generator.hook = owner_intervenes
    await h.business_message("Salom!", from_id=444, chat_id=444, message_id=1)

    assert Outcome.owner_message in h.assistant.outcomes
    assert h.assistant.outcomes[-1] == Outcome.owner_active
    assert len(h.sent_messages) == 0
    assert len(h.previews) == 1
    assert "owner_active" in (h.previews[0].note or "")


async def test_pause_resume_persistence_and_stranger_ignored(make_harness) -> None:
    h = make_harness(responses=["Javob"])
    await h.connect()

    # Stranger tries to /pause and send a normal message
    await h.direct_message("/pause", from_id=777888)
    await h.direct_message("Salom bot", from_id=777888)
    assert not h.assistant.state.paused
    assert len(h.sent_messages) == 0

    # Owner pauses
    await h.direct_message("/pause", from_id=OWNER_ID)
    assert h.assistant.state.paused
    assert len(h.sent_messages) == 1

    # Restart -> still paused
    h.restart()
    await h.connect()
    assert h.assistant.state.paused

    await h.business_message("Salom!", message_id=1)
    assert h.assistant.outcomes[-1] == Outcome.paused
    assert len(h.generator.calls) == 0

    # Owner resumes
    await h.direct_message("/resume", from_id=OWNER_ID)
    assert not h.assistant.state.paused

    await h.business_message("Yana salom!", message_id=2)
    assert h.assistant.outcomes[-1] == Outcome.previewed


async def test_start_command_for_owner_and_stranger(make_harness) -> None:
    h = make_harness(overrides={"AUTO_SEND": "true"})
    h.assistant.pause()

    await h.direct_message("/start", from_id=OWNER_ID)
    owner_reply = h.sent_messages[-1].text
    assert "Telegram Business" in owner_reply
    assert "Chatbots" in owner_reply
    assert "AUTO-SEND" in owner_reply
    assert "PAUZADA" in owner_reply
    assert "/pause" in owner_reply

    # Stranger /start gets short notice without profile details or AI call
    await h.direct_message("/start", from_id=999111)
    stranger_reply = h.sent_messages[-1].text
    assert "shaxsiy yordamchi bot" in stranger_reply
    assert "Sardor" not in stranger_reply
    assert len(h.generator.calls) == 0


async def test_rate_limit_reset_epoch_cooldown_persistence_and_resume(make_harness) -> None:
    h = make_harness(overrides={"AUTO_SEND": "true", "LIMIT_COOLDOWN_MINUTES": "30"})
    await h.connect()

    reset_ts = h.clock() + 600.0
    h.generator.queue.append(
        AILimitError("Quota exceeded", resets_at=reset_ts, reset_text="resets in 10m")
    )
    h.generator.queue.append("Yangi javob")

    await h.business_message("Salom 1", message_id=1)
    assert h.assistant.outcomes[-1] == Outcome.limited
    # Owner notified in auto-send mode
    assert any(m.chat_id == OWNER_ID for m in h.sent_messages)

    # Restart -> limit persists
    h.restart()
    await h.connect()
    h.clock.advance(300.0)
    await h.business_message("Salom 2", message_id=2)
    assert h.assistant.outcomes[-1] == Outcome.limited
    assert len(h.generator.calls) == 1

    # Advance past reset_ts -> works again
    h.clock.advance(301.0)
    await h.business_message("Salom 3", message_id=3)
    assert h.assistant.outcomes[-1] == Outcome.sent

    # Limit without resets_at uses LIMIT_COOLDOWN_MINUTES, and /resume clears it
    h.generator.queue.append(AILimitError("Rate limit hit"))
    h.generator.queue.append("Tozalangandan keyin")
    await h.business_message("Salom 4", message_id=4)
    assert h.assistant.outcomes[-1] == Outcome.limited

    await h.direct_message("/resume", from_id=OWNER_ID)
    await h.business_message("Salom 5", message_id=5)
    assert h.assistant.outcomes[-1] == Outcome.sent


async def test_timeout_and_auth_error_do_not_spam_owner_or_chat(make_harness) -> None:
    h = make_harness(
        overrides={"AUTO_SEND": "true"},
        responses=[
            AITimeout("Timed out"),
            AIAuthError("Invalid API key", detail="API_KEY_INVALID"),
        ],
    )
    await h.connect()

    await h.business_message("Xabar 1", chat_id=600, from_id=600, message_id=1)
    await h.business_message("Xabar 2", chat_id=600, from_id=600, message_id=2)

    assert list(h.assistant.outcomes) == [Outcome.error, Outcome.error]
    # Nothing sent to chat 600
    assert not any(m.chat_id == 600 for m in h.sent_messages)
    # Owner notified only once within the 10-minute cooldown window
    owner_notifications = [m for m in h.sent_messages if m.chat_id == OWNER_ID]
    assert len(owner_notifications) == 1
