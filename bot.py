"""Entry point for the personal Telegram Business assistant bot."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramUnauthorizedError

from assistant.config import (
    SETTING_KEYS,
    ConfigError,
    Settings,
    env_file_warning,
    load_settings,
)
from assistant.gemini_client import AIError
from assistant.handlers import ALLOWED_UPDATES, build_router
from assistant.language import detect_language
from assistant.prompts import build_system_prompt, build_user_prompt
from assistant.service import (
    build_assistant,
    is_profile_unfilled,
    make_ai_client,
    read_profile,
)
from assistant.store import ensure_private_dir

logger = logging.getLogger("assistant")


def settings_from_process_env(
    env_file: Union[str, Path] = ".env",
    *,
    require_telegram: bool = True,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Settings:
    """Read SETTING_KEYS from os.environ and always scrub them from os.environ."""
    captured: Dict[str, Optional[str]] = {}
    try:
        for key in SETTING_KEYS:
            if key in os.environ:
                captured[key] = os.environ[key]
        return load_settings(
            env_file,
            require_telegram=require_telegram,
            overrides=overrides,
            source=captured,
        )
    finally:
        for key in SETTING_KEYS:
            os.environ.pop(key, None)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Personal Telegram Business assistant bot powered by Google Gemini API."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to local .env configuration file (default: .env).",
    )
    parser.add_argument(
        "--settings-from-env",
        action="store_true",
        help="Load settings from process environment variables (Railway/Docker mode) and scrub them from os.environ.",
    )
    parser.add_argument(
        "--check-ai",
        "--check-gemini",
        dest="check_ai",
        action="store_true",
        help="Verify Gemini API connectivity and API key without connecting to Telegram.",
    )
    parser.add_argument(
        "--try",
        dest="try_message",
        metavar="TEXT",
        help="Generate a sample reply for TEXT using profile.md without connecting to Telegram.",
    )
    return parser


async def _run_check_ai(settings: Settings) -> int:
    ensure_private_dir(settings.data_dir)
    client = make_ai_client(settings)
    try:
        info = await client.inspect()
    except AIError as exc:
        print(f"Gemini API check failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info.get("reply"):
        print("Gemini API check failed: empty reply.", file=sys.stderr)
        return 1
    print(f"OK — authenticated via {client.auth_mode} (model: {client.model})")
    return 0


async def _run_try_message(settings: Settings, sample_text: str) -> int:
    ensure_private_dir(settings.data_dir)
    client = make_ai_client(settings)
    lang = detect_language(sample_text)
    profile_text = read_profile(settings.profile_path)
    system_prompt = build_system_prompt(profile_text, now=datetime.now(timezone.utc))
    user_prompt = build_user_prompt(
        [
            {
                "role": "other",
                "name": "Contact",
                "text": sample_text,
                "lang": lang,
            }
        ],
        language=lang,
    )
    try:
        reply = await client.generate(system_prompt, user_prompt)
    except AIError as exc:
        print(f"Failed to generate sample reply: {exc}", file=sys.stderr)
        return 1

    print(f"Language: {lang or 'unknown'}")
    print(f"Incoming: {sample_text}")
    print(f"Reply:    {reply}")
    return 0


async def _run_bot(settings: Settings, *, env_file: Union[str, Path]) -> int:
    if not settings.from_environment:
        warning = env_file_warning(env_file)
        if warning:
            logger.warning("%s", warning)

    ensure_private_dir(settings.data_dir)
    assistant = build_assistant(settings)
    dp = Dispatcher()
    dp.include_router(build_router(assistant))

    bot = Bot(token=settings.bot_token)
    try:
        try:
            me = await bot.get_me()
        except TelegramUnauthorizedError:
            logger.error(
                "Telegram rejected TELEGRAM_BOT_TOKEN (Unauthorized). "
                "Check your token from @BotFather."
            )
            return 2

        username = f"@{me.username}" if getattr(me, "username", None) else f"id={me.id}"
        logger.info("Bot %s is running", username)
        if settings.auto_send:
            logger.warning(
                "Mode: AUTO-SEND — generated replies WILL be sent live to Telegram Business chats!"
            )
        else:
            logger.info("Mode: PREVIEW — replies are only shown in the logs and data/previews.jsonl")

        logger.info(
            "AI auth: %s (model=%s)",
            settings.ai_auth_mode,
            settings.gemini_model,
        )
        if not settings.gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY is not set! Replies will fail until GEMINI_API_KEY is configured."
            )
        if assistant.state.paused:
            logger.warning(
                "Assistant is starting in PAUSED state. Send /resume to the bot to enable replies."
            )
        if is_profile_unfilled(settings.profile_path):
            logger.warning(
                "Profile file (%s) appears to be an unfilled template. Fill in profile.md for personalized replies.",
                settings.profile_path,
            )

        await dp.start_polling(bot, allowed_updates=ALLOWED_UPDATES)
        return 0
    finally:
        await bot.session.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    require_telegram = not (args.check_ai or args.try_message is not None)

    try:
        if args.settings_from_env:
            settings = settings_from_process_env(
                args.env_file,
                require_telegram=require_telegram,
            )
        else:
            settings = load_settings(
                args.env_file,
                require_telegram=require_telegram,
            )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.check_ai:
        return asyncio.run(_run_check_ai(settings))

    if args.try_message is not None:
        return asyncio.run(_run_try_message(settings, args.try_message))

    return asyncio.run(_run_bot(settings, env_file=args.env_file))


if __name__ == "__main__":
    sys.exit(main())
