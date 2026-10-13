"""Configuration loading and validation without polluting os.environ."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import stat
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from dotenv import dotenv_values

SETTING_KEYS: Tuple[str, ...] = (
    "TELEGRAM_BOT_TOKEN",
    "OWNER_ID",
    "AUTO_SEND",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_TIMEOUT_SECONDS",
    "LIMIT_COOLDOWN_MINUTES",
    "OWNER_TAKEOVER_MINUTES",
    "HISTORY_LIMIT",
    "DATA_DIR",
    "PROFILE_PATH",
)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
_LEGACY_DEFAULT_MODELS = frozenset({"gemini-2.5-flash"})

_BOT_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{5,}$")
_BOOL_TRUE = frozenset({"true", "1", "yes", "on"})
_BOOL_FALSE = frozenset({"false", "0", "no", "off"})


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid (never includes secret values)."""


@dataclass(frozen=True)
class Settings:
    """Validated, immutable application configuration."""

    bot_token: str = field(repr=False)
    owner_id: int
    auto_send: bool = False
    gemini_api_key: Optional[str] = field(default=None, repr=False)
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_timeout_seconds: int = 60
    limit_cooldown_minutes: int = 30
    owner_takeover_minutes: int = 15
    history_limit: int = 30
    data_dir: Path = field(default_factory=lambda: Path("data"))
    profile_path: Path = field(default_factory=lambda: Path("profile.md"))
    from_environment: bool = False

    @property
    def ai_auth_mode(self) -> str:
        if not self.gemini_api_key:
            return "missing GEMINI_API_KEY"
        if self.from_environment:
            return "GEMINI_API_KEY from environment variables"
        return "GEMINI_API_KEY from .env"


def _parse_bool(raw: Optional[str], *, key: str, location: str, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    lowered = raw.strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ConfigError(
        f"Invalid {key} in {location}: expected one of true/false/1/0/yes/no/on/off."
    )


def _parse_int(
    raw: Optional[str],
    *,
    key: str,
    location: str,
    default: Optional[int],
    minimum: Optional[int] = None,
) -> int:
    if raw is None or raw.strip() == "":
        if default is not None:
            return default
        raise ConfigError(f"Missing required {key} in {location}.")
    text = raw.strip()
    if not re.fullmatch(r"-?\d+", text):
        raise ConfigError(f"Invalid {key} in {location}: must be an integer.")
    value = int(text)
    if minimum is not None and value < minimum:
        raise ConfigError(
            f"Invalid {key} in {location}: must be at least {minimum}."
        )
    return value


def _validate_gemini_api_key(raw: Optional[str], *, location: str) -> Optional[str]:
    if raw is None:
        return None
    if raw == "":
        return None
    if any(ch.isspace() for ch in raw):
        raise ConfigError(
            f"Invalid GEMINI_API_KEY in {location}: must be on a single line with no whitespace."
        )
    cleaned = raw.strip()
    if cleaned.startswith("sk-ant-"):
        raise ConfigError(
            f"Invalid GEMINI_API_KEY in {location}: this looks like an Anthropic key, "
            "but this bot uses Google Gemini API (get a key from https://aistudio.google.com/apikey)."
        )
    if len(cleaned) < 10:
        raise ConfigError(
            f"Invalid GEMINI_API_KEY in {location}: API key is too short."
        )
    return cleaned


def load_settings(
    env_file: Union[str, Path] = ".env",
    *,
    require_telegram: bool = True,
    overrides: Optional[Mapping[str, Any]] = None,
    source: Optional[Mapping[str, Optional[str]]] = None,
) -> Settings:
    """Load and validate settings from either a .env file or an explicit source mapping."""
    env_path = Path(env_file)
    base_dir = env_path.resolve().parent if env_path.parent != Path("") else Path.cwd()

    from_environment = source is not None
    if source is not None:
        raw_values: Dict[str, Optional[str]] = {
            k: (str(v) if v is not None else None) for k, v in source.items()
        }
        location = "the environment variables"
    else:
        location = str(env_path)
        if not env_path.exists():
            if overrides is None:
                raise ConfigError(
                    f"Configuration file {env_path} was not found. "
                    "Create it with: cp .env.example .env && chmod 600 .env"
                )
            raw_values = {}
        else:
            loaded = dotenv_values(env_path)
            raw_values = {k: v for k, v in loaded.items()}

    if overrides:
        for k, v in overrides.items():
            raw_values[k] = str(v) if v is not None else None

    raw_bot_token = (raw_values.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if require_telegram:
        if not raw_bot_token:
            raise ConfigError(f"Missing TELEGRAM_BOT_TOKEN in {location}.")
        if not _BOT_TOKEN_RE.match(raw_bot_token):
            raise ConfigError(
                f"Invalid TELEGRAM_BOT_TOKEN in {location}: expected format '<digits>:<token>'."
            )
    else:
        if raw_bot_token and not _BOT_TOKEN_RE.match(raw_bot_token):
            raise ConfigError(
                f"Invalid TELEGRAM_BOT_TOKEN in {location}: expected format '<digits>:<token>'."
            )

    raw_owner_id = (raw_values.get("OWNER_ID") or "").strip()
    if require_telegram:
        if not raw_owner_id:
            raise ConfigError(f"Missing OWNER_ID in {location}.")
        if not raw_owner_id.isdigit() or int(raw_owner_id) <= 0:
            raise ConfigError(
                f"Invalid OWNER_ID in {location}: must be a numeric Telegram user ID."
            )
        owner_id = int(raw_owner_id)
    else:
        if raw_owner_id:
            if not raw_owner_id.isdigit() or int(raw_owner_id) <= 0:
                raise ConfigError(
                    f"Invalid OWNER_ID in {location}: must be a numeric Telegram user ID."
                )
            owner_id = int(raw_owner_id)
        else:
            owner_id = 0

    auto_send = _parse_bool(
        raw_values.get("AUTO_SEND"),
        key="AUTO_SEND",
        location=location,
        default=False,
    )

    gemini_api_key = _validate_gemini_api_key(
        raw_values.get("GEMINI_API_KEY"),
        location=location,
    )

    raw_model = (raw_values.get("GEMINI_MODEL") or "").strip()
    if not raw_model or raw_model in _LEGACY_DEFAULT_MODELS:
        gemini_model = DEFAULT_GEMINI_MODEL
    else:
        gemini_model = raw_model
    if any(ch.isspace() for ch in gemini_model):
        raise ConfigError(
            f"Invalid GEMINI_MODEL in {location}: model name must not contain whitespace."
        )

    gemini_timeout_seconds = _parse_int(
        raw_values.get("GEMINI_TIMEOUT_SECONDS"),
        key="GEMINI_TIMEOUT_SECONDS",
        location=location,
        default=60,
        minimum=1,
    )
    limit_cooldown_minutes = _parse_int(
        raw_values.get("LIMIT_COOLDOWN_MINUTES"),
        key="LIMIT_COOLDOWN_MINUTES",
        location=location,
        default=30,
        minimum=1,
    )
    owner_takeover_minutes = _parse_int(
        raw_values.get("OWNER_TAKEOVER_MINUTES"),
        key="OWNER_TAKEOVER_MINUTES",
        location=location,
        default=15,
        minimum=0,
    )
    history_limit = _parse_int(
        raw_values.get("HISTORY_LIMIT"),
        key="HISTORY_LIMIT",
        location=location,
        default=30,
        minimum=1,
    )

    raw_data_dir = (raw_values.get("DATA_DIR") or "").strip()
    if raw_data_dir:
        data_dir = Path(raw_data_dir)
        if not data_dir.is_absolute():
            data_dir = base_dir / data_dir
    else:
        data_dir = base_dir / "data"

    raw_profile_path = (raw_values.get("PROFILE_PATH") or "").strip()
    if raw_profile_path:
        profile_path = Path(raw_profile_path)
        if not profile_path.is_absolute():
            profile_path = base_dir / profile_path
    else:
        profile_path = base_dir / "profile.md"

    return Settings(
        bot_token=raw_bot_token,
        owner_id=owner_id,
        auto_send=auto_send,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        gemini_timeout_seconds=gemini_timeout_seconds,
        limit_cooldown_minutes=limit_cooldown_minutes,
        owner_takeover_minutes=owner_takeover_minutes,
        history_limit=history_limit,
        data_dir=data_dir,
        profile_path=profile_path,
        from_environment=from_environment,
    )


def _get_file_mode(target: Path) -> Optional[int]:
    if os.name == "nt":
        return None
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except OSError:
        return None


def env_file_warning(path: Union[str, Path]) -> Optional[str]:
    """Return a warning if the .env file is readable/writable by group or others."""
    target = Path(path)
    if not target.exists():
        return None
    mode = _get_file_mode(target)
    if mode is not None and (mode & 0o077):
        return (
            f"Permissions on {target} are too open ({oct(mode)}). "
            f"Run: chmod 600 {target}"
        )
    return None
