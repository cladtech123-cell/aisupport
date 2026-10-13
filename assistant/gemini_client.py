"""Async Google Gemini API client with key redaction, timeout, and error classification."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

import aiohttp

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

TransportFunc = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Awaitable[Tuple[int, str]],
]

_LIMIT_RE = re.compile(
    r"resource_exhausted|quota|rate[ _-]?limit|usage limit|hit your (?:usage )?limit|"
    r"limit reached|out of (?:extra )?usage|too many requests",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"api_key_invalid|api key not valid|invalid api key|authentication|"
    r"unauthori[sz]ed|api key expired|invalid_argument.*api key",
    re.IGNORECASE,
)
_EPOCH_PIPE_RE = re.compile(r"\|(\d{9,12})\b")
_RESET_TEXT_RE = re.compile(r"(resets\s+[^.|;\n]+)", re.IGNORECASE)
_RETRY_SECONDS_RE = re.compile(
    r"(?:retry\s+in\s+|\"retryDelay\"\s*:\s*\")(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)


class AIError(Exception):
    """Base error for AI generation failures."""


class AITimeout(AIError):
    """Raised when the AI request exceeds the configured timeout."""


class AIAuthError(AIError):
    """Raised when the Gemini API key is missing, invalid, or unauthorized."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class AILimitError(AIError):
    """Raised when the Gemini API quota or rate limit is reached."""

    def __init__(
        self,
        detail: str,
        resets_at: Optional[float] = None,
        reset_text: Optional[str] = None,
    ) -> None:
        self.detail = detail
        self.resets_at = resets_at
        self.reset_text = reset_text
        suffix = f" ({reset_text})" if reset_text else ""
        super().__init__(f"Gemini API usage limit reached{suffix}: {detail}")


def redact_secret(text: str, secret: Optional[str]) -> str:
    """Replace any occurrence of secret in text with [REDACTED]."""
    if not text or not secret:
        return text or ""
    return text.replace(secret, "[REDACTED]")


def classify_failure(
    message: str,
    status: Optional[int] = None,
    *,
    now: Optional[float] = None,
) -> AIError:
    """Classify an API error message and HTTP status into a specific AIError."""
    raw = (message or "").strip() or (
        f"HTTP status {status}" if status is not None else "Unknown AI error"
    )

    epoch_match = _EPOCH_PIPE_RE.search(raw)
    resets_at: Optional[float] = None
    if epoch_match:
        try:
            resets_at = float(int(epoch_match.group(1)))
        except ValueError:
            resets_at = None

    cleaned = _EPOCH_PIPE_RE.sub("", raw).strip()

    reset_match = _RESET_TEXT_RE.search(cleaned)
    reset_text: Optional[str] = reset_match.group(1).strip() if reset_match else None

    if resets_at is None:
        retry_match = _RETRY_SECONDS_RE.search(cleaned)
        if retry_match:
            try:
                delay_sec = float(retry_match.group(1))
                base_now = now if now is not None else time.time()
                resets_at = base_now + delay_sec
                if not reset_text:
                    reset_text = f"resets in {int(delay_sec)}s"
            except ValueError:
                pass

    if status == 429 or _LIMIT_RE.search(cleaned):
        return AILimitError(
            detail=cleaned[:500],
            resets_at=resets_at,
            reset_text=reset_text,
        )

    if status in (401, 403) or _AUTH_RE.search(cleaned):
        short_detail = cleaned[:300]
        return AIAuthError(
            "Gemini API key was rejected or is invalid. Generate a valid key at "
            "https://aistudio.google.com/apikey and update GEMINI_API_KEY. "
            f"API said: {short_detail}",
            detail=short_detail,
        )

    return AIError(cleaned[:500])


def parse_gemini_response(
    status: int,
    body: str,
    *,
    secret: Optional[str] = None,
) -> str:
    """Parse HTTP status and JSON body from Gemini generateContent endpoint."""
    safe_body = redact_secret(body or "", secret)

    try:
        data = json.loads(safe_body) if safe_body.strip() else {}
    except json.JSONDecodeError:
        if status >= 400:
            raise classify_failure(safe_body or f"HTTP {status}", status)
        raise AIError(f"Invalid JSON response from Gemini API: {safe_body[:300]}")

    if not isinstance(data, dict):
        raise AIError("Unexpected response structure from Gemini API.")

    if "error" in data or status >= 400:
        err_obj = data.get("error")
        if isinstance(err_obj, dict):
            err_status = err_obj.get("code") or status
            err_msg = str(err_obj.get("message") or "")
            err_code_str = str(err_obj.get("status") or "")
            details_str = json.dumps(err_obj.get("details") or [])
            combined = " ".join(
                part for part in (err_code_str, err_msg, details_str) if part
            )
            raise classify_failure(
                redact_secret(combined, secret),
                int(err_status) if isinstance(err_status, int) else status,
            )
        raise classify_failure(safe_body or f"HTTP {status}", status)

    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
            reason = prompt_feedback.get("blockReason")
            raise AIError(f"Gemini blocked the prompt (reason: {reason}).")
        raise AIError("Gemini returned no candidates.")

    first = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first.get("content") if isinstance(first.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []

    texts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])

    result = "".join(texts).strip()
    if not result:
        finish_reason = first.get("finishReason")
        raise AIError(
            f"Gemini returned an empty response (finishReason={finish_reason})."
        )
    return redact_secret(result, secret)


async def _default_http_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Tuple[int, str]:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=dict(headers), json=payload) as resp:
            text = await resp.text()
            return resp.status, text


class GeminiClient:
    """Async client for generating replies with Google Gemini API."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: str = "gemini-3.8-flash",
        timeout: int = 60,
        auth_label: Optional[str] = None,
        transport: Optional[TransportFunc] = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.timeout = max(1, int(timeout))
        self._auth_label = auth_label
        self._transport: TransportFunc = transport or _default_http_transport

    @property
    def auth_mode(self) -> str:
        if self._auth_label:
            return self._auth_label
        return "GEMINI_API_KEY" if self._api_key else "missing GEMINI_API_KEY"

    def __repr__(self) -> str:
        return (
            f"GeminiClient(model={self.model!r}, timeout={self.timeout!r}, "
            f"auth_mode={self.auth_mode!r})"
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a single reply from Gemini using system and user prompts."""
        if not self._api_key:
            raise AIAuthError(
                "GEMINI_API_KEY is not configured. Get a free API key from "
                "https://aistudio.google.com/apikey and set GEMINI_API_KEY in .env "
                "or Railway Variables.",
                detail="missing GEMINI_API_KEY",
            )

        url = f"{GEMINI_API_BASE}/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        payload: Dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.6,
            },
        }

        status = 0
        body = ""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                status, body = await asyncio.wait_for(
                    self._transport(url, headers, payload, float(self.timeout)),
                    timeout=float(self.timeout),
                )
            except asyncio.TimeoutError as exc:
                raise AITimeout(
                    f"Gemini API request timed out after {self.timeout}s."
                ) from exc
            except aiohttp.ClientError as exc:
                safe_err = redact_secret(str(exc), self._api_key)
                raise AIError(f"Gemini API network error: {safe_err[:300]}") from exc

            if status in (502, 503, 504) and attempt < max_attempts:
                await asyncio.sleep(1.5 * attempt)
                continue
            break

        return parse_gemini_response(status, body, secret=self._api_key)

    async def inspect(self) -> Dict[str, Any]:
        """Run a minimal connectivity and auth check against the Gemini API."""
        reply = await self.generate(
            system_prompt="Reply with exactly: OK",
            user_prompt="ping",
        )
        return {
            "auth": self.auth_mode,
            "provider": "google-gemini",
            "model": self.model,
            "reply": reply,
        }
