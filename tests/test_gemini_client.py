"""Offline tests for GeminiClient, error classification, response parsing, and secret redaction."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Mapping, Tuple

import pytest

from assistant.gemini_client import (
    AIAuthError,
    AIError,
    AILimitError,
    AITimeout,
    GeminiClient,
    classify_failure,
    parse_gemini_response,
)

SECRET_KEY = "AIzaSySuperSecretKeyValue999999"


async def test_gemini_client_strips_reply_and_keeps_key_out_of_url_and_repr() -> None:
    captured: Dict[str, Any] = {}

    async def fake_transport(
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Tuple[int, str]:
        captured["url"] = url
        captured["headers"] = dict(headers)
        captured["payload"] = dict(payload)
        body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "  Salom, yaxshi rahmat! \n"}],
                        }
                    }
                ]
            }
        )
        return 200, body

    client = GeminiClient(
        api_key=SECRET_KEY,
        model="gemini-2.5-flash",
        timeout=30,
        auth_label="GEMINI_API_KEY from .env",
        transport=fake_transport,
    )

    assert SECRET_KEY not in repr(client)
    reply = await client.generate("System rules", "User conversation")
    assert reply == "Salom, yaxshi rahmat!"
    assert SECRET_KEY not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == SECRET_KEY
    assert (
        captured["payload"]["systemInstruction"]["parts"][0]["text"] == "System rules"
    )
    assert (
        captured["payload"]["contents"][0]["parts"][0]["text"] == "User conversation"
    )


async def test_gemini_client_redacts_secret_in_auth_error() -> None:
    async def fake_transport(
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Tuple[int, str]:
        body = json.dumps(
            {
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": f"API key not valid: {SECRET_KEY}. Please pass a valid key.",
                }
            }
        )
        return 400, body

    client = GeminiClient(api_key=SECRET_KEY, transport=fake_transport)
    with pytest.raises(AIAuthError) as exc_info:
        await client.generate("sys", "usr")

    err_text = str(exc_info.value)
    assert SECRET_KEY not in err_text
    assert "[REDACTED]" in err_text
    assert "GEMINI_API_KEY" in err_text
    assert "API said:" in err_text


async def test_gemini_client_inspect_returns_auth_and_model() -> None:
    async def fake_transport(
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Tuple[int, str]:
        body = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}
        )
        return 200, body

    client = GeminiClient(
        api_key=SECRET_KEY,
        model="gemini-2.5-flash",
        auth_label="GEMINI_API_KEY from .env",
        transport=fake_transport,
    )
    info = await client.inspect()
    assert info["auth"] == "GEMINI_API_KEY from .env"
    assert info["provider"] == "google-gemini"
    assert info["model"] == "gemini-2.5-flash"
    assert info["reply"] == "OK"


async def test_gemini_client_timeout_raises_ai_timeout() -> None:
    async def slow_transport(
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout: float,
    ) -> Tuple[int, str]:
        await asyncio.sleep(5.0)
        return 200, "{}"

    client = GeminiClient(api_key=SECRET_KEY, timeout=1, transport=slow_transport)
    with pytest.raises(AITimeout):
        await client.generate("sys", "usr")


async def test_gemini_client_missing_api_key_raises_auth_error() -> None:
    client = GeminiClient(api_key=None)
    with pytest.raises(AIAuthError):
        await client.generate("sys", "usr")


def test_classify_failure_variants() -> None:
    # Epoch rate limit
    err_epoch = classify_failure("Usage limit reached|1760000500", 429)
    assert isinstance(err_epoch, AILimitError)
    assert err_epoch.resets_at == 1760000500.0
    assert "1760000500" not in err_epoch.detail

    # Text reset rate limit
    err_text = classify_failure("Quota exceeded, resets 3pm UTC.", 200)
    assert isinstance(err_text, AILimitError)
    assert err_text.reset_text == "resets 3pm UTC"

    # Gemini retryDelay rate limit
    err_retry = classify_failure(
        'RESOURCE_EXHAUSTED: Quota exceeded. [{"retryDelay": "45s"}]',
        429,
        now=1000.0,
    )
    assert isinstance(err_retry, AILimitError)
    assert err_retry.resets_at == 1045.0

    # Auth error
    err_auth = classify_failure("API_KEY_INVALID: key rejected", 403)
    assert isinstance(err_auth, AIAuthError)

    # Generic error
    err_generic = classify_failure("Internal server error", 500)
    assert type(err_generic) is AIError


def test_parse_gemini_response_empty_and_invalid_json() -> None:
    with pytest.raises(AIError):
        parse_gemini_response(200, '{"candidates": [{"content": {"parts": [{"text": "   "}]}}]}')

    with pytest.raises(AIError):
        parse_gemini_response(502, "Bad Gateway HTML")
