"""Lightweight heuristic language detection for Uzbek, Russian, and English."""

from __future__ import annotations

import re
from typing import Optional

APOSTROPHE_CHARS = "'ʻʼ’‘`´"
_APOSTROPHE_RE = re.compile(f"[{re.escape(APOSTROPHE_CHARS)}]")

_CYRILLIC_RE = re.compile(r"[а-яёўқғҳА-ЯЁЎҚҒҲ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_UZ_CYRILLIC_CHARS_RE = re.compile(r"[ўқғҳЎҚҒҲ]")

_UZ_CYRILLIC_WORDS = frozenset(
    {
        "салом",
        "ассалому",
        "алайкум",
        "рахмат",
        "раҳмат",
        "яхши",
        "нима",
        "қалесан",
        "қалайсан",
        "калесан",
        "йўқ",
        "йук",
        "бугун",
        "эртага",
        "керак",
        "хозир",
        "ҳозир",
        "кейин",
        "мен",
        "сен",
        "сиз",
        "бор",
        "хўп",
        "хуп",
        "бўпти",
        "бупти",
        "зўр",
        "зур",
        "қаерда",
        "каерда",
        "қаердасан",
        "каердасан",
        "қачон",
        "качон",
        "ака",
        "ука",
    }
)

_UZ_CYRILLIC_SUFFIXES = (
    "мисан",
    "мисиз",
    "мизми",
    "япсан",
    "япман",
    "япти",
    "япмиз",
    "асизми",
    "ганман",
    "дингми",
)

_UZ_LATIN_WORDS = frozenset(
    {
        "salom",
        "assalomu",
        "alaykum",
        "rahmat",
        "raxmat",
        "yaxshi",
        "nima",
        "qalesan",
        "qalaysan",
        "qayerda",
        "qayerdasan",
        "qachon",
        "bugun",
        "ertaga",
        "kerak",
        "xop",
        "hop",
        "bopti",
        "zor",
        "hozir",
        "keyin",
        "men",
        "sen",
        "siz",
        "yoq",
        "bor",
        "aka",
        "uka",
        "opa",
        "qanday",
        "tinchmi",
        "mayli",
        "albatta",
        "tushundim",
        "bilmadim",
        "iltimos",
        "kechirasiz",
    }
)

_UZ_LATIN_SUFFIXES = (
    "misan",
    "misiz",
    "mizmi",
    "yapsan",
    "yapman",
    "yapti",
    "yapmiz",
    "asizmi",
    "ganman",
    "dingmi",
)

_ENGLISH_APOSTROPHE_EXCEPTIONS = frozenset(
    {
        "o'clock",
        "g'day",
    }
)

_UZ_OG_RE = re.compile(r"[og]'[a-z]")


def _has_uzbek_og_letter(token: str) -> bool:
    """Return True if token contains Uzbek o' or g' rather than English possessive/o'clock."""
    if token in _ENGLISH_APOSTROPHE_EXCEPTIONS:
        return False
    # Ignore English possessive endings like "dog's", "king's", "thing's"
    cleaned = re.sub(r"'s$", "", token)
    return bool(_UZ_OG_RE.search(cleaned))


def detect_language(text: Optional[str]) -> Optional[str]:
    """Return 'uz', 'ru', 'en', or None if text has no alphabetic characters."""
    if not text:
        return None

    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))

    if cyrillic_count == 0 and latin_count == 0:
        return None

    normalized = _APOSTROPHE_RE.sub("'", text.lower())

    if cyrillic_count > latin_count:
        if _UZ_CYRILLIC_CHARS_RE.search(normalized):
            return "uz"
        cyr_words = re.findall(r"[а-яёўқғҳ]+", normalized)
        for word in cyr_words:
            if word in _UZ_CYRILLIC_WORDS:
                return "uz"
            if len(word) > 5 and word.endswith(_UZ_CYRILLIC_SUFFIXES):
                return "uz"
        return "ru"

    # Latin script branch
    raw_tokens = re.findall(r"[a-z]+(?:'[a-z]+)*", normalized)
    for token in raw_tokens:
        if _has_uzbek_og_letter(token):
            return "uz"
        stripped = token.replace("'", "")
        if stripped in _UZ_LATIN_WORDS:
            return "uz"
        if len(stripped) > 5 and stripped.endswith(_UZ_LATIN_SUFFIXES):
            return "uz"

    return "en"
