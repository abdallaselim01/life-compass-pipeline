"""
Small helpers for comparing Arabic text. Used for exact-duplicate detection
and for reporting; nothing in the pipeline uses them to judge meaning.
"""
from __future__ import annotations

import re

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_EDGE_PUNCT = re.compile(r"^\W+|\W+$", re.UNICODE)

_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ة": "ه", "ؤ": "", "ئ": "", "ء": "",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})


def strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text)


def core(token: str) -> str:
    """The token without leading/trailing punctuation or quotes."""
    return _EDGE_PUNCT.sub("", token)


def skeleton(word: str) -> str:
    """Spelling-insensitive form of a word."""
    return strip_diacritics(word).lower().translate(_FOLD)


def skel_token(token: str) -> str:
    return skeleton(core(token))


def words(text: str) -> list:
    """Skeleton words of a text (punctuation dropped, empties skipped)."""
    out = []
    for tok in text.split():
        s = skel_token(tok)
        if s:
            out.append(s)
    return out
