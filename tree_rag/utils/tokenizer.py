"""Tokenization helpers for English and Chinese text."""

from __future__ import annotations

import re


EN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
}


def _contains_cjk(text: str) -> bool:
    return bool(CJK_CHAR_RE.search(text))


def _tokenize_cjk_fallback(text: str) -> list[str]:
    tokens: list[str] = []
    word: list[str] = []
    for ch in text:
        if CJK_CHAR_RE.match(ch):
            if word:
                tokens.extend(EN_TOKEN_RE.findall("".join(word).lower()))
                word = []
            tokens.append(ch)
            continue
        word.append(ch)
    if word:
        tokens.extend(EN_TOKEN_RE.findall("".join(word).lower()))
    return tokens


def tokenize(text: str) -> list[str]:
    normalized = text.strip().lower()
    if not normalized:
        return []

    if _contains_cjk(normalized):
        try:
            import jieba  # type: ignore

            tokens = [item.strip().lower() for item in jieba.cut(normalized) if item.strip()]
        except ImportError:
            tokens = _tokenize_cjk_fallback(normalized)
    else:
        tokens = EN_TOKEN_RE.findall(normalized)

    return [token for token in tokens if token not in STOPWORDS]
