"""Numbering pattern parsing for heading inference."""

from __future__ import annotations

from dataclasses import dataclass
import re


ARABIC_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[\.\)\:\-]|\s)+(.+)$")
LETTER_RE = re.compile(r"^([A-Z](?:\.\d+)*)(?:[\.\)\:\-]|\s)+(.+)$")
ROMAN_RE = re.compile(
    r"^((?:M{0,4})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{1,3}))(?:[\.\)\:\-]|\s)+(.+)$",
    re.IGNORECASE,
)
CHINESE_RE = re.compile(r"^第([一二三四五六七八九十百]+)([章节部分篇])\s*(.*)$")
PREFIX_RE = re.compile(r"^(Chapter|Part|Section)\s+([A-Za-z0-9.]+)\s*(?:[\.:\-]\s*)?(.+)$", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^Appendix\s+([A-Z](?:\.\d+)*)\s*(?:[\.:\-]\s*)?(.+)?$", re.IGNORECASE)


@dataclass(slots=True)
class NumberingParse:
    numbering: str | None
    numbering_type: str | None
    numbering_depth: int
    title_without_numbering: str


def _depth_from_numbering(numbering: str, numbering_type: str) -> int:
    if not numbering:
        return 0

    if numbering_type in {"arabic", "letter", "appendix", "prefix"}:
        return numbering.count(".") + 1
    if numbering_type in {"roman", "chinese"}:
        return 1
    return 0


def parse_numbering(text: str) -> NumberingParse:
    stripped = text.strip()
    if not stripped:
        return NumberingParse(None, None, 0, "")

    appendix_match = APPENDIX_RE.match(stripped)
    if appendix_match is not None:
        numbering = appendix_match.group(1).strip()
        title = (appendix_match.group(2) or "").strip() or stripped
        return NumberingParse(numbering, "appendix", _depth_from_numbering(numbering, "appendix"), title)

    prefix_match = PREFIX_RE.match(stripped)
    if prefix_match is not None:
        numbering = prefix_match.group(2).strip()
        title = prefix_match.group(3).strip() if prefix_match.group(3) else stripped
        return NumberingParse(numbering, "prefix", _depth_from_numbering(numbering, "prefix"), title)

    arabic_match = ARABIC_RE.match(stripped)
    if arabic_match is not None:
        numbering = arabic_match.group(1).strip()
        title = arabic_match.group(2).strip()
        return NumberingParse(numbering, "arabic", _depth_from_numbering(numbering, "arabic"), title)

    roman_match = ROMAN_RE.match(stripped)
    if roman_match is not None:
        numbering = roman_match.group(1).strip()
        title = roman_match.group(2).strip()
        return NumberingParse(numbering, "roman", _depth_from_numbering(numbering, "roman"), title)

    letter_match = LETTER_RE.match(stripped)
    if letter_match is not None:
        numbering = letter_match.group(1).strip()
        title = letter_match.group(2).strip()
        return NumberingParse(numbering, "letter", _depth_from_numbering(numbering, "letter"), title)

    chinese_match = CHINESE_RE.match(stripped)
    if chinese_match is not None:
        numbering = f"第{chinese_match.group(1)}{chinese_match.group(2)}"
        trailing = chinese_match.group(3).strip()
        title = trailing if trailing else stripped
        return NumberingParse(numbering, "chinese", _depth_from_numbering(numbering, "chinese"), title)

    return NumberingParse(None, None, 0, stripped)
