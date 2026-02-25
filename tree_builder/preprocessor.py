"""Preprocessing utilities for robust markdown heading extraction."""

from __future__ import annotations

from dataclasses import dataclass
import re


ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ATX_HEADING_LOOSE_RE = re.compile(r"^(#{1,6})(\S.*?)\s*$")
TRAILING_HASH_RE = re.compile(r"\s+#+\s*$")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
BOLD_ITALIC_RE = re.compile(r"(\*\*|__|\*|_)(.+?)\1")
MULTI_SPACE_RE = re.compile(r"\s+")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
ENDING_PUNCTUATION_RE = re.compile(r"[\.!\?;:。！？；：]$")
NUMBERING_HINT_RE = re.compile(
    r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*|[IVXLCDM]+|第[一二三四五六七八九十百]+[章节部分篇])(?:[\.\s\):\-]+|\s+).+",
    re.IGNORECASE,
)
SPECIAL_HINT_RE = re.compile(
    r"^(abstract|introduction|related work|methods?|experiments?|results?|discussion|conclusion|references|acknowledg?ments?|appendix)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RawHeading:
    line_index: int
    hash_count: int
    raw_text: str
    has_hash_marker: bool
    line_text: str


def mark_code_blocks(lines: list[str]) -> set[int]:
    """Return line indices that belong to fenced code blocks."""
    in_code = False
    code_lines: set[int] = set()

    for index, line in enumerate(lines):
        if FENCE_RE.match(line.strip()):
            code_lines.add(index)
            in_code = not in_code
            continue
        if in_code:
            code_lines.add(index)

    return code_lines


def _strip_markdown_formatting(text: str) -> str:
    cleaned = text
    cleaned = TRAILING_HASH_RE.sub("", cleaned)
    cleaned = MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    # Apply emphasis replacement iteratively for nested formatting.
    for _ in range(3):
        updated = BOLD_ITALIC_RE.sub(r"\2", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = cleaned.strip().strip("\u3000")
    cleaned = cleaned.rstrip("。.；;:")
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def normalize_heading(raw_line: str) -> dict[str, int | str | bool] | None:
    """Normalize one heading line and return parsed fields."""
    normalized_line = raw_line.lstrip("\ufeff")
    match = ATX_HEADING_RE.match(normalized_line)
    if match is None:
        match = ATX_HEADING_LOOSE_RE.match(normalized_line)
    if match is None:
        return None

    hashes = match.group(1)
    raw_text = _strip_markdown_formatting(match.group(2))
    if not raw_text:
        return None

    return {
        "hash_count": len(hashes),
        "raw_text": raw_text,
        "has_hash_marker": True,
    }


def detect_unmarked_heading(line: str, prev_line: str, next_line: str) -> bool:
    """Heuristic detection for heading lines that do not use '#' markers."""
    stripped = line.lstrip("\ufeff").strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if stripped.startswith("#"):
        return False
    if "," in stripped or "，" in stripped:
        return False
    if ENDING_PUNCTUATION_RE.search(stripped):
        return False

    prev_blank = not prev_line.strip()
    next_blank = not next_line.strip()
    isolated = prev_blank and next_blank

    looks_numbered = NUMBERING_HINT_RE.match(stripped) is not None
    looks_special = SPECIAL_HINT_RE.match(stripped) is not None

    return isolated and (looks_numbered or looks_special)


def extract_raw_headings(lines: list[str]) -> tuple[list[RawHeading], set[int]]:
    """Extract ordered headings from markdown lines, skipping code blocks."""
    code_block_lines = mark_code_blocks(lines)
    raw_headings: list[RawHeading] = []

    for index, line in enumerate(lines):
        if index in code_block_lines:
            continue

        normalized = normalize_heading(line)
        if normalized is not None:
            raw_headings.append(
                RawHeading(
                    line_index=index,
                    hash_count=int(normalized["hash_count"]),
                    raw_text=str(normalized["raw_text"]),
                    has_hash_marker=bool(normalized["has_hash_marker"]),
                    line_text=line,
                )
            )
            continue

        prev_line = lines[index - 1] if index > 0 else ""
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if detect_unmarked_heading(line, prev_line, next_line):
            raw_headings.append(
                RawHeading(
                    line_index=index,
                    hash_count=0,
                    raw_text=line.strip(),
                    has_hash_marker=False,
                    line_text=line,
                )
            )

    return raw_headings, code_block_lines
