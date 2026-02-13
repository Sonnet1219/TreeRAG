"""Markdown heading parsing and section splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
NUMBERED_HEADING_RE = re.compile(r"^([\d]+(?:\.[\d]+)*)[\.\s\)\-]?\s*(.+)$")
LETTER_NUMBERED_HEADING_RE = re.compile(r"^([A-Z](?:\.[\d]+)+)[\.\s\)\-]?\s*(.+)$")
FENCE_RE = re.compile(r"^\s*```")
TRAILING_HASH_RE = re.compile(r"\s+#+\s*$")


@dataclass
class HeadingInfo:
    hash_count: int
    numbering: str | None
    clean_title: str
    inferred_level: int
    heading_raw: str


@dataclass
class Section:
    heading: HeadingInfo
    content: str
    index: int


def parse_heading_line(line: str) -> HeadingInfo | None:
    """Parse one ATX heading line and infer level."""
    match = ATX_HEADING_RE.match(line)
    if match is None:
        return None

    hashes = match.group(1)
    heading_raw = TRAILING_HASH_RE.sub("", match.group(2)).strip()

    numbering: str | None = None
    clean_title = heading_raw
    for pattern in (NUMBERED_HEADING_RE, LETTER_NUMBERED_HEADING_RE):
        number_match = pattern.match(heading_raw)
        if number_match is not None:
            numbering = number_match.group(1)
            clean_title = number_match.group(2).strip()
            break

    if numbering is None:
        inferred_level = min(len(hashes), 3)
    else:
        inferred_level = min(numbering.count(".") + 1, 3)

    return HeadingInfo(
        hash_count=len(hashes),
        numbering=numbering,
        clean_title=clean_title,
        inferred_level=inferred_level,
        heading_raw=heading_raw,
    )


def _parse_sections_with_preamble(text: str) -> tuple[list[Section], str]:
    sections: list[Section] = []
    preamble_lines: list[str] = []

    current_heading: HeadingInfo | None = None
    current_lines: list[str] = []
    in_fence = False
    section_index = 0

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if current_heading is None:
                preamble_lines.append(line)
            else:
                current_lines.append(line)
            continue

        if not in_fence:
            heading = parse_heading_line(line)
            if heading is not None:
                if current_heading is not None:
                    sections.append(
                        Section(
                            heading=current_heading,
                            content="\n".join(current_lines).strip(),
                            index=section_index,
                        )
                    )
                section_index += 1
                current_heading = heading
                current_lines = []
                continue

        if current_heading is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append(
            Section(
                heading=current_heading,
                content="\n".join(current_lines).strip(),
                index=section_index,
            )
        )

    return sections, "\n".join(preamble_lines).strip()


def parse_markdown_sections(text: str) -> list[Section]:
    """Parse markdown text into sections keyed by headings."""
    sections, _ = _parse_sections_with_preamble(text)
    return sections


def parse_markdown_with_preamble(text: str) -> tuple[list[Section], str]:
    """Parse sections and return text before first heading as preamble."""
    return _parse_sections_with_preamble(text)


def parse_markdown_file(path: Path) -> list[Section]:
    """Load markdown file and parse sections."""
    text = path.read_text(encoding="utf-8")
    return parse_markdown_sections(text)
