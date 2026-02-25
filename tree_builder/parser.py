"""Markdown heading parsing and section splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_builder.preprocessor import RawHeading, extract_raw_headings, normalize_heading
from tree_builder.rule_engine import DocumentContext, infer_level, infer_levels
from tree_builder.signals import extract_all_signals, extract_heading_signals


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
    """Parse one heading line and infer a level."""
    normalized = normalize_heading(line)
    if normalized is None:
        return None

    raw_heading = RawHeading(
        line_index=0,
        hash_count=int(normalized["hash_count"]),
        raw_text=str(normalized["raw_text"]),
        has_hash_marker=bool(normalized["has_hash_marker"]),
        line_text=line,
    )

    signals = extract_heading_signals(0, raw_heading)
    level, _, _ = infer_level(signals, DocumentContext([signals]))
    return HeadingInfo(
        hash_count=raw_heading.hash_count,
        numbering=signals.numbering,
        clean_title=signals.heading_text,
        inferred_level=level,
        heading_raw=raw_heading.raw_text,
    )


def _build_sections(
    lines: list[str],
    raw_headings: list[RawHeading],
) -> list[Section]:
    if not raw_headings:
        return []

    signals = extract_all_signals(raw_headings)
    inferences = infer_levels(signals)
    sections: list[Section] = []

    for index, heading in enumerate(raw_headings):
        start = heading.line_index + 1
        end = raw_headings[index + 1].line_index if index + 1 < len(raw_headings) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        inference = inferences[index]
        sections.append(
            Section(
                heading=HeadingInfo(
                    hash_count=heading.hash_count,
                    numbering=inference.signals.numbering,
                    clean_title=inference.signals.heading_text,
                    inferred_level=inference.inferred_level,
                    heading_raw=heading.raw_text,
                ),
                content=content,
                index=index + 1,
            )
        )

    return sections


def parse_markdown_sections(text: str) -> list[Section]:
    """Parse markdown text into sections keyed by inferred heading levels."""
    lines = text.splitlines()
    raw_headings, _ = extract_raw_headings(lines)
    return _build_sections(lines, raw_headings)


def parse_markdown_with_preamble(text: str) -> tuple[list[Section], str]:
    """Parse sections and return text before first heading as preamble."""
    lines = text.splitlines()
    raw_headings, _ = extract_raw_headings(lines)
    sections = _build_sections(lines, raw_headings)
    if not raw_headings:
        return sections, text.strip()
    preamble = "\n".join(lines[: raw_headings[0].line_index]).strip()
    return sections, preamble


def parse_markdown_file(path: Path) -> list[Section]:
    """Load markdown file and parse sections."""
    text = path.read_text(encoding="utf-8")
    return parse_markdown_sections(text)
