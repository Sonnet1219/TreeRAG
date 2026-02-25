"""Signal extraction for robust heading level inference."""

from __future__ import annotations

from dataclasses import dataclass

from tree_builder.numbering import parse_numbering
from tree_builder.preprocessor import RawHeading
from tree_builder.special_sections import match_special_section


@dataclass(slots=True)
class HeadingSignals:
    index: int
    line_index: int
    raw_text: str
    hash_count: int
    has_hash_marker: bool

    numbering: str | None
    numbering_type: str | None
    numbering_depth: int

    is_special_section: bool
    special_section_level: int

    text_length: int
    heading_text: str


def extract_heading_signals(index: int, raw_heading: RawHeading) -> HeadingSignals:
    numbering = parse_numbering(raw_heading.raw_text)
    special_level = match_special_section(numbering.title_without_numbering)

    return HeadingSignals(
        index=index,
        line_index=raw_heading.line_index,
        raw_text=raw_heading.raw_text,
        hash_count=raw_heading.hash_count,
        has_hash_marker=raw_heading.has_hash_marker,
        numbering=numbering.numbering,
        numbering_type=numbering.numbering_type,
        numbering_depth=numbering.numbering_depth,
        is_special_section=special_level is not None,
        special_section_level=special_level or 1,
        text_length=len(numbering.title_without_numbering),
        heading_text=numbering.title_without_numbering,
    )


def extract_all_signals(raw_headings: list[RawHeading]) -> list[HeadingSignals]:
    return [extract_heading_signals(index, heading) for index, heading in enumerate(raw_headings)]
