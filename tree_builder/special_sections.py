"""Special section matching used by rule-based level inference."""

from __future__ import annotations

SPECIAL_SECTIONS: dict[str, int] = {
    "abstract": 1,
    "摘要": 1,
    "introduction": 1,
    "引言": 1,
    "绪论": 1,
    "related work": 1,
    "background": 1,
    "methodology": 1,
    "method": 1,
    "methods": 1,
    "approach": 1,
    "experiments": 1,
    "evaluation": 1,
    "results": 1,
    "discussion": 1,
    "conclusion": 1,
    "conclusions": 1,
    "summary": 1,
    "acknowledgments": 1,
    "acknowledgements": 1,
    "references": 1,
    "bibliography": 1,
    "appendix": 1,
    "supplementary": 1,
    "future work": 1,
}


def match_special_section(heading_text: str) -> int | None:
    """Return default section level if heading matches known special sections."""
    normalized = heading_text.lower().strip()
    for pattern, level in SPECIAL_SECTIONS.items():
        if normalized == pattern or normalized.startswith(f"{pattern} ") or normalized.startswith(f"{pattern}:"):
            return level
    return None
