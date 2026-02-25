"""Rule-based heading level inference and confidence scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tree_builder.signals import HeadingSignals


LLM_CONFIDENCE_THRESHOLD = 0.6


@dataclass(slots=True)
class LevelInference:
    signals: HeadingSignals
    inferred_level: int
    confidence: float
    reason: str
    source: str = "rule"


class DocumentContext:
    """Document-wide signals that help classify ambiguous heading levels."""

    def __init__(self, all_headings: list[HeadingSignals]) -> None:
        self.all_headings = all_headings
        self.hash_distribution = Counter(
            heading.hash_count for heading in all_headings if heading.has_hash_marker
        )
        self.has_any_numbering = any(heading.numbering_depth > 0 for heading in all_headings)
        self.numbering_coverage = (
            sum(1 for heading in all_headings if heading.numbering_depth > 0) / len(all_headings)
            if all_headings
            else 0.0
        )

    def check_hash_consistency(self) -> str:
        if not self.hash_distribution:
            return "inconsistent"
        if len(self.hash_distribution) == 1:
            return "all_same"
        return "consistent"

    def get_dominant_numbering_type(self) -> str | None:
        types = [heading.numbering_type for heading in self.all_headings if heading.numbering_type]
        if not types:
            return None
        return Counter(types).most_common(1)[0][0]


def infer_level(signals: HeadingSignals, context: DocumentContext, max_depth: int = 3) -> tuple[int, float, str]:
    """Infer one heading level with confidence and textual reason."""
    if signals.numbering_depth > 0:
        level = min(signals.numbering_depth, max_depth)
        if signals.has_hash_marker and signals.hash_count == level:
            return level, 1.0, "numbering_depth_matches_hash"
        if signals.has_hash_marker and signals.hash_count != level:
            return level, 0.9, "numbering_depth_overrides_hash"
        return level, 0.85, "numbering_depth_without_hash"

    if signals.is_special_section:
        level = min(signals.special_section_level, max_depth)
        if signals.has_hash_marker and signals.hash_count == level:
            return level, 0.9, "special_section_matches_hash"
        if signals.has_hash_marker:
            adjusted = min(max(signals.hash_count, 1), max_depth)
            return adjusted, 0.65, "hash_overrides_special_section"
        return level, 0.75, "special_section_without_hash"

    if signals.has_hash_marker:
        level = min(max(signals.hash_count, 1), max_depth)
        consistency = context.check_hash_consistency()
        if consistency == "consistent":
            return level, 0.8, "hash_consistent"
        if consistency == "all_same":
            return level, 0.3, "hash_all_same"
        return level, 0.5, "hash_inconsistent"

    return 1, 0.2, "no_hash_no_numbering"


def infer_levels(signals_list: list[HeadingSignals], max_depth: int = 3) -> list[LevelInference]:
    context = DocumentContext(signals_list)
    results: list[LevelInference] = []
    for signals in signals_list:
        level, confidence, reason = infer_level(signals, context=context, max_depth=max_depth)
        results.append(
            LevelInference(
                signals=signals,
                inferred_level=level,
                confidence=confidence,
                reason=reason,
                source="rule",
            )
        )
    return results


def confidence_stats(
    inferences: list[LevelInference],
    threshold: float = LLM_CONFIDENCE_THRESHOLD,
) -> dict[str, int]:
    high = sum(1 for item in inferences if item.confidence >= 0.8)
    medium = sum(1 for item in inferences if threshold <= item.confidence < 0.8)
    low = sum(1 for item in inferences if item.confidence < threshold)
    return {
        "high": high,
        "medium": medium,
        "low": low,
        "total": len(inferences),
    }


def needs_llm_correction(
    inferences: list[LevelInference],
    threshold: float = LLM_CONFIDENCE_THRESHOLD,
) -> bool:
    if not inferences:
        return False

    low_confidence_count = sum(1 for item in inferences if item.confidence < threshold)
    low_ratio = low_confidence_count / len(inferences)
    if low_ratio > 0.3:
        return True

    if all(item.signals.hash_count == inferences[0].signals.hash_count for item in inferences):
        if not any(item.signals.numbering_depth > 0 for item in inferences):
            return True

    levels = [item.inferred_level for item in inferences]
    for index in range(1, len(levels)):
        if levels[index] - levels[index - 1] > 1:
            return True

    return False


def select_llm_mode(
    inferences: list[LevelInference],
    threshold: float = LLM_CONFIDENCE_THRESHOLD,
) -> str:
    if not inferences:
        return "partial"
    low_confidence_count = sum(1 for item in inferences if item.confidence < threshold)
    ratio = low_confidence_count / len(inferences)
    return "full" if ratio > 0.5 else "partial"
