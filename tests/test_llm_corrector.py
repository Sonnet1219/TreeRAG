import unittest

from tree_builder.llm_corrector import (
    llm_infer_partial,
    merge_llm_corrections,
)
from tree_builder.preprocessor import RawHeading
from tree_builder.rule_engine import LevelInference, needs_llm_correction, select_llm_mode
from tree_builder.signals import HeadingSignals


class _FakeLLMClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def chat_completion(self, **kwargs):  # noqa: ANN003
        return self.payload


def _make_signals(index: int) -> HeadingSignals:
    return HeadingSignals(
        index=index,
        line_index=index,
        raw_text=f"Heading {index}",
        hash_count=1,
        has_hash_marker=True,
        numbering=None,
        numbering_type=None,
        numbering_depth=0,
        is_special_section=False,
        special_section_level=1,
        text_length=8,
        heading_text=f"Heading {index}",
    )


def _make_inference(index: int, level: int, confidence: float) -> LevelInference:
    return LevelInference(
        signals=_make_signals(index),
        inferred_level=level,
        confidence=confidence,
        reason="test",
        source="rule",
    )


class LLMCorrectorTests(unittest.TestCase):
    def test_trigger_and_mode_selection(self) -> None:
        mostly_low = [
            _make_inference(0, 1, 0.2),
            _make_inference(1, 1, 0.3),
            _make_inference(2, 2, 0.4),
            _make_inference(3, 1, 0.9),
        ]
        self.assertTrue(needs_llm_correction(mostly_low))
        self.assertEqual(select_llm_mode(mostly_low), "full")

        few_low = [
            _make_inference(0, 1, 0.9),
            _make_inference(1, 2, 0.95),
            _make_inference(2, 2, 0.4),
        ]
        self.assertEqual(select_llm_mode(few_low), "partial")

    def test_merge_strategy_prefers_rule_for_high_conflict(self) -> None:
        rule_results = [
            _make_inference(0, 1, 0.9),
            _make_inference(1, 1, 0.3),
        ]
        merged = merge_llm_corrections(
            rule_results=rule_results,
            llm_results={
                0: type("S", (), {"index": 0, "level": 3, "reasoning": "conflict"})(),
                1: type("S", (), {"index": 1, "level": 2, "reasoning": "better"})(),
            },
        )

        self.assertEqual(merged[0].inferred_level, 1)
        self.assertEqual(merged[0].source, "rule")
        self.assertEqual(merged[1].inferred_level, 2)
        self.assertEqual(merged[1].source, "llm")
        self.assertEqual(merged[1].confidence, 0.85)

    def test_invalid_llm_response_raises_runtime_error(self) -> None:
        raw_headings = [RawHeading(0, 1, "Intro", True, "# Intro")]
        rule_results = [_make_inference(0, 1, 0.2)]
        with self.assertRaises(RuntimeError):
            llm_infer_partial(
                raw_headings=raw_headings,
                rule_results=rule_results,
                llm_client=_FakeLLMClient("not-json"),
                model="gpt-test",
            )

    def test_partial_llm_response_parses_and_returns_levels(self) -> None:
        raw_headings = [RawHeading(0, 1, "Intro", True, "# Intro")]
        rule_results = [_make_inference(0, 1, 0.2)]
        suggestions = llm_infer_partial(
            raw_headings=raw_headings,
            rule_results=rule_results,
            llm_client=_FakeLLMClient('{"results": [{"index": 0, "level": 2, "reasoning": "nested"}]}'),
            model="gpt-test",
        )
        self.assertIn(0, suggestions)
        self.assertEqual(suggestions[0].level, 2)


if __name__ == "__main__":
    unittest.main()
