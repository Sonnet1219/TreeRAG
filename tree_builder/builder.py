"""High-level robust tree building pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any

from tree_builder.llm_corrector import (
    TreeStructureLLMClient,
    llm_infer_full_structure,
    llm_infer_partial,
    merge_llm_corrections,
)
from tree_builder.parser import HeadingInfo, Section
from tree_builder.preamble import generate_preamble_summaries, inject_preamble_leaves
from tree_builder.preprocessor import RawHeading, extract_raw_headings
from tree_builder.rule_engine import (
    LLM_CONFIDENCE_THRESHOLD,
    confidence_stats,
    infer_levels,
    needs_llm_correction,
    select_llm_mode,
)
from tree_builder.signals import extract_all_signals
from tree_builder.summary import Summarizer, generate_summaries
from tree_builder.tree import DocumentTree, build_document_tree, validate_and_fix_tree


MAX_DEPTH = 3


@dataclass
class BuildReport:
    warnings: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_mode: str = "none"
    confidence_stats: dict[str, int] = field(default_factory=dict)
    low_confidence_headings: list[dict[str, Any]] = field(default_factory=list)
    preamble_injected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_root_preamble(lines: list[str], raw_headings: list[RawHeading]) -> str:
    if not raw_headings:
        return "\n".join(lines).strip()
    first_line = raw_headings[0].line_index
    return "\n".join(lines[:first_line]).strip()


def _build_sections(
    lines: list[str],
    raw_headings: list[RawHeading],
    inferences,
) -> list[Section]:
    if not raw_headings:
        return []

    sections: list[Section] = []
    for index, heading in enumerate(raw_headings):
        start = heading.line_index + 1
        end = raw_headings[index + 1].line_index if index + 1 < len(raw_headings) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        inference = inferences[index]
        section_heading = HeadingInfo(
            hash_count=heading.hash_count,
            numbering=inference.signals.numbering,
            clean_title=inference.signals.heading_text,
            inferred_level=inference.inferred_level,
            heading_raw=heading.raw_text,
        )
        sections.append(
            Section(
                heading=section_heading,
                content=content,
                index=index + 1,
            )
        )
    return sections


def build_robust_tree(
    markdown_text: str,
    doc_id: str,
    llm_client: TreeStructureLLMClient | None = None,
    llm_model: str | None = None,
    max_depth: int = MAX_DEPTH,
) -> tuple[DocumentTree, BuildReport]:
    """Build a document tree using preprocess -> rule engine -> optional LLM correction."""
    lines = markdown_text.splitlines()
    report = BuildReport()

    raw_headings, _ = extract_raw_headings(lines)
    if not raw_headings:
        tree = build_document_tree(doc_id=doc_id, sections=[], root_content=markdown_text)
        report.warnings.append("No headings detected; tree contains only root content.")
        report.confidence_stats = {"high": 0, "medium": 0, "low": 0, "total": 0}
        return tree, report

    signals_list = extract_all_signals(raw_headings)
    rule_results = infer_levels(signals_list, max_depth=max_depth)
    report.confidence_stats = confidence_stats(rule_results)
    report.low_confidence_headings = [
        {
            "index": result.signals.index,
            "heading": raw_headings[result.signals.index].raw_text,
            "level": result.inferred_level,
            "confidence": round(result.confidence, 3),
            "reason": result.reason,
        }
        for result in rule_results
        if result.confidence < LLM_CONFIDENCE_THRESHOLD
    ]

    final_results = rule_results
    should_use_llm = llm_client is not None and needs_llm_correction(rule_results)
    if should_use_llm:
        mode = select_llm_mode(rule_results)
        model = llm_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        report.llm_used = True
        report.llm_mode = mode

        if mode == "full":
            llm_levels = llm_infer_full_structure(
                raw_headings=raw_headings,
                rule_results=rule_results,
                llm_client=llm_client,
                model=model,
                max_depth=max_depth,
            )
        else:
            llm_levels = llm_infer_partial(
                raw_headings=raw_headings,
                rule_results=rule_results,
                llm_client=llm_client,
                model=model,
                threshold=LLM_CONFIDENCE_THRESHOLD,
                max_depth=max_depth,
            )

        final_results = merge_llm_corrections(
            rule_results=rule_results,
            llm_results=llm_levels,
            threshold=LLM_CONFIDENCE_THRESHOLD,
        )
    elif llm_client is None and needs_llm_correction(rule_results):
        report.warnings.append(
            "Low-confidence headings detected but LLM correction is disabled."
        )

    root_content = _extract_root_preamble(lines, raw_headings)
    sections = _build_sections(lines, raw_headings, final_results)
    tree = build_document_tree(doc_id=doc_id, sections=sections, root_content=root_content)

    report.fixes = validate_and_fix_tree(tree.root, max_depth=max_depth)
    tree.recompute_counts()
    return tree, report


def build_document(
    markdown_text: str,
    doc_id: str,
    summarizer: Summarizer,
    llm_client: TreeStructureLLMClient | None = None,
    llm_model: str | None = None,
    max_depth: int = MAX_DEPTH,
) -> tuple[DocumentTree, BuildReport]:
    """Build robust tree, summarize, inject preamble nodes, and summarize preambles."""
    tree, report = build_robust_tree(
        markdown_text=markdown_text,
        doc_id=doc_id,
        llm_client=llm_client,
        llm_model=llm_model,
        max_depth=max_depth,
    )

    generate_summaries(tree, summarizer)
    injected_count = inject_preamble_leaves(tree.root)
    report.preamble_injected = injected_count
    if injected_count:
        generate_preamble_summaries(tree.root, summarizer)

    tree.recompute_counts()
    return tree, report
