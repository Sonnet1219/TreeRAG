"""Tree Builder package."""

from tree_builder.builder import BuildReport, build_document, build_robust_tree
from tree_builder.parser import HeadingInfo, Section, parse_heading_line, parse_markdown_file, parse_markdown_sections
from tree_builder.preamble import generate_preamble_summaries, inject_preamble_leaves
from tree_builder.summary import (
    LLMSummarizerStub,
    MockSummarizer,
    OpenAICompatibleSummarizer,
    Summarizer,
    build_llm_summarizer_from_env,
    generate_summaries,
)
from tree_builder.tree import (
    DocumentTree,
    TreeNode,
    build_document_tree,
    postorder_nodes,
    traverse_all_nodes,
    validate_and_fix_tree,
)
from tree_builder.visualizer import document_tree_to_dict, export_document_tree_json, print_document_tree

__all__ = [
    "BuildReport",
    "DocumentTree",
    "HeadingInfo",
    "LLMSummarizerStub",
    "MockSummarizer",
    "OpenAICompatibleSummarizer",
    "Section",
    "Summarizer",
    "TreeNode",
    "build_document",
    "build_document_tree",
    "build_llm_summarizer_from_env",
    "build_robust_tree",
    "document_tree_to_dict",
    "export_document_tree_json",
    "generate_preamble_summaries",
    "generate_summaries",
    "inject_preamble_leaves",
    "parse_heading_line",
    "parse_markdown_file",
    "parse_markdown_sections",
    "postorder_nodes",
    "print_document_tree",
    "traverse_all_nodes",
    "validate_and_fix_tree",
]
