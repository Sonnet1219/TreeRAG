"""CLI entrypoint for the Tree Builder component."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tree_builder.env import load_env
from tree_builder.parser import parse_markdown_with_preamble
from tree_builder.summary import MockSummarizer, build_llm_summarizer_from_env, generate_summaries
from tree_builder.tree import build_document_tree
from tree_builder.visualizer import export_document_tree_json, print_document_tree


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a document tree from a Markdown file.")
    parser.add_argument("input_markdown", type=Path, help="Path to markdown file.")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock", help="Summary generation mode.")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Reserved provider option for llm mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <input_stem>.tree.json.",
    )
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    load_env()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        markdown_text = args.input_markdown.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read markdown file: {exc}", file=sys.stderr)
        return 1

    sections, preamble = parse_markdown_with_preamble(markdown_text)
    tree = build_document_tree(
        doc_id=args.input_markdown.stem,
        sections=sections,
        root_content=preamble,
    )

    if args.mode == "mock":
        summarizer = MockSummarizer()
    else:
        try:
            summarizer = build_llm_summarizer_from_env(args.provider)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    try:
        generate_summaries(tree, summarizer)
    except (NotImplementedError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    print_document_tree(tree)

    output_path = args.output or args.input_markdown.with_suffix(".tree.json")
    try:
        export_document_tree_json(tree, output_path)
    except OSError as exc:
        print(f"Failed to write JSON output: {exc}", file=sys.stderr)
        return 1

    print(f"JSON exported to: {output_path}")
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
