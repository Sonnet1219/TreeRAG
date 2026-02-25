"""CLI entrypoint for the Tree Builder component."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tree_builder.builder import BuildReport, build_document
from tree_builder.env import load_env
from tree_builder.llm_corrector import build_openai_tree_llm_client_from_env
from tree_builder.summary import MockSummarizer, build_llm_summarizer_from_env
from tree_builder.visualizer import export_document_tree_json, print_document_tree


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a document tree from a Markdown file.")
    parser.add_argument("input_markdown", type=Path, help="Path to markdown file.")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock", help="Summary generation mode.")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider option for llm mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <input_stem>.tree.json.",
    )
    return parser


def _print_build_report(report: BuildReport) -> None:
    stats = report.confidence_stats or {"high": 0, "medium": 0, "low": 0, "total": 0}
    print(
        "Build report: "
        f"headings={stats.get('total', 0)}, "
        f"high={stats.get('high', 0)}, "
        f"medium={stats.get('medium', 0)}, "
        f"low={stats.get('low', 0)}, "
        f"llm_used={report.llm_used}, "
        f"llm_mode={report.llm_mode}, "
        f"preamble_injected={report.preamble_injected}"
    )

    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for fix in report.fixes:
        print(f"FIX: {fix}")


def run_cli(argv: list[str] | None = None) -> int:
    load_env()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        markdown_text = args.input_markdown.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read markdown file: {exc}", file=sys.stderr)
        return 1

    llm_client = None
    llm_model: str | None = None

    if args.mode == "mock":
        summarizer = MockSummarizer()
    else:
        if args.provider != "openai":
            print("Only provider 'openai' is supported for robust tree llm mode.", file=sys.stderr)
            return 2

        try:
            summarizer = build_llm_summarizer_from_env(args.provider)
            llm_client, llm_model = build_openai_tree_llm_client_from_env()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    try:
        tree, report = build_document(
            markdown_text=markdown_text,
            doc_id=args.input_markdown.stem,
            summarizer=summarizer,
            llm_client=llm_client,
            llm_model=llm_model,
        )
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    _print_build_report(report)
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
