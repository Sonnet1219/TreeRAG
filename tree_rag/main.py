"""CLI entrypoint for TreeRAG Phase 2."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

from tree_rag.config import load_rag_config
from tree_rag.indexing.index_store import build_index_from_tree, load_index, load_tree_input, save_index
from tree_rag.pipeline import run_pipeline
from tree_rag.types import PipelineResult
from tree_rag.utils.openai_client import OpenAICompatibleClient


LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TreeRAG Phase 2 CLI")
    parser.add_argument(
        "--log-level",
        default=os.getenv("RAG_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for progress output.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="Build index from .tree.json or .md")
    index_parser.add_argument("--input", type=Path, required=True, help="Input .tree.json or .md file")
    index_parser.add_argument("--output", type=Path, required=True, help="Output index directory")
    index_parser.add_argument("--mock", action="store_true", help="Use mock mode")

    query_parser = sub.add_parser("query", help="Run one query on an existing index")
    query_parser.add_argument("--index", type=Path, required=True, help="Index directory")
    query_parser.add_argument("--query", type=str, required=True, help="User query")
    query_parser.add_argument("--mock", action="store_true", help="Use mock mode")

    interactive_parser = sub.add_parser("interactive", help="Interactive query loop")
    interactive_parser.add_argument("--index", type=Path, required=True, help="Index directory")
    interactive_parser.add_argument("--mock", action="store_true", help="Use mock mode")
    return parser


def _make_client(mock: bool):
    config = load_rag_config(load_dotenv=True)
    if mock:
        return config, None
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when --mock is not set.")
    client = OpenAICompatibleClient(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        timeout_seconds=config.timeout_seconds,
        max_retries=max(1, int(os.getenv("RAG_HTTP_MAX_RETRIES", "3"))),
        retry_backoff_seconds=max(0.0, float(os.getenv("RAG_HTTP_BACKOFF_SECONDS", "1.0"))),
    )
    return config, client


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _print_pipeline_result(result: PipelineResult) -> None:
    print("=" * 60)
    print(f"Query: {result.query}")
    print("=" * 60)
    print("\n>>> Step 1: Node Locating")
    print(f"Thinking: {result.step1_thinking}")
    print(f"Located {len(result.step1_nodes)} leaf nodes:")
    for item in result.step1_nodes:
        print(f"  [{item.node_id}] sub_query: {item.sub_query}")

    print("\n>>> Step 2: Hybrid Retrieval + Rerank")
    print(f"Retrieved {len(result.step2_retrieved)} chunks:")
    for idx, chunk in enumerate(result.step2_retrieved, start=1):
        detail = chunk.retrieval_detail
        dense = detail.get("dense_score", 0.0)
        bm25 = detail.get("bm25_score", 0.0)
        fused = detail.get("fused_score", 0.0)
        rerank = detail.get("rerank_score")
        rerank_text = f"  rerank={rerank}" if rerank is not None else ""
        print(
            f"  #{idx} [{chunk.chunk.source_node_id}] {chunk.chunk.heading_path}\n"
            f"     {chunk.chunk.text[:120]}\n"
            f"     dense={dense}  bm25={bm25}  fused={fused}{rerank_text}"
        )

    print("\n>>> Step 3: Answer")
    print(result.answer)
    print("=" * 60)


def run_cli(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    LOGGER.info("Starting command '%s' (mock=%s)", args.command, args.mock)

    try:
        config, client = _make_client(mock=args.mock)
        LOGGER.info(
            "Runtime config loaded. base_url=%s llm_model=%s embed_model=%s rerank_model=%s timeout=%.1fs",
            config.openai_base_url,
            config.llm_model,
            config.embed_model,
            config.rerank_model,
            config.timeout_seconds,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "index":
        try:
            LOGGER.info("Loading input file: %s", args.input)
            tree_data = load_tree_input(args.input)
            LOGGER.info("Input parsed. doc_id=%s", tree_data.get("doc_id", "unknown_doc"))
            LOGGER.info("Building retrieval index...")
            index = build_index_from_tree(tree_data=tree_data, config=config, mock=args.mock, client=client)
            LOGGER.info("Persisting index to: %s", args.output)
            save_index(index=index, output_dir=args.output)
            LOGGER.info("Index persisted successfully.")
        except Exception as exc:
            print(f"Failed to build index: {exc}", file=sys.stderr)
            return 1

        print(f"Index built: {args.output}")
        print(
            f"doc_id={index.doc_id}, nodes={len(index.nodes)}, chunks={len(index.all_chunks)}"
        )
        return 0

    if args.command == "query":
        try:
            LOGGER.info("Loading index from: %s", args.index)
            index = load_index(args.index)
            LOGGER.info(
                "Index loaded. doc_id=%s nodes=%d chunks=%d",
                index.doc_id,
                len(index.nodes),
                len(index.all_chunks),
            )
            LOGGER.info("Running retrieval pipeline for one query...")
            result = run_pipeline(
                query=args.query,
                index=index,
                config=config,
                mock=args.mock,
                client=client,
            )
            LOGGER.info("Pipeline finished successfully.")
        except Exception as exc:
            print(f"Failed to run query: {exc}", file=sys.stderr)
            return 1

        _print_pipeline_result(result)
        return 0

    # interactive
    try:
        LOGGER.info("Loading index from: %s", args.index)
        index = load_index(args.index)
        LOGGER.info(
            "Index loaded. doc_id=%s nodes=%d chunks=%d",
            index.doc_id,
            len(index.nodes),
            len(index.all_chunks),
        )
    except Exception as exc:
        print(f"Failed to load index: {exc}", file=sys.stderr)
        return 1

    print("Interactive mode. Enter empty line or 'exit' to quit.")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"exit", "quit"}:
            break
        LOGGER.info("Running retrieval pipeline for interactive query.")
        result = run_pipeline(
            query=query,
            index=index,
            config=config,
            mock=args.mock,
            client=client,
        )
        LOGGER.info("Interactive query completed.")
        _print_pipeline_result(result)

    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
