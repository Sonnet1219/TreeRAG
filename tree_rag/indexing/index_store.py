"""Build, serialize, and load retrieval indexes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from tree_builder.builder import build_document
from tree_builder.summary import MockSummarizer
from tree_builder.visualizer import document_tree_to_dict
from tree_rag.indexing.bm25_builder import build_bm25_index
from tree_rag.indexing.chunker import chunk_content
from tree_rag.indexing.embedder import build_embedder
from tree_rag.types import Chunk, IndexedNode, RagConfig, RagIndex
from tree_rag.utils.openai_client import OpenAICompatibleClient
from tree_rag.utils.tokenizer import tokenize


LOGGER = logging.getLogger(__name__)


def _iter_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    stack = [node]
    result: list[dict[str, Any]] = []
    while stack:
        current = stack.pop()
        result.append(current)
        children = current.get("children", [])
        if isinstance(children, list):
            stack.extend(reversed(children))
    return result


def _collect_leaf_nodes(tree_data: dict[str, Any]) -> list[dict[str, Any]]:
    root = tree_data.get("tree", {})
    leaves: list[dict[str, Any]] = []
    for node in _iter_nodes(root):
        children = node.get("children", [])
        is_leaf = node.get("is_leaf")
        if is_leaf is None:
            is_leaf = not bool(children)
        if is_leaf and node.get("node_id") != "root":
            leaves.append(node)
    return leaves


def load_tree_input(input_path: Path) -> dict[str, Any]:
    suffix = input_path.suffix.lower()
    if suffix == ".json" and input_path.name.endswith(".tree.json"):
        LOGGER.info("Reading prebuilt tree json: %s", input_path)
        return json.loads(input_path.read_text(encoding="utf-8"))

    if suffix == ".md":
        LOGGER.info("Reading markdown source: %s", input_path)
        markdown_text = input_path.read_text(encoding="utf-8")
        LOGGER.info("Parsing markdown and building robust tree...")
        tree, report = build_document(
            markdown_text=markdown_text,
            doc_id=input_path.stem,
            summarizer=MockSummarizer(),
            llm_client=None,
        )
        LOGGER.info(
            "Tree build report: llm_used=%s low_conf=%d preamble_injected=%d",
            report.llm_used,
            len(report.low_confidence_headings),
            report.preamble_injected,
        )
        return document_tree_to_dict(tree)

    raise ValueError("Input must be a .tree.json or .md file.")


def build_index_from_tree(
    tree_data: dict[str, Any],
    config: RagConfig,
    mock: bool,
    client: OpenAICompatibleClient | None = None,
) -> RagIndex:
    embedder = build_embedder(config=config, mock=mock, client=client)
    leaves = _collect_leaf_nodes(tree_data)
    LOGGER.info("Index build started. leaf_nodes=%d", len(leaves))
    nodes: dict[str, IndexedNode] = {}
    all_chunks: list[Chunk] = []

    for position, leaf in enumerate(leaves, start=1):
        node_id = str(leaf.get("node_id", "")).strip()
        heading_path = str(leaf.get("heading_path", leaf.get("heading", node_id)))
        content = str(leaf.get("content", ""))
        chunk_texts = chunk_content(content)
        LOGGER.info(
            "Indexing leaf %d/%d: node_id=%s chunks=%d",
            position,
            len(leaves),
            node_id,
            len(chunk_texts),
        )

        chunks: list[Chunk] = []
        if chunk_texts:
            vectors = embedder.embed_texts(chunk_texts)
            for idx, (text, vector) in enumerate(zip(chunk_texts, vectors)):
                chunk = Chunk(
                    chunk_id=f"{node_id}_chunk_{idx:02d}",
                    text=text,
                    source_node_id=node_id,
                    heading_path=heading_path,
                    embedding=[float(x) for x in vector],
                )
                chunks.append(chunk)
                all_chunks.append(chunk)

        tokenized = [tokenize(chunk.text) for chunk in chunks] if chunks else [[]]
        bm25_index = build_bm25_index(tokenized)

        nodes[node_id] = IndexedNode(
            node_id=node_id,
            heading_path=heading_path,
            chunks=chunks,
            bm25_index=bm25_index,
        )

    LOGGER.info(
        "Index build completed. doc_id=%s nodes=%d chunks=%d",
        tree_data.get("doc_id", "unknown_doc"),
        len(nodes),
        len(all_chunks),
    )
    return RagIndex(
        doc_id=str(tree_data.get("doc_id", "unknown_doc")),
        tree_data=tree_data,
        nodes=nodes,
        all_chunks=all_chunks,
    )


def save_index(index: RagIndex, output_dir: Path) -> None:
    LOGGER.info("Saving index files to directory: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_chunk_ids = {
        node_id: [chunk.chunk_id for chunk in node.chunks]
        for node_id, node in index.nodes.items()
    }
    node_heading_paths = {
        node_id: node.heading_path
        for node_id, node in index.nodes.items()
    }

    metadata = {
        "doc_id": index.doc_id,
        "node_count": len(index.nodes),
        "chunk_count": len(index.all_chunks),
        "node_chunk_ids": node_chunk_ids,
        "node_heading_paths": node_heading_paths,
        "tree_data": index.tree_data,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in index.all_chunks:
            row = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_node_id": chunk.source_node_id,
                "heading_path": chunk.heading_path,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if index.all_chunks:
        matrix = np.array([chunk.embedding for chunk in index.all_chunks], dtype=float)
    else:
        matrix = np.empty((0, 0), dtype=float)
    np.save(output_dir / "embeddings.npy", matrix)

    bm25_map = {node_id: node.bm25_index for node_id, node in index.nodes.items()}
    with (output_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(bm25_map, handle)
    LOGGER.info("Index files saved: metadata.json, chunks.jsonl, embeddings.npy, bm25.pkl")


def load_index(index_dir: Path) -> RagIndex:
    LOGGER.info("Loading index files from directory: %s", index_dir)
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    tree_data = metadata["tree_data"]
    node_chunk_ids = metadata["node_chunk_ids"]
    node_heading_paths = metadata["node_heading_paths"]

    chunks: list[Chunk] = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    source_node_id=row["source_node_id"],
                    heading_path=row["heading_path"],
                    embedding=[],
                )
            )

    matrix = np.load(index_dir / "embeddings.npy")
    for idx, chunk in enumerate(chunks):
        if idx < len(matrix):
            chunk.embedding = [float(x) for x in matrix[idx].tolist()]
        else:
            chunk.embedding = []

    with (index_dir / "bm25.pkl").open("rb") as handle:
        bm25_map = pickle.load(handle)

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    nodes: dict[str, IndexedNode] = {}
    for node_id, chunk_ids in node_chunk_ids.items():
        node_chunks = [chunk_by_id[cid] for cid in chunk_ids if cid in chunk_by_id]
        nodes[node_id] = IndexedNode(
            node_id=node_id,
            heading_path=node_heading_paths.get(node_id, node_id),
            chunks=node_chunks,
            bm25_index=bm25_map.get(node_id),
        )

    index = RagIndex(
        doc_id=metadata["doc_id"],
        tree_data=tree_data,
        nodes=nodes,
        all_chunks=chunks,
    )
    LOGGER.info(
        "Index loaded successfully. doc_id=%s nodes=%d chunks=%d",
        index.doc_id,
        len(index.nodes),
        len(index.all_chunks),
    )
    return index
