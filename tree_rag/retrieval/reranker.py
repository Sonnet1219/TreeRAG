"""Step 2.3: Cross-node reranking with fallback."""

from __future__ import annotations

import math
import logging
from typing import Iterable

from tree_rag.types import RagConfig, RetrievedChunk
from tree_rag.utils.openai_client import OpenAICompatibleClient


LOGGER = logging.getLogger(__name__)


def _fused_score(item: RetrievedChunk) -> float:
    return float(item.retrieval_detail.get("fused_score", 0.0))


def _sorted_chunks(chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(
        chunks,
        key=lambda item: (
            -float(item.score),
            -_fused_score(item),
            item.chunk.chunk_id,
        ),
    )


def _node_count_map(chunks: Iterable[RetrievedChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in chunks:
        node_id = item.chunk.source_node_id
        counts[node_id] = counts.get(node_id, 0) + 1
    return counts


def _score_candidates(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    config: RagConfig,
    client: OpenAICompatibleClient | None,
    mock: bool,
) -> list[RetrievedChunk]:
    if mock or client is None:
        return retrieved_chunks

    documents = [item.chunk.text for item in retrieved_chunks]
    try:
        scores = client.rerank(
            model=config.rerank_model,
            query=query,
            documents=documents,
            top_n=None,
        )
        if len(scores) != len(retrieved_chunks):
            raise RuntimeError("Rerank score length mismatch.")
    except Exception as exc:
        LOGGER.warning("Rerank failed, fallback to fused score: %s", exc)
        return retrieved_chunks

    for item, score in zip(retrieved_chunks, scores):
        item.retrieval_detail["rerank_score"] = round(float(score), 4)
        item.score = float(score)
    return retrieved_chunks


def _select_with_node_diversity(
    scored_chunks: list[RetrievedChunk],
    config: RagConfig,
    resolved_top_k: int,
) -> list[RetrievedChunk]:
    if not scored_chunks:
        return []

    ranked = _sorted_chunks(scored_chunks)
    k = min(resolved_top_k, len(ranked))
    available_unique_nodes = len({item.chunk.source_node_id for item in ranked})

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Rerank candidate node distribution: %s", _node_count_map(ranked))

    if config.rerank_diversify:
        if config.rerank_min_unique_nodes > 0:
            target_unique_nodes = min(
                available_unique_nodes,
                k,
                config.rerank_min_unique_nodes,
            )
        else:
            target_unique_nodes = min(available_unique_nodes, k, math.ceil(k / 2))
    else:
        target_unique_nodes = 0

    selected: list[RetrievedChunk] = []
    selected_ids: set[int] = set()
    seeded_nodes: set[str] = set()

    if config.rerank_diversify and target_unique_nodes > 0:
        for item in ranked:
            node_id = item.chunk.source_node_id
            if node_id in seeded_nodes:
                continue
            selected.append(item)
            selected_ids.add(id(item))
            seeded_nodes.add(node_id)
            if len(seeded_nodes) >= target_unique_nodes or len(selected) >= k:
                break

    for item in ranked:
        if len(selected) >= k:
            break
        if id(item) in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(id(item))

    selected = _sorted_chunks(selected)
    selected_unique_nodes = len({item.chunk.source_node_id for item in selected})

    LOGGER.info(
        "Rerank selection: candidates=%d unique_nodes=%d top_k=%d diversity=%s target_unique=%d selected_unique=%d",
        len(scored_chunks),
        available_unique_nodes,
        k,
        config.rerank_diversify,
        target_unique_nodes,
        selected_unique_nodes,
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Rerank selected node distribution: %s", _node_count_map(selected))

    return selected


def rerank_chunks(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    config: RagConfig,
    client: OpenAICompatibleClient | None,
    mock: bool,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    if not retrieved_chunks:
        return []

    resolved_top_k = max(1, top_k or config.top_k)
    scored_chunks = _score_candidates(
        query=query,
        retrieved_chunks=retrieved_chunks,
        config=config,
        client=client,
        mock=mock,
    )
    return _select_with_node_diversity(
        scored_chunks=scored_chunks,
        config=config,
        resolved_top_k=resolved_top_k,
    )
