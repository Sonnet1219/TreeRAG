"""Step 2.3: Cross-node reranking with fallback."""

from __future__ import annotations

import logging

from tree_rag.types import RagConfig, RetrievedChunk
from tree_rag.utils.openai_client import OpenAICompatibleClient


LOGGER = logging.getLogger(__name__)


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
    if mock or client is None:
        return sorted(retrieved_chunks, key=lambda item: item.score, reverse=True)[:resolved_top_k]

    documents = [item.chunk.text for item in retrieved_chunks]
    try:
        scores = client.rerank(
            model=config.rerank_model,
            query=query,
            documents=documents,
            top_n=resolved_top_k,
        )
        if len(scores) != len(retrieved_chunks):
            raise RuntimeError("Rerank score length mismatch.")
    except Exception as exc:
        LOGGER.warning("Rerank failed, fallback to fused score: %s", exc)
        return sorted(retrieved_chunks, key=lambda item: item.score, reverse=True)[:resolved_top_k]

    for item, score in zip(retrieved_chunks, scores):
        item.retrieval_detail["rerank_score"] = round(float(score), 4)
        item.score = float(score)

    return sorted(retrieved_chunks, key=lambda item: item.score, reverse=True)[:resolved_top_k]
