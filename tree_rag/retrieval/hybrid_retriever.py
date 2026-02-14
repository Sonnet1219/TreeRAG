"""Step 2: Dense + BM25 hybrid retrieval within a leaf node."""

from __future__ import annotations

from tree_rag.types import IndexedNode, RetrievedChunk
from tree_rag.utils.similarity import cosine_similarity, min_max_normalize
from tree_rag.utils.tokenizer import tokenize


def hybrid_retrieve(
    node: IndexedNode,
    query: str,
    top_k: int,
    dense_weight: float,
    bm25_weight: float,
    embed_query_fn,
) -> list[RetrievedChunk]:
    chunks = node.chunks
    if not chunks:
        return []

    query_embedding = embed_query_fn(query)
    dense_scores = [
        cosine_similarity(query_embedding, chunk.embedding)
        for chunk in chunks
    ]

    tokenized_query = tokenize(query)
    if node.bm25_index is not None:
        bm25_scores = list(node.bm25_index.get_scores(tokenized_query))
    else:
        bm25_scores = [0.0 for _ in chunks]

    if len(bm25_scores) < len(chunks):
        bm25_scores.extend([0.0] * (len(chunks) - len(bm25_scores)))
    bm25_scores = bm25_scores[: len(chunks)]

    dense_norm = min_max_normalize(dense_scores)
    bm25_norm = min_max_normalize([float(v) for v in bm25_scores])
    fused_scores = [
        dense_weight * dense + bm25_weight * bm25
        for dense, bm25 in zip(dense_norm, bm25_norm)
    ]

    ranked = sorted(
        zip(chunks, fused_scores, dense_scores, bm25_scores),
        key=lambda item: item[1],
        reverse=True,
    )
    resolved_top_k = max(1, top_k)

    return [
        RetrievedChunk(
            chunk=chunk,
            score=float(fused),
            retrieval_detail={
                "dense_score": round(float(dense), 4),
                "bm25_score": round(float(bm25), 4),
                "fused_score": round(float(fused), 4),
            },
        )
        for chunk, fused, dense, bm25 in ranked[:resolved_top_k]
    ]
