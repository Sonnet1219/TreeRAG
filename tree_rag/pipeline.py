"""End-to-end fixed three-step TreeRAG pipeline."""

from __future__ import annotations

import logging

from tree_rag.indexing.embedder import build_embedder
from tree_rag.retrieval.hybrid_retriever import hybrid_retrieve
from tree_rag.retrieval.node_locator import locate_nodes
from tree_rag.retrieval.reranker import rerank_chunks
from tree_rag.retrieval.synthesizer import synthesize
from tree_rag.types import PipelineResult, RagConfig, RagIndex
from tree_rag.utils.openai_client import OpenAICompatibleClient


LOGGER = logging.getLogger(__name__)


def run_pipeline(
    query: str,
    index: RagIndex,
    config: RagConfig,
    mock: bool,
    client: OpenAICompatibleClient | None = None,
) -> PipelineResult:
    LOGGER.info("Step 1/3 Node Locating started.")
    located_nodes, thinking = locate_nodes(
        query=query,
        tree_data=index.tree_data,
        config=config,
        client=client,
        mock=mock,
        top_k=config.top_k,
    )
    LOGGER.info("Step 1/3 completed. Located %d nodes.", len(located_nodes))
    LOGGER.info("Step 1/3 thinking: %s", thinking)

    LOGGER.info("Step 2/3 Hybrid Retrieval started.")
    embedder = build_embedder(config=config, mock=mock, client=client)

    def _embed_query(text: str) -> list[float]:
        vectors = embedder.embed_texts([text])
        if not vectors:
            return []
        return vectors[0]

    all_retrieved = []
    for node_info in located_nodes:
        node = index.nodes.get(node_info.node_id)
        if node is None:
            LOGGER.warning("Located node '%s' is missing in index, skipping.", node_info.node_id)
            continue
        LOGGER.info(
            "Retrieving chunks from node %s (%d chunks in node).",
            node_info.node_id,
            len(node.chunks),
        )
        all_retrieved.extend(
            hybrid_retrieve(
                node=node,
                query=node_info.sub_query,
                top_k=config.top_k,
                dense_weight=config.dense_weight,
                bm25_weight=config.bm25_weight,
                embed_query_fn=_embed_query,
            )
        )

    LOGGER.info("Step 2/3 retrieval completed. Candidate chunks=%d", len(all_retrieved))
    LOGGER.info("Step 2/3 rerank started.")
    reranked = rerank_chunks(
        query=query,
        retrieved_chunks=all_retrieved,
        config=config,
        client=client,
        mock=mock,
        top_k=config.top_k,
    )
    LOGGER.info("Step 2/3 rerank completed. Top chunks=%d", len(reranked))

    LOGGER.info("Step 3/3 Synthesis started.")
    answer = synthesize(
        query=query,
        retrieved_chunks=reranked,
        config=config,
        client=client,
        mock=mock,
    )
    LOGGER.info("Step 3/3 completed.")

    return PipelineResult(
        query=query,
        step1_thinking=thinking,
        step1_nodes=located_nodes,
        step2_retrieved=reranked,
        answer=answer,
    )
