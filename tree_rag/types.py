"""Core data models used by TreeRAG Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_node_id: str
    heading_path: str
    embedding: list[float]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    retrieval_detail: dict[str, float] = field(default_factory=dict)


@dataclass
class NodeLocateResult:
    node_id: str
    sub_query: str


@dataclass
class PipelineResult:
    query: str
    step1_thinking: str
    step1_nodes: list[NodeLocateResult]
    step2_retrieved: list[RetrievedChunk]
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "step1_thinking": self.step1_thinking,
            "step1_nodes": [
                {"node_id": item.node_id, "sub_query": item.sub_query}
                for item in self.step1_nodes
            ],
            "step2_retrieved": [
                {
                    "text": rc.chunk.text,
                    "heading_path": rc.chunk.heading_path,
                    "scores": rc.retrieval_detail,
                }
                for rc in self.step2_retrieved
            ],
            "answer": self.answer,
        }


@dataclass
class RagConfig:
    openai_api_key: str
    openai_base_url: str
    llm_model: str
    embed_model: str
    rerank_model: str
    timeout_seconds: float
    top_k: int
    dense_weight: float
    bm25_weight: float
    rerank_diversify: bool
    rerank_min_unique_nodes: int


@dataclass
class IndexedNode:
    node_id: str
    heading_path: str
    chunks: list[Chunk]
    bm25_index: Any


@dataclass
class RagIndex:
    doc_id: str
    tree_data: dict[str, Any]
    nodes: dict[str, IndexedNode]
    all_chunks: list[Chunk]

    @property
    def tree(self) -> dict[str, Any]:
        return self.tree_data.get("tree", {})
