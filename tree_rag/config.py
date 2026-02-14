"""Configuration loader for TreeRAG Phase 2."""

from __future__ import annotations

import os

from tree_builder.env import load_env
from tree_rag.types import RagConfig


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_rag_config(load_dotenv: bool = True) -> RagConfig:
    if load_dotenv:
        load_env()

    return RagConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        llm_model=os.getenv("RAG_LLM_MODEL", "qwen-plus").strip(),
        embed_model=os.getenv("RAG_EMBED_MODEL", "text-embedding-v3").strip(),
        rerank_model=os.getenv("RAG_RERANK_MODEL", "gte-rerank-v2").strip(),
        timeout_seconds=_get_float(
            "RAG_TIMEOUT_SECONDS",
            _get_float("OPENAI_TIMEOUT_SECONDS", 30.0),
        ),
        top_k=max(1, _get_int("RAG_TOP_K", 5)),
        dense_weight=_get_float("RAG_DENSE_WEIGHT", 0.5),
        bm25_weight=_get_float("RAG_BM25_WEIGHT", 0.5),
    )
