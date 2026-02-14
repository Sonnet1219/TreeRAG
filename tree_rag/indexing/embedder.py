"""Embedding backends for indexing and retrieval."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from tree_rag.types import RagConfig
from tree_rag.utils.openai_client import OpenAICompatibleClient


LOGGER = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts."""


def _hash_embedding(text: str, dim: int = 64) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    cursor = digest
    while len(values) < dim:
        for byte in cursor:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) >= dim:
                break
        cursor = hashlib.sha256(cursor).digest()
    return values


class MockEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(text, dim=self.dim) for text in texts]


class OpenAIEmbedder:
    def __init__(self, client: OpenAICompatibleClient, model: str, batch_size: int = 10) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_number = (start // self.batch_size) + 1
            LOGGER.info(
                "Embedding batch %d/%d (size=%d) using model=%s",
                batch_number,
                total_batches,
                len(batch),
                self.model,
            )
            vectors.extend(self.client.embeddings(model=self.model, texts=batch))
        LOGGER.info("Embedding completed for %d texts.", len(texts))
        return vectors


def build_embedder(
    config: RagConfig,
    mock: bool,
    client: OpenAICompatibleClient | None = None,
) -> Embedder:
    if mock:
        return MockEmbedder()
    if client is None:
        raise ValueError("OpenAI client is required in non-mock embedding mode.")
    return OpenAIEmbedder(client=client, model=config.embed_model)
