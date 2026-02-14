"""BM25 builder with optional dependency on rank_bm25."""

from __future__ import annotations

import math
from typing import Protocol


class BM25Like(Protocol):
    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Return BM25 scores aligned with source documents."""


class FallbackBM25:
    """Minimal BM25 implementation used when rank_bm25 is unavailable."""

    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_docs)
        self.avgdl = (
            sum(len(doc) for doc in tokenized_docs) / self.doc_count
            if self.doc_count
            else 0.0
        )
        self.doc_freq: dict[str, int] = {}
        for doc in tokenized_docs:
            for token in set(doc):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.term_freq: list[dict[str, int]] = []
        for doc in tokenized_docs:
            tf: dict[str, int] = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            self.term_freq.append(tf)

    def _idf(self, token: str) -> float:
        n_q = self.doc_freq.get(token, 0)
        if n_q == 0:
            return 0.0
        numerator = self.doc_count - n_q + 0.5
        denominator = n_q + 0.5
        return math.log(1.0 + numerator / denominator)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        if not self.tokenized_docs:
            return []
        scores = [0.0 for _ in self.tokenized_docs]
        for idx, doc_tokens in enumerate(self.tokenized_docs):
            dl = len(doc_tokens)
            tf = self.term_freq[idx]
            score = 0.0
            for token in query_tokens:
                f = tf.get(token, 0)
                if f == 0:
                    continue
                idf = self._idf(token)
                denominator = f + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                score += idf * ((f * (self.k1 + 1.0)) / denominator)
            scores[idx] = score
        return scores


def build_bm25_index(tokenized_chunks: list[list[str]]) -> BM25Like:
    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        return BM25Okapi(tokenized_chunks)
    except ImportError:
        return FallbackBM25(tokenized_chunks)
