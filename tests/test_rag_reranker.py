import unittest

from tree_rag.config import load_rag_config
from tree_rag.retrieval.reranker import rerank_chunks
from tree_rag.types import Chunk, RetrievedChunk


class _GoodClient:
    def rerank(self, **kwargs):  # noqa: ANN003
        return [0.1, 0.9]


class _BadClient:
    def rerank(self, **kwargs):  # noqa: ANN003
        raise RuntimeError("boom")


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="c1",
                text="first",
                source_node_id="n1",
                heading_path="A",
                embedding=[0.1],
            ),
            score=0.8,
            retrieval_detail={"fused_score": 0.8},
        ),
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="c2",
                text="second",
                source_node_id="n2",
                heading_path="B",
                embedding=[0.1],
            ),
            score=0.5,
            retrieval_detail={"fused_score": 0.5},
        ),
    ]


class RerankerTests(unittest.TestCase):
    def test_rerank_updates_scores(self) -> None:
        config = load_rag_config(load_dotenv=False)
        results = rerank_chunks(
            query="q",
            retrieved_chunks=_chunks(),
            config=config,
            client=_GoodClient(),
            mock=False,
            top_k=2,
        )
        self.assertEqual(results[0].chunk.chunk_id, "c2")
        self.assertIn("rerank_score", results[0].retrieval_detail)

    def test_rerank_failure_fallbacks_to_fused_order(self) -> None:
        config = load_rag_config(load_dotenv=False)
        with self.assertLogs("tree_rag.retrieval.reranker", level="WARNING"):
            results = rerank_chunks(
                query="q",
                retrieved_chunks=_chunks(),
                config=config,
                client=_BadClient(),
                mock=False,
                top_k=2,
            )
        self.assertEqual(results[0].chunk.chunk_id, "c1")


if __name__ == "__main__":
    unittest.main()
