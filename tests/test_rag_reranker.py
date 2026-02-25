import unittest

from tree_rag.config import load_rag_config
from tree_rag.retrieval.reranker import rerank_chunks
from tree_rag.types import Chunk, RetrievedChunk


class _StaticScoreClient:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.last_kwargs: dict | None = None

    def rerank(self, **kwargs):  # noqa: ANN003
        self.last_kwargs = kwargs
        return list(self.scores)


class _BadClient:
    def rerank(self, **kwargs):  # noqa: ANN003
        raise RuntimeError("boom")


def _config():
    config = load_rag_config(load_dotenv=False)
    config.rerank_diversify = True
    config.rerank_min_unique_nodes = 0
    return config


def _retrieved(chunk_id: str, node_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=f"text {chunk_id}",
            source_node_id=node_id,
            heading_path=f"Heading {node_id}",
            embedding=[0.1],
        ),
        score=score,
        retrieval_detail={"fused_score": score},
    )


def _two_chunks() -> list[RetrievedChunk]:
    return [
        _retrieved("c1", "n1", 0.8),
        _retrieved("c2", "n2", 0.5),
    ]


def _dominating_chunks() -> list[RetrievedChunk]:
    return [
        _retrieved("n1_c1", "n1", 0.99),
        _retrieved("n1_c2", "n1", 0.97),
        _retrieved("n1_c3", "n1", 0.96),
        _retrieved("n1_c4", "n1", 0.95),
        _retrieved("n1_c5", "n1", 0.94),
        _retrieved("n2_c1", "n2", 0.60),
        _retrieved("n3_c1", "n3", 0.59),
    ]


class RerankerTests(unittest.TestCase):
    def test_rerank_updates_scores(self) -> None:
        config = _config()
        client = _StaticScoreClient([0.1, 0.9])
        results = rerank_chunks(
            query="q",
            retrieved_chunks=_two_chunks(),
            config=config,
            client=client,
            mock=False,
            top_k=2,
        )
        self.assertEqual(results[0].chunk.chunk_id, "c2")
        self.assertIn("rerank_score", results[0].retrieval_detail)

    def test_prevents_single_node_domination_with_auto_coverage(self) -> None:
        config = _config()
        scores = [0.99, 0.98, 0.97, 0.96, 0.95, 0.60, 0.59]
        client = _StaticScoreClient(scores)
        results = rerank_chunks(
            query="cross-country variation drivers",
            retrieved_chunks=_dominating_chunks(),
            config=config,
            client=client,
            mock=False,
            top_k=5,
        )

        self.assertEqual(len(results), 5)
        node_ids = [item.chunk.source_node_id for item in results]
        self.assertGreaterEqual(len(set(node_ids)), 3)
        self.assertIn("n2", node_ids)
        self.assertIn("n3", node_ids)

    def test_single_node_candidates_keep_top_scores(self) -> None:
        config = _config()
        chunks = [_retrieved(f"c{i}", "n1", 1.0 - i * 0.01) for i in range(6)]
        results = rerank_chunks(
            query="q",
            retrieved_chunks=chunks,
            config=config,
            client=None,
            mock=True,
            top_k=5,
        )

        self.assertEqual([item.chunk.chunk_id for item in results], [f"c{i}" for i in range(5)])

    def test_top_k_one_matches_global_top_one(self) -> None:
        config = _config()
        chunks = [
            _retrieved("c1", "n1", 0.8),
            _retrieved("c2", "n2", 0.95),
            _retrieved("c3", "n3", 0.7),
        ]
        results = rerank_chunks(
            query="q",
            retrieved_chunks=chunks,
            config=config,
            client=None,
            mock=True,
            top_k=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_rerank_failure_fallback_still_applies_diversity(self) -> None:
        config = _config()
        with self.assertLogs("tree_rag.retrieval.reranker", level="WARNING"):
            results = rerank_chunks(
                query="q",
                retrieved_chunks=_dominating_chunks(),
                config=config,
                client=_BadClient(),
                mock=False,
                top_k=5,
            )

        node_ids = [item.chunk.source_node_id for item in results]
        self.assertEqual(len(results), 5)
        self.assertGreaterEqual(len(set(node_ids)), 3)
        self.assertIn("n2", node_ids)
        self.assertIn("n3", node_ids)

    def test_rerank_does_not_send_top_n_truncation(self) -> None:
        config = _config()
        client = _StaticScoreClient([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        results = rerank_chunks(
            query="q",
            retrieved_chunks=[
                _retrieved("c1", "n1", 0.1),
                _retrieved("c2", "n1", 0.1),
                _retrieved("c3", "n2", 0.1),
                _retrieved("c4", "n2", 0.1),
                _retrieved("c5", "n3", 0.1),
                _retrieved("c6", "n3", 0.1),
            ],
            config=config,
            client=client,
            mock=False,
            top_k=5,
        )

        self.assertEqual(len(results), 5)
        self.assertIsNotNone(client.last_kwargs)
        self.assertIn("top_n", client.last_kwargs)
        self.assertIsNone(client.last_kwargs["top_n"])


if __name__ == "__main__":
    unittest.main()
