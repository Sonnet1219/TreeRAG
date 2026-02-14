import unittest

from tree_rag.indexing.bm25_builder import build_bm25_index
from tree_rag.retrieval.hybrid_retriever import hybrid_retrieve
from tree_rag.types import Chunk, IndexedNode
from tree_rag.utils.similarity import min_max_normalize
from tree_rag.utils.tokenizer import tokenize


class HybridRetrieverTests(unittest.TestCase):
    def test_min_max_normalize_handles_constant_vector(self) -> None:
        self.assertEqual(min_max_normalize([2.0, 2.0, 2.0]), [0.0, 0.0, 0.0])

    def test_hybrid_retrieve_returns_top_k(self) -> None:
        chunks = [
            Chunk(
                chunk_id="c1",
                text="contextual bandit router design",
                source_node_id="n1",
                heading_path="Methods",
                embedding=[1.0, 0.0, 0.0],
            ),
            Chunk(
                chunk_id="c2",
                text="ablation study result",
                source_node_id="n1",
                heading_path="Methods",
                embedding=[0.0, 1.0, 0.0],
            ),
        ]
        bm25 = build_bm25_index([tokenize(c.text) for c in chunks])
        node = IndexedNode(node_id="n1", heading_path="Methods", chunks=chunks, bm25_index=bm25)

        def _embed_query(_: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        results = hybrid_retrieve(
            node=node,
            query="contextual bandit",
            top_k=1,
            dense_weight=0.5,
            bm25_weight=0.5,
            embed_query_fn=_embed_query,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertIn("dense_score", results[0].retrieval_detail)
        self.assertIn("bm25_score", results[0].retrieval_detail)
        self.assertIn("fused_score", results[0].retrieval_detail)

    def test_empty_chunks_returns_empty_list(self) -> None:
        bm25 = build_bm25_index([["x"]])
        node = IndexedNode(node_id="n1", heading_path="Methods", chunks=[], bm25_index=bm25)
        results = hybrid_retrieve(
            node=node,
            query="test",
            top_k=3,
            dense_weight=0.5,
            bm25_weight=0.5,
            embed_query_fn=lambda _: [1.0],
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
