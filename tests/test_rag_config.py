import os
import unittest
from unittest.mock import patch

from tree_rag.config import load_rag_config


class RagConfigTests(unittest.TestCase):
    def test_defaults_are_applied(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = load_rag_config(load_dotenv=False)

        self.assertEqual(config.llm_model, "qwen-plus")
        self.assertEqual(config.embed_model, "text-embedding-v3")
        self.assertEqual(config.rerank_model, "gte-rerank-v2")
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.dense_weight, 0.5)
        self.assertEqual(config.bm25_weight, 0.5)
        self.assertTrue(config.rerank_diversify)
        self.assertEqual(config.rerank_min_unique_nodes, 0)
        self.assertEqual(config.openai_base_url, "https://api.openai.com/v1")

    def test_env_overrides_are_applied(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://example.com/v1",
                "RAG_LLM_MODEL": "qwen-max",
                "RAG_EMBED_MODEL": "embed-v1",
                "RAG_RERANK_MODEL": "rerank-v1",
                "RAG_TIMEOUT_SECONDS": "45",
                "RAG_TOP_K": "7",
                "RAG_DENSE_WEIGHT": "0.7",
                "RAG_BM25_WEIGHT": "0.3",
                "RAG_RERANK_DIVERSIFY": "false",
                "RAG_RERANK_MIN_UNIQUE_NODES": "3",
            },
            clear=True,
        ):
            config = load_rag_config(load_dotenv=False)

        self.assertEqual(config.openai_api_key, "test-key")
        self.assertEqual(config.openai_base_url, "https://example.com/v1")
        self.assertEqual(config.llm_model, "qwen-max")
        self.assertEqual(config.embed_model, "embed-v1")
        self.assertEqual(config.rerank_model, "rerank-v1")
        self.assertEqual(config.timeout_seconds, 45.0)
        self.assertEqual(config.top_k, 7)
        self.assertEqual(config.dense_weight, 0.7)
        self.assertEqual(config.bm25_weight, 0.3)
        self.assertFalse(config.rerank_diversify)
        self.assertEqual(config.rerank_min_unique_nodes, 3)


if __name__ == "__main__":
    unittest.main()
