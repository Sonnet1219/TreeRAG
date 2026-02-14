import json
import unittest
from urllib import error as urlerror
from unittest.mock import patch

from tree_rag.utils.openai_client import OpenAICompatibleClient


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class OpenAIClientTests(unittest.TestCase):
    def test_chat_completion_calls_chat_endpoint(self) -> None:
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="https://example.local/v1",
            timeout_seconds=5.0,
        )
        payload = {"choices": [{"message": {"content": "answer"}}]}
        with patch("tree_rag.utils.openai_client.request.urlopen", return_value=_FakeHTTPResponse(payload)) as mocked:
            text = client.chat_completion(
                model="qwen-plus",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                temperature=0.2,
            )
        self.assertEqual(text, "answer")
        req = mocked.call_args.args[0]
        self.assertEqual(req.full_url, "https://example.local/v1/chat/completions")

    def test_embeddings_calls_embeddings_endpoint(self) -> None:
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="https://example.local/v1",
            timeout_seconds=5.0,
        )
        payload = {"data": [{"embedding": [0.1, 0.2]}]}
        with patch("tree_rag.utils.openai_client.request.urlopen", return_value=_FakeHTTPResponse(payload)) as mocked:
            vectors = client.embeddings(model="embed-v1", texts=["hello"])
        self.assertEqual(vectors, [[0.1, 0.2]])
        req = mocked.call_args.args[0]
        self.assertEqual(req.full_url, "https://example.local/v1/embeddings")

    def test_rerank_calls_rerank_endpoint(self) -> None:
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="https://example.local/v1",
            timeout_seconds=5.0,
        )
        payload = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]}
        with patch("tree_rag.utils.openai_client.request.urlopen", return_value=_FakeHTTPResponse(payload)) as mocked:
            scores = client.rerank(
                model="rerank-v1",
                query="q",
                documents=["d1", "d2"],
                top_n=2,
            )
        self.assertEqual(scores[1], 0.9)
        req = mocked.call_args.args[0]
        self.assertEqual(req.full_url, "https://example.local/v1/rerank")

    def test_dashscope_rerank_uses_native_endpoint(self) -> None:
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout_seconds=5.0,
        )
        payload = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }
        }
        with patch("tree_rag.utils.openai_client.request.urlopen", return_value=_FakeHTTPResponse(payload)) as mocked:
            scores = client.rerank(
                model="gte-rerank-v2",
                query="q",
                documents=["d1", "d2"],
                top_n=2,
            )
        self.assertEqual(scores[1], 0.9)
        req = mocked.call_args.args[0]
        self.assertEqual(
            req.full_url,
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        )

    def test_embeddings_retries_on_transient_network_error(self) -> None:
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="https://example.local/v1",
            timeout_seconds=5.0,
            max_retries=2,
            retry_backoff_seconds=0.0,
        )
        payload = {"data": [{"embedding": [0.1, 0.2]}]}
        with patch(
            "tree_rag.utils.openai_client.request.urlopen",
            side_effect=[urlerror.URLError("timed out"), _FakeHTTPResponse(payload)],
        ) as mocked:
            vectors = client.embeddings(model="embed-v1", texts=["hello"])
        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertEqual(mocked.call_count, 2)


if __name__ == "__main__":
    unittest.main()
