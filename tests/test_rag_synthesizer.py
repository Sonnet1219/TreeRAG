import unittest

from tree_rag.config import load_rag_config
from tree_rag.retrieval.synthesizer import build_context, synthesize
from tree_rag.types import Chunk, RetrievedChunk


class _FakeClient:
    def chat_completion(self, **kwargs):  # noqa: ANN003
        return "final answer"


class SynthesizerTests(unittest.TestCase):
    def test_build_context_contains_source_paths(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="c1",
                    text="Alpha",
                    source_node_id="n1",
                    heading_path="Methods > Router Design",
                    embedding=[0.1],
                ),
                score=0.9,
                retrieval_detail={},
            )
        ]
        context = build_context(chunks)
        self.assertIn("[Evidence1]", context)
        self.assertIn("Methods > Router Design", context)

    def test_mock_synthesis_returns_joined_evidence(self) -> None:
        config = load_rag_config(load_dotenv=False)
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="c1",
                    text="Alpha",
                    source_node_id="n1",
                    heading_path="Methods",
                    embedding=[0.1],
                ),
                score=0.9,
                retrieval_detail={},
            )
        ]
        answer = synthesize("question", chunks, config=config, client=None, mock=True)
        self.assertIn("Based on retrieved evidence", answer)
        self.assertIn("Source: Methods", answer)

    def test_llm_synthesis_uses_client(self) -> None:
        config = load_rag_config(load_dotenv=False)
        chunks = [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="c1",
                    text="Alpha",
                    source_node_id="n1",
                    heading_path="Methods",
                    embedding=[0.1],
                ),
                score=0.9,
                retrieval_detail={},
            )
        ]
        answer = synthesize("question", chunks, config=config, client=_FakeClient(), mock=False)
        self.assertEqual(answer, "final answer")


if __name__ == "__main__":
    unittest.main()
