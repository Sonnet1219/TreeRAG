import json
import os
from unittest.mock import patch
import unittest

from tree_builder.parser import parse_markdown_sections
from tree_builder.summary import (
    LLMSummarizerStub,
    MockSummarizer,
    OpenAICompatibleSummarizer,
    build_llm_summarizer_from_env,
    generate_summaries,
)
from tree_builder.tree import build_document_tree


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class SummaryTests(unittest.TestCase):
    def test_leaf_summary_comes_from_content(self) -> None:
        markdown = """# Intro
Alpha beta gamma about the research topic.
"""
        tree = build_document_tree("doc", parse_markdown_sections(markdown))
        generate_summaries(tree, MockSummarizer(max_chars=100))

        leaf = tree.root.children[0]
        self.assertIn("Alpha beta gamma", leaf.summary)

    def test_parent_summary_comes_from_children_summaries(self) -> None:
        markdown = """# Methods
High level methods section.

## Dataset A
This child describes dataset A.

## Dataset B
This child describes dataset B.
"""
        tree = build_document_tree("doc", parse_markdown_sections(markdown))
        generate_summaries(tree, MockSummarizer(max_chars=100))

        parent = tree.root.children[0]
        self.assertTrue(parent.summary)
        self.assertIn("dataset", parent.summary.lower())

    def test_empty_content_gets_placeholder_summary(self) -> None:
        markdown = "# Empty Section\n\n"
        tree = build_document_tree("doc", parse_markdown_sections(markdown))
        generate_summaries(tree, MockSummarizer(max_chars=100))

        leaf = tree.root.children[0]
        self.assertTrue(leaf.summary)
        self.assertIn("no content", leaf.summary.lower())

    def test_llm_stub_raises_not_implemented(self) -> None:
        markdown = """# Intro
Some content.
"""
        tree = build_document_tree("doc", parse_markdown_sections(markdown))
        with self.assertRaises(NotImplementedError):
            generate_summaries(tree, LLMSummarizerStub(provider="openai"))

    def test_openai_summarizer_calls_chat_completion_endpoint(self) -> None:
        summarizer = OpenAICompatibleSummarizer(
            api_key="test-key",
            base_url="https://example.local/v1",
            model="gpt-test",
            timeout_seconds=5.0,
            max_tokens=64,
        )
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Concise section summary.",
                    }
                }
            ]
        }

        with patch("tree_builder.summary.request.urlopen", return_value=_FakeHTTPResponse(payload)) as mocked_urlopen:
            summary = summarizer.summarize_leaf("Intro", "Some content to summarize.")

        self.assertEqual(summary, "Concise section summary.")
        self.assertEqual(mocked_urlopen.call_count, 1)

        sent_request = mocked_urlopen.call_args.args[0]
        self.assertEqual(sent_request.full_url, "https://example.local/v1/chat/completions")

        body = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(body["model"], "gpt-test")
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(len(body["messages"]), 2)

    def test_build_llm_summarizer_from_env_requires_openai_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://example.local/v1"}, clear=True):
            with self.assertRaises(ValueError):
                build_llm_summarizer_from_env("openai")


if __name__ == "__main__":
    unittest.main()
