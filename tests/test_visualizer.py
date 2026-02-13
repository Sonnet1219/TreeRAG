from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from tree_builder.parser import parse_markdown_sections
from tree_builder.summary import MockSummarizer, generate_summaries
from tree_builder.tree import build_document_tree
from tree_builder.visualizer import (
    document_tree_to_dict,
    export_document_tree_json,
    print_document_tree,
)


def _build_sample_tree():
    markdown = """# Intro
Intro content.

## Detail
Detail content.
"""
    tree = build_document_tree("sample", parse_markdown_sections(markdown))
    generate_summaries(tree, MockSummarizer())
    return tree


class VisualizerTests(unittest.TestCase):
    def test_document_tree_to_dict_has_required_fields(self) -> None:
        tree = _build_sample_tree()
        data = document_tree_to_dict(tree)

        self.assertEqual(set(data.keys()), {"doc_id", "node_count", "leaf_count", "tree"})

        root = data["tree"]
        self.assertEqual(
            set(root.keys()),
            {"node_id", "heading", "level", "content", "summary", "heading_path", "is_leaf", "children"},
        )

    def test_is_leaf_matches_children(self) -> None:
        tree = _build_sample_tree()
        data = document_tree_to_dict(tree)

        def assert_node(node: dict) -> None:
            self.assertEqual(node["is_leaf"], len(node["children"]) == 0)
            for child in node["children"]:
                assert_node(child)

        assert_node(data["tree"])

    def test_export_json_round_trip(self) -> None:
        tree = _build_sample_tree()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tree.json"
            export_document_tree_json(tree, output_path)

            self.assertTrue(output_path.exists())
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["doc_id"], "sample")
            self.assertEqual(loaded["node_count"], 2)
            self.assertEqual(loaded["leaf_count"], 1)

    def test_print_document_tree_contains_core_fields(self) -> None:
        tree = _build_sample_tree()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_document_tree(tree, summary_preview_chars=20)

        output = buffer.getvalue()
        self.assertIn("Document Tree: sample", output)
        self.assertIn("[L1] Intro", output)
        self.assertIn("Summary:", output)


if __name__ == "__main__":
    unittest.main()
