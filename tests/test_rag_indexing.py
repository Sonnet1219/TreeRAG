import json
from pathlib import Path
import tempfile
import unittest

from tree_rag.config import load_rag_config
from tree_rag.indexing.chunker import chunk_content
from tree_rag.indexing.index_store import build_index_from_tree, load_index, load_tree_input, save_index


def _sample_tree() -> dict:
    return {
        "doc_id": "sample_doc",
        "tree": {
            "node_id": "root",
            "heading": "ROOT",
            "level": 0,
            "summary": "",
            "content": "",
            "heading_path": "",
            "is_leaf": False,
            "children": [
                {
                    "node_id": "n1",
                    "heading": "Methods",
                    "level": 1,
                    "summary": "Method summary",
                    "content": "Alpha beta gamma.\n\nDelta epsilon zeta.",
                    "heading_path": "Methods",
                    "is_leaf": True,
                    "children": [],
                },
                {
                    "node_id": "n2",
                    "heading": "Results",
                    "level": 1,
                    "summary": "Result summary",
                    "content": "Accuracy reaches 73.2% in EUR/USD experiments.",
                    "heading_path": "Results",
                    "is_leaf": True,
                    "children": [],
                },
            ],
        },
    }


class ChunkerTests(unittest.TestCase):
    def test_chunk_overlap_is_applied(self) -> None:
        content = "A" * 250
        chunks = chunk_content(content, chunk_size=100, overlap=20, min_chars=10)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0][-20:], chunks[1][:20])


class IndexStoreTests(unittest.TestCase):
    def test_build_save_load_roundtrip(self) -> None:
        config = load_rag_config(load_dotenv=False)
        tree = _sample_tree()
        index = build_index_from_tree(tree, config=config, mock=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "index"
            save_index(index, output_dir)

            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertTrue((output_dir / "chunks.jsonl").exists())
            self.assertTrue((output_dir / "embeddings.npy").exists())
            self.assertTrue((output_dir / "bm25.pkl").exists())

            loaded = load_index(output_dir)

        self.assertEqual(loaded.doc_id, index.doc_id)
        self.assertEqual(set(loaded.nodes.keys()), set(index.nodes.keys()))
        self.assertEqual(len(loaded.all_chunks), len(index.all_chunks))
        self.assertTrue(loaded.tree)

    def test_metadata_is_json(self) -> None:
        config = load_rag_config(load_dotenv=False)
        tree = _sample_tree()
        index = build_index_from_tree(tree, config=config, mock=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "index"
            save_index(index, output_dir)
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["doc_id"], "sample_doc")
        self.assertGreater(metadata["chunk_count"], 0)

    def test_markdown_input_includes_preamble_leaf_nodes(self) -> None:
        markdown = """# Methods
We propose a two-component architecture.

## Encoder
Encoder details.

## Router
Router details.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "paper.md"
            md_path.write_text(markdown, encoding="utf-8")

            tree_data = load_tree_input(md_path)
            config = load_rag_config(load_dotenv=False)
            index = build_index_from_tree(tree_data=tree_data, config=config, mock=True)

        preamble_nodes = [node_id for node_id in index.nodes if node_id.endswith("_preamble")]
        self.assertTrue(preamble_nodes)
        self.assertGreaterEqual(len(index.nodes), 3)


if __name__ == "__main__":
    unittest.main()
