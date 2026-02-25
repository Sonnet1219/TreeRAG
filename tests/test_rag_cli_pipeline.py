from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tree_rag.main", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _tree_json_text() -> str:
    return """{
  "doc_id": "sample",
  "node_count": 2,
  "leaf_count": 2,
  "tree": {
    "node_id": "root",
    "heading": "ROOT",
    "level": 0,
    "content": "",
    "summary": "",
    "heading_path": "",
    "is_leaf": false,
    "children": [
      {
        "node_id": "n1",
        "heading": "Methods",
        "level": 1,
        "content": "AdaRouter uses contextual bandit routing strategy.",
        "summary": "router design and mechanism",
        "heading_path": "Methods",
        "is_leaf": true,
        "children": []
      },
      {
        "node_id": "n2",
        "heading": "Performance",
        "level": 1,
        "content": "Accuracy reaches 73.2% on EUR/USD.",
        "summary": "performance numbers",
        "heading_path": "Performance",
        "is_leaf": true,
        "children": []
      }
    ]
  }
}"""


class RagCliTests(unittest.TestCase):
    def test_index_then_query_in_mock_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_tree = tmp_path / "sample.tree.json"
            output_dir = tmp_path / "index"
            input_tree.write_text(_tree_json_text(), encoding="utf-8")

            index_result = _run_cli(
                ["index", "--input", str(input_tree), "--output", str(output_dir), "--mock"]
            )
            self.assertEqual(index_result.returncode, 0, msg=index_result.stderr)
            self.assertTrue((output_dir / "metadata.json").exists())

            query_result = _run_cli(
                [
                    "query",
                    "--index",
                    str(output_dir),
                    "--query",
                    "AdaRouter 的核心机制是什么？",
                    "--mock",
                ]
            )
            self.assertEqual(query_result.returncode, 0, msg=query_result.stderr)
            self.assertIn("Step 1: Node Locating", query_result.stdout)
            self.assertIn("Step 3: Answer", query_result.stdout)

    def test_index_from_markdown_in_mock_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_md = tmp_path / "sample.md"
            output_dir = tmp_path / "index_md"
            input_md.write_text(
                "# Intro\nLightRAG improves retrieval.\n\n# Methods\nDual-level retrieval overview.\n\n## Local Retrieval\nUses local keywords.\n\n## Global Retrieval\nUses global context.\n",
                encoding="utf-8",
            )

            index_result = _run_cli(
                ["index", "--input", str(input_md), "--output", str(output_dir), "--mock"]
            )
            self.assertEqual(index_result.returncode, 0, msg=index_result.stderr)
            self.assertTrue((output_dir / "metadata.json").exists())

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            node_ids = set(metadata["node_chunk_ids"].keys())
            self.assertTrue(any(node_id.endswith("_preamble") for node_id in node_ids))


if __name__ == "__main__":
    unittest.main()
