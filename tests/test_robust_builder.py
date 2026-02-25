from pathlib import Path
import unittest

from tree_builder.builder import build_robust_tree
from tree_builder.tree import postorder_nodes


TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "tree_builder" / "test_data"


EXPECTED = {
    "test_standard.md": {
        "node_count": 12,
        "leaf_count": 8,
        "levels": {
            "Introduction": 1,
            "Background": 2,
            "Problem Statement": 3,
            "Methods": 1,
            "Model Architecture": 2,
        },
    },
    "test_flat_numbered.md": {
        "node_count": 9,
        "leaf_count": 6,
        "levels": {
            "1 Introduction": 1,
            "1.1 Background": 2,
            "2 Methods": 1,
            "3.1 Performance": 2,
        },
    },
    "test_flat_no_number.md": {
        "node_count": 8,
        "leaf_count": 8,
        "levels": {
            "Introduction": 1,
            "Methods": 1,
            "Conclusion": 1,
        },
    },
    "test_mixed_numbering.md": {
        "node_count": 6,
        "leaf_count": 4,
        "levels": {
            "1. Introduction": 1,
            "1.1 Background": 2,
            "A. Appendix": 1,
            "A.1 Dataset Details": 2,
        },
    },
    "test_noisy.md": {
        "node_count": 4,
        "leaf_count": 3,
        "levels": {
            "Introduction": 1,
            "Background": 2,
            "3. Methods": 1,
            "4 Results": 1,
        },
    },
    "test_level_jump.md": {
        "node_count": 5,
        "leaf_count": 4,
        "levels": {
            "Introduction": 1,
            "Detail A": 2,
            "Detail B": 2,
            "Methods": 2,
            "Conclusion": 1,
        },
    },
    "test_deep.md": {
        "node_count": 5,
        "leaf_count": 3,
        "levels": {
            "1 Introduction": 1,
            "1.1 Background": 2,
            "1.1.1 History": 3,
            "1.1.1.1 Early Work": 3,
            "1.1.1.1.1 Foundations": 3,
        },
    },
    "test_chinese.md": {
        "node_count": 5,
        "leaf_count": 5,
        "levels": {
            "第一章 绪论": 1,
            "第一节 背景": 1,
            "第二章 方法": 1,
            "第二节 数据处理": 1,
            "第三章 结论": 1,
        },
    },
}


class RobustBuilderTests(unittest.TestCase):
    def test_robust_builder_matrix(self) -> None:
        for filename, expected in EXPECTED.items():
            with self.subTest(filename=filename):
                markdown = (TEST_DATA_DIR / filename).read_text(encoding="utf-8")
                tree, report = build_robust_tree(markdown, doc_id=filename.replace(".md", ""), llm_client=None)

                self.assertEqual(tree.node_count, expected["node_count"])
                self.assertEqual(tree.leaf_count, expected["leaf_count"])

                levels = {
                    node.heading: node.level
                    for node in postorder_nodes(tree.root)
                    if node.level > 0
                }
                for heading, level in expected["levels"].items():
                    self.assertEqual(levels.get(heading), level)

                if filename == "test_flat_no_number.md":
                    self.assertGreater(len(report.low_confidence_headings), 0)
                if filename == "test_noisy.md":
                    self.assertNotIn("fake heading in code block", levels)


if __name__ == "__main__":
    unittest.main()
