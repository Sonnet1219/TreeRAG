from pathlib import Path
import unittest

from tree_builder.parser import parse_markdown_file, parse_markdown_sections
from tree_builder.tree import build_document_tree, postorder_nodes


TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "tree_builder" / "test_data"


def _levels_histogram(tree) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for node in postorder_nodes(tree.root):
        if node.level == 0:
            continue
        histogram[node.level] = histogram.get(node.level, 0) + 1
    return histogram


class TreeBuildTests(unittest.TestCase):
    def test_standard_markdown_structure(self) -> None:
        sections = parse_markdown_file(TEST_DATA_DIR / "test_standard.md")
        tree = build_document_tree("test_standard", sections)

        self.assertEqual([node.heading for node in tree.root.children], [
            "Introduction",
            "Methods",
            "Experiments",
            "Conclusion",
        ])

        introduction = tree.root.children[0]
        self.assertEqual([node.heading for node in introduction.children], [
            "Background",
            "Motivation",
        ])

        motivation = introduction.children[1]
        self.assertEqual([node.heading for node in motivation.children], [
            "Problem Statement",
            "Research Questions",
        ])
        self.assertEqual(motivation.children[0].heading_path, "Introduction > Motivation > Problem Statement")

        methods = tree.root.children[1]
        self.assertEqual([node.heading for node in methods.children], [
            "Data Collection",
            "Model Architecture",
        ])
        self.assertEqual([node.heading for node in methods.children[1].children], [
            "Encoder Design",
            "Decoder Design",
        ])

        self.assertEqual(tree.node_count, 12)
        self.assertEqual(tree.leaf_count, 8)

    def test_flat_markdown_structure_uses_numbering(self) -> None:
        sections = parse_markdown_file(TEST_DATA_DIR / "test_flat.md")
        tree = build_document_tree("test_flat", sections)

        self.assertEqual([node.heading for node in tree.root.children], [
            "1 Introduction",
            "2 Methods",
            "3 Experiments",
            "4 Conclusion",
        ])

        intro = tree.root.children[0]
        self.assertEqual([node.heading for node in intro.children], [
            "1.1 Background",
            "1.2 Motivation",
        ])
        self.assertEqual(intro.children[0].level, 2)

        methods = tree.root.children[1]
        self.assertEqual([node.heading for node in methods.children], [
            "2.1 Data Collection",
            "2.2 Model Architecture",
        ])
        self.assertEqual([node.heading for node in methods.children[0].children], [
            "2.1.1 Dataset A",
            "2.1.2 Dataset B",
        ])
        self.assertEqual(methods.children[0].children[0].level, 3)

        self.assertEqual(tree.node_count, 10)
        self.assertEqual(tree.leaf_count, 7)

    def test_standard_and_flat_have_comparable_topology(self) -> None:
        standard = build_document_tree(
            "test_standard",
            parse_markdown_file(TEST_DATA_DIR / "test_standard.md"),
        )
        flat = build_document_tree(
            "test_flat",
            parse_markdown_file(TEST_DATA_DIR / "test_flat.md"),
        )

        standard_levels = _levels_histogram(standard)
        flat_levels = _levels_histogram(flat)

        self.assertEqual(standard_levels.get(1), 4)
        self.assertEqual(flat_levels.get(1), 4)
        self.assertEqual(standard_levels.get(2), 4)
        self.assertEqual(flat_levels.get(2), 4)
        self.assertGreater(standard_levels.get(3, 0), 0)
        self.assertGreater(flat_levels.get(3, 0), 0)

    def test_node_id_is_unique_for_duplicate_numbering(self) -> None:
        markdown = """# 1 Intro
First.

# 1 Intro Again
Second.
"""
        sections = parse_markdown_sections(markdown)
        tree = build_document_tree("doc", sections)
        ids = [node.node_id for node in postorder_nodes(tree.root) if node.level > 0]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(any("_n1" in node_id for node_id in ids))


if __name__ == "__main__":
    unittest.main()
