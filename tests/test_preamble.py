import unittest

from tree_builder.builder import build_robust_tree
from tree_builder.preamble import generate_preamble_summaries, inject_preamble_leaves
from tree_builder.summary import MockSummarizer, generate_summaries
from tree_builder.tree import postorder_nodes


SAMPLE_MARKDOWN = """# Abstract
This paper proposes AdaRouter, a novel adaptive routing method for forex trading.

# 1 Introduction
The field of algorithmic trading has evolved rapidly over the past decade.
This section provides background and motivation for our research.

## 1.1 Background
Foreign exchange markets process over $6 trillion in daily volume.

## 1.2 Motivation
Static routing strategies fail to adapt to changing market conditions.

# 2 Methods
We propose a two-component architecture consisting of an encoder and a router.
The overall design philosophy emphasizes adaptability and real-time decision making.

## 2.1 Encoder Design
The encoder uses a Transformer architecture to process time-series data.

## 2.2 Router Design
The router employs a contextual bandit framework for dynamic routing decisions.

# 3 Experiments
We evaluate AdaRouter on multiple currency pairs spanning 2020-2023.
All experiments were conducted on NVIDIA A100 GPUs with identical hyperparameters.

## 3.1 Performance
AdaRouter achieves 73.2% accuracy on EUR/USD, outperforming the baseline.

## 3.2 Ablation Study
Removing the adaptive routing module results in a 8.1% accuracy drop.

# 4 Conclusion
We presented AdaRouter, demonstrating significant improvements over static methods.
"""


class PreambleTests(unittest.TestCase):
    def test_inject_preamble_nodes_and_counts(self) -> None:
        tree, _ = build_robust_tree(SAMPLE_MARKDOWN, doc_id="paper", llm_client=None)
        self.assertEqual(tree.node_count, 11)
        self.assertEqual(tree.leaf_count, 8)

        generate_summaries(tree, MockSummarizer(max_chars=120))
        injected = inject_preamble_leaves(tree.root)
        generate_preamble_summaries(tree.root, MockSummarizer(max_chars=120))
        tree.recompute_counts()

        self.assertEqual(injected, 3)
        self.assertEqual(tree.node_count, 14)
        self.assertEqual(tree.leaf_count, 11)

        top_headings = [node.heading for node in tree.root.children]
        self.assertIn("Abstract", top_headings)
        self.assertIn("4 Conclusion", top_headings)

        intro = next(node for node in tree.root.children if node.heading == "1 Introduction")
        methods = next(node for node in tree.root.children if node.heading == "2 Methods")
        experiments = next(node for node in tree.root.children if node.heading == "3 Experiments")

        for parent in (intro, methods, experiments):
            self.assertFalse(parent.is_leaf)
            self.assertTrue(parent.children)
            first_child = parent.children[0]
            self.assertTrue(first_child.node_id.endswith("_preamble"))
            self.assertTrue(first_child.heading.endswith("(Preamble)"))
            self.assertEqual(first_child.level, parent.level + 1)
            self.assertTrue(first_child.is_leaf)
            self.assertTrue(first_child.summary)

        preambles = [
            node for node in postorder_nodes(tree.root)
            if node.node_id.endswith("_preamble")
        ]
        self.assertEqual(len(preambles), 3)


if __name__ == "__main__":
    unittest.main()
