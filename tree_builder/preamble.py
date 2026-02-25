"""Preamble leaf injection for non-leaf nodes with direct content."""

from __future__ import annotations

from tree_builder.summary import Summarizer
from tree_builder.tree import TreeNode, postorder_nodes


def inject_preamble_leaves(root: TreeNode) -> int:
    """Inject one preamble leaf for each non-leaf node that has direct content."""
    injected = 0

    for node in postorder_nodes(root):
        if node.level == 0:
            continue
        if node.is_leaf:
            continue
        if not node.content.strip():
            continue

        preamble = TreeNode(
            node_id=f"{node.node_id}_preamble",
            heading=f"{node.heading} (Preamble)",
            level=node.level + 1,
            content=node.content,
            summary="",
            parent=node,
            children=[],
            heading_path=(f"{node.heading_path} > Preamble" if node.heading_path else "Preamble"),
        )

        node.children.insert(0, preamble)
        node.content = ""
        injected += 1

    return injected


def generate_preamble_summaries(root: TreeNode, summarizer: Summarizer) -> int:
    """Generate summary for injected preamble leaves."""
    summarized = 0
    for node in postorder_nodes(root):
        if not node.node_id.endswith("_preamble"):
            continue
        node.summary = summarizer.summarize_leaf(node.heading, node.content[:200])
        summarized += 1
    return summarized
