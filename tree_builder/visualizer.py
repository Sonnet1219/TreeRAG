"""Tree rendering and serialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tree_builder.tree import DocumentTree, TreeNode


def _summary_preview(summary: str, max_chars: int) -> str:
    return " ".join(summary.split())[:max_chars]


def _node_to_dict(node: TreeNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "heading": node.heading,
        "level": node.level,
        "content": node.content,
        "summary": node.summary,
        "heading_path": node.heading_path,
        "is_leaf": node.is_leaf,
        "children": [_node_to_dict(child) for child in node.children],
    }


def document_tree_to_dict(tree: DocumentTree) -> dict[str, Any]:
    """Serialize DocumentTree into a JSON-compatible dictionary."""
    return {
        "doc_id": tree.doc_id,
        "node_count": tree.node_count,
        "leaf_count": tree.leaf_count,
        "tree": _node_to_dict(tree.root),
    }


def export_document_tree_json(tree: DocumentTree, output_path: Path) -> None:
    """Export tree to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document_tree_to_dict(tree), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_document_tree(tree: DocumentTree, summary_preview_chars: int = 50) -> None:
    """Print a readable ASCII tree with summary and content info."""
    print(f"Document Tree: {tree.doc_id} ({tree.node_count} nodes, {tree.leaf_count} leaves)")
    print("=" * 60)

    def print_node(node: TreeNode, prefix: str, is_last: bool) -> None:
        connector = "`-- " if is_last else "|-- "
        leaf_mark = " <- LEAF" if node.is_leaf else ""
        print(f"{prefix}{connector}[L{node.level}] {node.heading} ({len(node.content)} chars){leaf_mark}")

        summary_prefix = "    " if is_last else "|   "
        preview = _summary_preview(node.summary, summary_preview_chars)
        print(f"{prefix}{summary_prefix}Summary: \"{preview}\"")

        child_prefix = prefix + ("    " if is_last else "|   ")
        for index, child in enumerate(node.children):
            print_node(child, child_prefix, index == len(node.children) - 1)

    for index, child in enumerate(tree.root.children):
        print_node(child, "", index == len(tree.root.children) - 1)
