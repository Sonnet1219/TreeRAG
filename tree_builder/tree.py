"""Document tree data model and construction logic."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Optional

from tree_builder.parser import Section


NODE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class TreeNode:
    node_id: str
    heading: str
    level: int
    content: str
    summary: str
    parent: Optional["TreeNode"] = None
    children: list["TreeNode"] = field(default_factory=list)
    heading_path: str = ""

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class DocumentTree:
    doc_id: str
    root: TreeNode
    leaf_count: int
    node_count: int

    def recompute_counts(self) -> None:
        all_nodes = [node for node in traverse_all_nodes(self.root) if node.level > 0]
        self.node_count = len(all_nodes)
        self.leaf_count = sum(1 for node in all_nodes if node.is_leaf)


def _sanitize_node_suffix(raw_suffix: str, fallback_index: int) -> str:
    suffix = NODE_ID_RE.sub("_", raw_suffix).strip("_")
    if suffix:
        return suffix
    return f"s{fallback_index}"


def _make_node_id(doc_id: str, section: Section, seen_ids: dict[str, int]) -> str:
    raw_suffix = section.heading.numbering or f"s{section.index}"
    suffix = _sanitize_node_suffix(raw_suffix, section.index)
    base = f"{doc_id}_{suffix}"
    count = seen_ids.get(base, 0)
    seen_ids[base] = count + 1
    if count == 0:
        return base
    return f"{base}_n{count}"


def _iter_non_root_nodes(root: TreeNode) -> Iterable[TreeNode]:
    stack = list(reversed(root.children))
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def traverse_all_nodes(root: TreeNode) -> list[TreeNode]:
    """Return all nodes in pre-order, including root."""
    ordered: list[TreeNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        ordered.append(node)
        stack.extend(reversed(node.children))
    return ordered


def build_document_tree(
    doc_id: str,
    sections: list[Section],
    root_content: str = "",
) -> DocumentTree:
    """Build a document tree from parsed sections using a stack algorithm."""
    root = TreeNode(
        node_id="root",
        heading="ROOT",
        level=0,
        content=root_content.strip(),
        summary="",
        parent=None,
        children=[],
        heading_path="",
    )

    stack: list[TreeNode] = [root]
    seen_ids: dict[str, int] = {}

    for section in sections:
        level = min(max(section.heading.inferred_level, 1), 3)
        while stack and stack[-1].level >= level:
            stack.pop()
        if not stack:
            stack = [root]

        parent = stack[-1]
        heading = section.heading.heading_raw
        heading_path = heading if parent.level == 0 else f"{parent.heading_path} > {heading}"

        node = TreeNode(
            node_id=_make_node_id(doc_id, section, seen_ids),
            heading=heading,
            level=level,
            content=section.content.strip(),
            summary="",
            parent=parent,
            children=[],
            heading_path=heading_path,
        )
        parent.children.append(node)
        stack.append(node)

    tree = DocumentTree(doc_id=doc_id, root=root, leaf_count=0, node_count=0)
    tree.recompute_counts()
    return tree


def postorder_nodes(root: TreeNode) -> list[TreeNode]:
    """Return tree nodes in post-order, including root as the last element."""
    ordered: list[TreeNode] = []

    def visit(node: TreeNode) -> None:
        for child in node.children:
            visit(child)
        ordered.append(node)

    visit(root)
    return ordered


def validate_and_fix_tree(root: TreeNode, max_depth: int = 3) -> list[str]:
    """Validate and repair common structural issues after level inference."""
    fixes: list[str] = []

    for node in traverse_all_nodes(root):
        if node.parent is None:
            if node is not root:
                node.parent = root
                root.children.append(node)
                node.level = 1
                fixes.append(f"Orphan node adopted: {node.heading}")
            continue

        if node.level > node.parent.level + 1:
            old_level = node.level
            node.level = node.parent.level + 1
            fixes.append(
                f"Level gap adjusted: {node.heading} L{old_level} -> L{node.level}"
            )

        if node.level > max_depth:
            old_level = node.level
            node.level = max_depth
            fixes.append(
                f"Depth overflow adjusted: {node.heading} L{old_level} -> L{node.level}"
            )

    for node in list(traverse_all_nodes(root)):
        if node is root:
            continue
        if not node.children and not node.content.strip() and not node.summary.strip():
            parent = node.parent
            if parent is not None and node in parent.children:
                parent.children.remove(node)
                fixes.append(f"Empty node pruned: {node.heading}")

    return fixes
