"""Document tree data model and construction logic."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

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


def _iter_non_root_nodes(root: TreeNode):
    stack = list(reversed(root.children))
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


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

    all_nodes = list(_iter_non_root_nodes(root))
    node_count = len(all_nodes)
    leaf_count = sum(1 for node in all_nodes if node.is_leaf)
    return DocumentTree(doc_id=doc_id, root=root, leaf_count=leaf_count, node_count=node_count)


def postorder_nodes(root: TreeNode) -> list[TreeNode]:
    """Return tree nodes in post-order, including root as the last element."""
    ordered: list[TreeNode] = []

    def visit(node: TreeNode) -> None:
        for child in node.children:
            visit(child)
        ordered.append(node)

    visit(root)
    return ordered
