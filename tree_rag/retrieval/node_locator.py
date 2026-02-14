"""Step 1: locate relevant leaf nodes from serialized tree structure."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tree_rag.types import NodeLocateResult, RagConfig
from tree_rag.utils.openai_client import OpenAICompatibleClient
from tree_rag.utils.tokenizer import tokenize


LOGGER = logging.getLogger(__name__)


NODE_LOCATE_PROMPT = """You are a document retrieval expert. Given a user question and a tree-structured
document outline, identify the leaf nodes most likely to contain the answer.

Rules:
1. Return only leaf nodes (marked as "Leaf Node").
2. Return 1-5 of the most relevant nodes.
3. For each selected node, generate a focused sub_query describing what information to retrieve.

User Question: {query}

Document Structure:
{serialized_tree}

Return strict JSON:
{{
  "thinking": "Briefly explain what information is needed and where it is likely located.",
  "results": [
    {{"node_id": "0007", "sub_query": "example sub query"}}
  ]
}}
"""


def _iter_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    stack = [node]
    result: list[dict[str, Any]] = []
    while stack:
        current = stack.pop()
        result.append(current)
        children = current.get("children", [])
        if isinstance(children, list):
            stack.extend(reversed(children))
    return result


def _leaf_nodes(tree_data: dict[str, Any]) -> list[dict[str, Any]]:
    root = tree_data.get("tree", {})
    leaves: list[dict[str, Any]] = []
    for node in _iter_nodes(root):
        children = node.get("children", [])
        is_leaf = node.get("is_leaf")
        if is_leaf is None:
            is_leaf = not bool(children)
        if is_leaf and node.get("node_id") != "root":
            leaves.append(node)
    return leaves


def serialize_tree(tree_data: dict[str, Any]) -> str:
    lines = ["Document Structure:"]

    def visit(node: dict[str, Any], depth: int) -> None:
        if node.get("node_id") != "root":
            mark = " (Leaf Node)" if node.get("is_leaf", not node.get("children")) else ""
            heading = node.get("heading", "")
            node_id = node.get("node_id", "")
            summary = str(node.get("summary", "")).strip()
            indent = "  " * depth
            lines.append(f"{indent}[{node_id}] {heading}{mark}")
            if summary:
                lines.append(f"{indent}  Summary: {summary}")
        for child in node.get("children", []):
            visit(child, depth + 1)

    visit(tree_data.get("tree", {}), 0)
    return "\n".join(lines)


def _keyword_locate(query: str, tree_data: dict[str, Any], top_k: int) -> tuple[list[NodeLocateResult], str]:
    query_tokens = set(tokenize(query))
    candidates: list[tuple[float, dict[str, Any]]] = []
    for node in _leaf_nodes(tree_data):
        text = f"{node.get('heading', '')} {node.get('summary', '')}".strip()
        node_tokens = set(tokenize(text))
        overlap = len(query_tokens.intersection(node_tokens))
        substring_boost = 0.0
        lowered = text.lower()
        for token in query_tokens:
            if token and token in lowered:
                substring_boost += 0.25
        score = overlap + substring_boost
        candidates.append((score, node))

    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    selected = [node for _, node in ranked[: max(1, top_k)] if node.get("node_id")]
    if not selected and candidates:
        selected = [candidates[0][1]]

    results = [
        NodeLocateResult(
            node_id=str(node["node_id"]),
            sub_query=query,
        )
        for node in selected
    ]
    thinking = "Keyword fallback based on heading and summary matching."
    return results, thinking


def _extract_json_payload(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def locate_nodes(
    query: str,
    tree_data: dict[str, Any],
    config: RagConfig,
    client: OpenAICompatibleClient | None,
    mock: bool,
    top_k: int | None = None,
) -> tuple[list[NodeLocateResult], str]:
    resolved_top_k = max(1, min(top_k or config.top_k, 5))
    if mock or client is None:
        LOGGER.info("Node locating is using keyword fallback (mock mode or missing client).")
        return _keyword_locate(query=query, tree_data=tree_data, top_k=resolved_top_k)

    prompt = NODE_LOCATE_PROMPT.format(
        query=query,
        serialized_tree=serialize_tree(tree_data),
    )
    try:
        response = client.chat_completion(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": "You are a precise retrieval assistant. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        payload = _extract_json_payload(response)
    except Exception as exc:
        LOGGER.warning(
            "LLM node locating failed (%s). Falling back to keyword matching.",
            exc,
        )
        return _keyword_locate(query=query, tree_data=tree_data, top_k=resolved_top_k)

    leaf_ids = {str(node.get("node_id")) for node in _leaf_nodes(tree_data)}
    raw_results = payload.get("results", [])
    results: list[NodeLocateResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", "")).strip()
        if node_id not in leaf_ids:
            continue
        sub_query = str(item.get("sub_query", query)).strip() or query
        results.append(NodeLocateResult(node_id=node_id, sub_query=sub_query))
        if len(results) >= resolved_top_k:
            break

    if not results:
        LOGGER.warning("LLM node locating returned no valid leaf nodes. Falling back to keyword matching.")
        return _keyword_locate(query=query, tree_data=tree_data, top_k=resolved_top_k)

    thinking = str(payload.get("thinking", "")).strip() or "LLM returned candidate nodes."
    return results, thinking
