import unittest

from tree_rag.config import load_rag_config
from tree_rag.retrieval.node_locator import locate_nodes, serialize_tree


class _FakeClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def chat_completion(self, **kwargs):  # noqa: ANN003
        return self.payload


def _tree() -> dict:
    return {
        "doc_id": "d1",
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
                    "heading": "Intro",
                    "level": 1,
                    "summary": "overview",
                    "content": "",
                    "heading_path": "Intro",
                    "is_leaf": True,
                    "children": [],
                },
                {
                    "node_id": "n2",
                    "heading": "Methods",
                    "level": 1,
                    "summary": "contains contextual bandit",
                    "content": "",
                    "heading_path": "Methods",
                    "is_leaf": True,
                    "children": [],
                },
            ],
        },
    }


class NodeLocatorTests(unittest.TestCase):
    def test_serialize_tree_contains_leaf_marker(self) -> None:
        serialized = serialize_tree(_tree())
        self.assertIn("Leaf Node", serialized)
        self.assertIn("[n2] Methods", serialized)

    def test_mock_locator_returns_leaf_nodes(self) -> None:
        config = load_rag_config(load_dotenv=False)
        nodes, thinking = locate_nodes(
            query="contextual bandit 是什么",
            tree_data=_tree(),
            config=config,
            client=None,
            mock=True,
            top_k=3,
        )
        self.assertTrue(nodes)
        self.assertEqual(nodes[0].node_id, "n2")
        self.assertTrue(thinking)

    def test_invalid_json_falls_back_to_mock(self) -> None:
        config = load_rag_config(load_dotenv=False)
        client = _FakeClient(payload="not-json")
        nodes, _ = locate_nodes(
            query="methods",
            tree_data=_tree(),
            config=config,
            client=client,
            mock=False,
            top_k=3,
        )
        self.assertTrue(nodes)
        self.assertEqual(nodes[0].node_id, "n2")


if __name__ == "__main__":
    unittest.main()
