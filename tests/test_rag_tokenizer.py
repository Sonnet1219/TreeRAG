import unittest

from tree_rag.utils.tokenizer import tokenize


class TokenizerTests(unittest.TestCase):
    def test_english_tokenize_removes_stopwords(self) -> None:
        tokens = tokenize("The router is adaptive and effective")
        self.assertIn("router", tokens)
        self.assertIn("adaptive", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)

    def test_chinese_tokenize_returns_non_empty_tokens(self) -> None:
        tokens = tokenize("路由策略可以提升性能")
        self.assertTrue(tokens)


if __name__ == "__main__":
    unittest.main()
