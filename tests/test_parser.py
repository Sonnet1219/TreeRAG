import unittest

from tree_builder.parser import parse_heading_line, parse_markdown_sections


class ParseHeadingLineTests(unittest.TestCase):
    def test_numbering_priority_over_hash_count(self) -> None:
        info = parse_heading_line("# 1.2.3 Deep Section")

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.hash_count, 1)
        self.assertEqual(info.numbering, "1.2.3")
        self.assertEqual(info.clean_title, "Deep Section")
        self.assertEqual(info.inferred_level, 3)

    def test_letter_numbering_supported(self) -> None:
        info = parse_heading_line("# A.1.2 Appendix Notes")

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.numbering, "A.1.2")
        self.assertEqual(info.clean_title, "Appendix Notes")
        self.assertEqual(info.inferred_level, 3)

    def test_level_cap_to_three(self) -> None:
        numbered = parse_heading_line("###### 1.2.3.4.5 Too Deep")
        plain = parse_heading_line("###### Also Too Deep")

        self.assertIsNotNone(numbered)
        self.assertIsNotNone(plain)
        assert numbered is not None
        assert plain is not None
        self.assertEqual(numbered.inferred_level, 3)
        self.assertEqual(plain.inferred_level, 3)

    def test_non_heading_line_returns_none(self) -> None:
        self.assertIsNone(parse_heading_line("No heading here"))


class ParseMarkdownSectionsTests(unittest.TestCase):
    def test_parse_sections_ignores_code_fence_headings(self) -> None:
        text = """# Intro
Content line.

```python
# inside code fence should be ignored
def foo():
    return 1
```

## Next
More content.
"""
        sections = parse_markdown_sections(text)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].heading.heading_raw, "Intro")
        self.assertEqual(sections[1].heading.heading_raw, "Next")

    def test_no_heading_returns_empty_list(self) -> None:
        text = "Plain text only.\nStill no headings."
        sections = parse_markdown_sections(text)
        self.assertEqual(sections, [])

    def test_case_1_2_3_level_inference(self) -> None:
        text = """# 1 Intro
## 1.1 Background
# 1.2.1 Nested with flat hash
"""
        sections = parse_markdown_sections(text)
        levels = [section.heading.inferred_level for section in sections]
        self.assertEqual(levels, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
