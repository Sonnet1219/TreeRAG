# Tree Builder

Tree Builder parses a Markdown document into a 3-level section tree, generates section summaries, prints an ASCII view in terminal, and exports JSON.

## Requirements

- Python 3.10+
- Optional: `python-dotenv` for `.env` loading (a built-in fallback loader is also provided)

## Quick Start

```bash
python -m tree_builder.main tree_builder/test_data/test_standard.md --mode mock
```

This command:

1. Parses markdown headings into sections.
2. Infers heading levels using numbering-first fallback-to-hash logic.
3. Builds a tree with max depth 3.
4. Generates summaries with `MockSummarizer`.
5. Prints the tree and writes JSON to `tree_builder/test_data/test_standard.tree.json`.

## CLI

```bash
python -m tree_builder.main <input_markdown> [--mode mock|llm] [--provider openai|anthropic] [--output path.json]
```

- `--mode mock` (default): local truncation summary.
- `--mode llm`: real OpenAI-compatible chat completion for `--provider openai`.
- `--output`: optional output path. If omitted, output is `<input_stem>.tree.json` next to input file.

## LLM Configuration (.env)

Create or update `.env` in project root:

```bash
OPENAI_API_KEY=your_real_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_TOKENS=120
OPENAI_TEMPERATURE=0.2
```

Then run:

```bash
python -m tree_builder.main path/to/your.md --mode llm --provider openai
```

If `OPENAI_API_KEY` is missing, CLI exits with a clear error.
If you want to load a different env file, set `TREE_BUILDER_ENV_FILE=/path/to/custom.env`.

## Included Test Data

- `tree_builder/test_data/test_standard.md`
- `tree_builder/test_data/test_flat.md`

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- Only ATX headings are recognized (`#` to `######`).
- Numbered headings like `1.2.3` and `A.1.2` take priority for level inference.
- Heading levels are capped at 3.
- Headings inside fenced code blocks are ignored.
