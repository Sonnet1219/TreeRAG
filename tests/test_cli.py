import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import unittest

from tree_builder.main import run_cli
from tree_builder.summary import MockSummarizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tree_builder.main", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


class CliTests(unittest.TestCase):
    def test_mock_mode_runs_and_writes_default_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = Path(tmpdir) / "sample.md"
            markdown_path.write_text("# Intro\nSome content.\n", encoding="utf-8")

            result = _run_cli([str(markdown_path), "--mode", "mock"])

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(markdown_path.with_suffix(".tree.json").exists())
            self.assertIn("Document Tree", result.stdout)

    def test_output_argument_overrides_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = Path(tmpdir) / "sample.md"
            output_path = Path(tmpdir) / "custom_output.json"
            markdown_path.write_text("# Intro\nSome content.\n", encoding="utf-8")

            result = _run_cli(
                [
                    str(markdown_path),
                    "--mode",
                    "mock",
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())

    def test_llm_mode_missing_env_returns_nonzero_and_error_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = Path(tmpdir) / "sample.md"
            markdown_path.write_text("# Intro\nSome content.\n", encoding="utf-8")

            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.pop("OPENAI_BASE_URL", None)
            env.pop("OPENAI_MODEL", None)
            env["TREE_BUILDER_ENV_FILE"] = str(Path(tmpdir) / "missing.env")

            result = _run_cli(
                [
                    str(markdown_path),
                    "--mode",
                    "llm",
                    "--provider",
                    "openai",
                ],
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("OPENAI_API_KEY", result.stderr)

    def test_llm_mode_uses_layer3_correction_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = Path(tmpdir) / "sample.md"
            output_path = Path(tmpdir) / "sample.tree.json"
            markdown_path.write_text(
                "# Introduction\nintro text\n\n# Background\nbackground text\n\n# Methods\nmethods text\n",
                encoding="utf-8",
            )

            class _FakeClient:
                def __init__(self) -> None:
                    self.calls = 0

                def chat_completion(self, **kwargs):  # noqa: ANN003
                    self.calls += 1
                    return (
                        '{"results": ['
                        '{"index": 0, "level": 1, "reasoning": "top"},'
                        '{"index": 1, "level": 2, "reasoning": "child"},'
                        '{"index": 2, "level": 1, "reasoning": "top"}'
                        "]}"
                    )

            fake_client = _FakeClient()
            with patch("tree_builder.main.build_llm_summarizer_from_env", return_value=MockSummarizer()):
                with patch(
                    "tree_builder.main.build_openai_tree_llm_client_from_env",
                    return_value=(fake_client, "gpt-test"),
                ):
                    code = run_cli(
                        [
                            str(markdown_path),
                            "--mode",
                            "llm",
                            "--provider",
                            "openai",
                            "--output",
                            str(output_path),
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertGreater(fake_client.calls, 0)
            self.assertTrue(output_path.exists())

    def test_llm_mode_strict_failure_when_layer3_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = Path(tmpdir) / "sample.md"
            markdown_path.write_text(
                "# Introduction\nintro text\n\n# Background\nbackground text\n",
                encoding="utf-8",
            )

            class _FailingClient:
                def chat_completion(self, **kwargs):  # noqa: ANN003
                    raise RuntimeError("llm correction failure")

            with patch("tree_builder.main.build_llm_summarizer_from_env", return_value=MockSummarizer()):
                with patch(
                    "tree_builder.main.build_openai_tree_llm_client_from_env",
                    return_value=(_FailingClient(), "gpt-test"),
                ):
                    code = run_cli(
                        [
                            str(markdown_path),
                            "--mode",
                            "llm",
                            "--provider",
                            "openai",
                        ]
                    )

            self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
