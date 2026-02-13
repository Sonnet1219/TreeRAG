import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
