"""Environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_env_fallback(env_path: Path, override: bool) -> bool:
    if not env_path.exists():
        return False

    loaded = False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded = True
    return loaded


def load_env(env_path: Path | None = None, override: bool = False) -> bool:
    """Load environment variables from .env using python-dotenv when available."""
    env_file_override = os.getenv("TREE_BUILDER_ENV_FILE", "").strip()
    if env_path is not None:
        target_path = env_path
    elif env_file_override:
        target_path = Path(env_file_override)
    else:
        target_path = Path.cwd() / ".env"

    try:
        from dotenv import load_dotenv
    except ImportError:
        return _load_env_fallback(target_path, override=override)

    return bool(load_dotenv(dotenv_path=target_path, override=override))
