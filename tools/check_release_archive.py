from __future__ import annotations

import fnmatch
import sys
import zipfile
from pathlib import Path

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    ".uv-venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "logs",
    "transcriber_local.egg-info",
    ".tmp_transcriber_temp",
}
FORBIDDEN_NAMES = {
    ".coverage",
    "coverage.xml",
    "HF_TOKEN.txt",
    "hf_token.txt",
}
FORBIDDEN_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.whl",
    "*.tar.gz",
    "*.transcribing.lock",
    "*.transcription.json",
    "*.srt",
    "*_llm.txt",
)


def is_forbidden(name: str) -> bool:
    path = Path(name)
    if set(path.parts) & FORBIDDEN_PARTS:
        return True
    if path.name in FORBIDDEN_NAMES:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in FORBIDDEN_PATTERNS)


def main(argv: list[str]) -> int:
    archive_path = Path(argv[1]) if len(argv) > 1 else Path("dist/transcriber-local-clean.zip")
    if not archive_path.exists():
        print(f"Archive not found: {archive_path}", file=sys.stderr)
        return 2
    with zipfile.ZipFile(archive_path) as archive:
        bad = [name for name in archive.namelist() if is_forbidden(name)]
    if bad:
        print("Forbidden files in archive:", file=sys.stderr)
        for name in bad:
            print(f"  {name}", file=sys.stderr)
        return 1
    print(f"Archive passed hygiene check: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
