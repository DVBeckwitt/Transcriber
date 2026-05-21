from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "transcriber-local-clean.zip"

EXCLUDED_PARTS = {
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
EXCLUDED_NAMES = {
    ".coverage",
    "coverage.xml",
    "HF_TOKEN.txt",
    "hf_token.txt",
}
EXCLUDED_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.srt",
    "*_llm.txt",
    "*.transcribing.lock",
    "*.transcription.json",
    "*.whl",
    "*.tar.gz",
)


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDED_PARTS:
        return True
    if path.name in EXCLUDED_NAMES:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_PATTERNS)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_dir() or should_exclude(path):
                continue
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
