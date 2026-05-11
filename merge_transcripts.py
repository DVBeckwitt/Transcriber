from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT_NAME = "merged_transcript.txt"
SKIP_FILENAMES = {
    "hf_token.txt",
    "hf_token.example.txt",
}
SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp_transcriber_temp",
    ".uv-venv",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "logs",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively merge .txt transcript files from a folder into one file.")
    parser.add_argument("folder", type=Path, help="Folder to scan recursively for .txt files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output file path. Defaults to {DEFAULT_OUTPUT_NAME} in the folder root.",
    )
    return parser.parse_args(argv)


def is_skipped_path(path: Path, root: Path, output_path: Path) -> bool:
    if path.resolve() == output_path.resolve():
        return True
    if path.name.casefold() in SKIP_FILENAMES:
        return True
    skipped_parts = (part.casefold() for part in path.relative_to(root).parts[:-1])
    return any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in skipped_parts)


def collect_transcript_files(root: Path, output_path: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*.txt"), key=lambda p: p.relative_to(root).as_posix().lower())
        if path.is_file() and not is_skipped_path(path, root, output_path)
    ]


def merge_transcript_files(root: Path, output_path: Path) -> tuple[int, Path]:
    root = root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    transcript_files = collect_transcript_files(root, output_path)
    with output_path.open("w", encoding="utf-8", newline="\n") as merged:
        merged.write(f"Merged transcript files from: {root}\n")
        merged.write(f"File count: {len(transcript_files)}\n\n")

        for index, path in enumerate(transcript_files, start=1):
            relative = path.relative_to(root).as_posix()
            merged.write(f"===== {index}. {relative} =====\n")
            text = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
            if text:
                merged.write(text)
                merged.write("\n")
            merged.write("\n")

    return len(transcript_files), output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.folder
    output_path = args.output or (root / DEFAULT_OUTPUT_NAME)

    try:
        count, resolved_output = merge_transcript_files(root, output_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Merged {count} transcript files into: {resolved_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
