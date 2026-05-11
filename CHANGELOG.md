# Changelog

## Unreleased - 2026-05-11

### Added
- Added project quality tooling with `pyproject.toml`, `uv.lock`, Ruff, mypy, pytest configuration, package build validation, and GitHub Actions CI.
- Added `AGENTS.md` with repo map, validation commands, boundaries, current ship status, and rollback notes.
- Added watcher support for the Escuela video folder policy from the recordings watcher.
- Added `merge_transcripts.py` for recursive transcript text merging.

### Fixed
- Fixed watcher completed-file movement so missing `.srt` files leave media in place for retry.
- Fixed watcher completed-file movement so a failed `.srt` move rolls the media file back to the source folder.
- Fixed transcript merge filtering so Hugging Face token files are skipped case-insensitively.
- Fixed transcript merge scanning so generated/cache directories are skipped.

### Removed
- Removed tracked generated artifacts: bytecode caches, sample media output, and sample WhisperX log output.

### Status
- Local quality gates pass: pytest, Ruff lint, Ruff format check, mypy, CLI help, uv lock sync, and package build.
- Release type: local CLI/tooling update.
- Rollback: revert the release commit.
