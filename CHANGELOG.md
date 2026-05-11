# Changelog

## Unreleased - 2026-05-11

### Added
- Added project quality tooling with `pyproject.toml`, `uv.lock`, Ruff, mypy, pytest configuration, package build validation, and GitHub Actions CI.
- Added `AGENTS.md` with repo map, validation commands, boundaries, current ship status, and rollback notes.
- Added contributor workflow, pull request checklist, issue templates, CODEOWNERS, security policy, and Dependabot configuration.
- Added `Makefile` validation targets for setup, tests, coverage, lint, formatting, type checking, CLI smoke checks, package build, dependency audit, and hooks.
- Added strict Python dependency audit to CI with `pip-audit`.
- Added pytest coverage enforcement, pre-commit hooks, and Gitleaks secret scanning.
- Added architecture map and watcher file movement ADR.
- Added watcher support for the Escuela video folder policy from the recordings watcher.
- Added `merge_transcripts.py` for recursive transcript text merging.

### Fixed
- Fixed watcher completed-file movement so missing `.srt` files leave media in place for retry.
- Fixed watcher completed-file movement so a failed `.srt` move rolls the media file back to the source folder.
- Fixed transcript merge filtering so Hugging Face token files are skipped case-insensitively.
- Fixed transcript merge scanning so generated/cache directories are skipped.

### Changed
- Simplified config preset setup, temporary directory candidate de-duplication, SRT finalization, confidence cleanup, and transcript merge collection without changing public behavior.

### Removed
- Removed tracked generated artifacts: bytecode caches, sample media output, and sample WhisperX log output.

### Status
- Feature status: agentic legibility hardening is shipped as tooling and governance only.
- Local quality gates pass: pytest with coverage, Ruff lint, Ruff format check, mypy, CLI help, uv lock sync, package build, dependency audit, pre-commit hooks, and secret scanning.
- Release type: local CLI/tooling update.
- Migration: none required; public CLI behavior and output contracts are unchanged.
- Rollback: revert the release commit.
