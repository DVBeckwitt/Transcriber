# Changelog

## Unreleased - 2026-05-13

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
- Added `--speaker-labels` and `--no-speaker-labels` to control whether generated SRT files include diarization speaker labels.
- Added an interactive speaker-label prompt after the language and quality/fast prompts for one-off CLI runs.
- Added `live_translate_quality.bat` as an accuracy-first live Spanish-to-English launcher that uses cascade mode, writes English and bilingual logs, and prints live audio diagnostics.
- Added live troubleshooting docs for loopback diagnostics and 16 kHz signed 16-bit mono PCM chunk-size sanity checks.
- Added `tools/evaluate_live_transcript.py` for pure-Python CER/WER checks against live bilingual transcript files, covered by unit tests and mypy validation.
- Added German as a selectable language with `--lang de`, legacy `g`/`de` tokens, and the interactive language prompt.
- Added file English conversion settings with `--english-output-mode off|direct|post|auto`, interactive settings selection, `--post-translate-to-english`, source `.es.srt` / `.de.srt` preservation, English `.en.srt` output, and post-translation audit reports.
- Added local OpenAI-compatible server post-translation through the optional `translation-server` extra.

### Fixed
- Fixed watcher completed-file movement so missing `.srt` files leave media in place for retry.
- Fixed watcher completed-file movement so a failed `.srt` move rolls the media file back to the source folder.
- Fixed transcript merge filtering so Hugging Face token files are skipped case-insensitively.
- Fixed transcript merge scanning so generated/cache directories are skipped.
- Fixed explicit `--english-output-mode post` so missing translation infrastructure or translation failure marks the run failed instead of silently succeeding with source-only output.

### Changed
- Simplified config preset setup, temporary directory candidate de-duplication, SRT finalization, confidence cleanup, and transcript merge collection without changing public behavior.
- Simplified the speaker-label prompt/config control flow without changing interactive prompts, CLI flags, or diarization behavior.
- Changed the user-facing speaker label path so `--no-speaker-labels` skips diarization and Hugging Face token loading.
- Kept `--diarize` and `--no-diarize` as backward-compatible aliases; no deprecation or migration is required.
- Changed automatic Spanish live static prompts to match translation mode: direct mode asks for English translation, while cascade mode asks for Spanish ASR and explicitly avoids translation.
- Changed file post-translation to batch local server requests and reject non-loopback server URLs by default to avoid accidental transcript exfiltration.
- Updated the locked WhisperLiveKit package from `0.2.20.post1` to `0.2.21`.

### Removed
- Removed tracked generated artifacts: bytecode caches, sample media output, and sample WhisperX log output.
- Removed dead in-place Spanish SRT translation helpers and unused post-translation config/path fields left over from the previous helper path.

### Status
- Feature status: agentic legibility hardening is shipped as tooling and governance only.
- Feature status: SRT speaker label option is ready for release and covered by config, interactive prompt, watcher, summary, token cue, and fallback cue regression tests.
- Feature status: file English conversion is ready for release and covered by config, interactive prompt, post-translation, German auto mode, source preservation, failure handling, server request, server batching, URL safety, status, and preflight regression tests.
- Bug status: explicit post-translation failure handling is fixed; `post` fails when English output cannot be produced, while `auto` remains fallback-friendly.
- Refactor status: speaker-label prompt simplification is ready for release; no bug fix, user migration, deprecation, or CI contract change is required.
- Local quality gates pass: pytest with coverage, Ruff lint, Ruff format check, mypy, CLI help, uv lock sync, package build, dependency audit, pre-commit hooks, and secret scanning.
- Release type: local CLI/tooling update.
- Migration: none required; old diarization flags and `--translate-to-english` remain supported while clearer speaker-label and English-output mode flags are preferred.
- Rollback: revert the release commit.
