# Transcriber Agent Guide

## Purpose

This repository is a local WhisperX launcher for audio/video transcription, subtitle cleanup, optional Spanish-to-English translation, and folder watch mode.

## Repo Map

- `transcriber/__main__.py`: CLI, transcription flow, subtitle handling, translation helpers, watch mode, and file movement policy.
- `transcriber/live.py`: Optional live-mode coordinator for Windows system-audio captions through WhisperLiveKit.
- `transcriber/live_audio.py`: WASAPI loopback discovery/capture, loopback WAV tests, and 16 kHz mono PCM conversion.
- `transcriber/live_wlk.py`: WhisperLiveKit subprocess/WebSocket integration and caption message parsing.
- `transcriber/live_window.py`: Tkinter always-on-top caption popup.
- `merge_transcripts.py`: Utility for recursively merging transcript text files.
- `tests/test_helpers.py`: Unit tests for CLI config, watcher policy, translation helpers, confidence cleanup, and transcript merging.
- `README.md`: User setup, runtime behavior, CLI examples, and troubleshooting.
- `CONTRIBUTING.md`: Branch, review, validation, and rollback workflow.
- `SECURITY.md`: Vulnerability reporting, secret handling, and dependency audit policy.
- `docs/architecture.md`: Runtime flow, module boundaries, stable interfaces, and change guide.
- `docs/decisions/`: Accepted architecture and process decisions.
- `*.bat`: Windows launchers for one-off transcription and watcher processes.
- `logs/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.uv-venv/`: Generated local artifacts; do not edit or commit them.

## Setup

Prefer uv:

```powershell
uv sync
```

If the default `.venv` is locked on Windows, use a disposable local environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".uv-venv"
uv sync
```

Fallback without uv:

```powershell
python -m pip install -e .
python -m pip install pytest ruff mypy
```

## Validation Commands

Run these before handing off changes:

```powershell
make validate
```

If `make` is unavailable, run the component commands:

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m transcriber --help
uv build
uv run --with pip-audit pip-audit .
uv run pre-commit run --all-files
```

If using `$env:UV_PROJECT_ENVIRONMENT = ".uv-venv"`, keep that variable set for every `uv run` command in the same shell.

## Current Ship Status

- Status date: 2026-05-15.
- Optional live caption mode is implemented for Windows PC speaker output through WhisperLiveKit and ready for local manual validation. Live dependencies stay behind the `live` extra and require Python 3.11+; the base package remains Python 3.10-compatible. Unit tests cover CLI dispatch, PCM conversion, WLK message parsing, direct/cascade transcript semantics, bilingual transcript formatting, window formatting, missing dependency diagnostics, executable resolution, and WLK startup cleanup.
- Live launcher status: `live_translate.bat` runs from the repository root and prefers `VIRTUAL_ENV`, repo-local `.uv-venv`, repo-local `.venv`, then `%USERPROFILE%\.venv`. The Python integration resolves that active environment's WhisperLiveKit executable before PATH, starts WLK with the LocalAgreement backend policy, avoids the observed SimulStreaming `StorageView` encoder crash, and writes committed direct-mode English captions to `logs\live_english_transcript.txt` by default. The direct-mode bilingual log defect is fixed by rejecting `--live-save-bilingual-transcript` unless cascade mode is selected. The previous missing-live-extra traceback path now exits with a clear message. Startup smoke has passed with the `live` extra installed; full Windows loopback audio validation is still pending.
- Agentic legibility hardening is ready for release: reproducible uv environment, `Makefile` validation, coverage gate, CI quality gates, agent guide, governance docs, security policy, architecture docs, and generated-artifact cleanup are in place.
- Watcher move failure handling is fixed and covered by regression tests. Missing or locked `.srt` files no longer strand moved media without a retryable source.
- Transcript merge secret filtering is fixed and covered by regression tests. Token files are skipped case-insensitively, and generated/cache directories are excluded from recursive scans.
- Speaker label option is ready for release and covered by regression tests. Interactive one-off runs prompt for speaker labels after language and quality/fast mode. `--speaker-labels` / `--no-speaker-labels` control SRT `SPEAKER_00:` labels; disabling labels skips diarization and Hugging Face token loading. Existing `--diarize` / `--no-diarize` remain backward-compatible aliases.
- Code simplification is complete for the current refactor slice. The speaker-label prompt/config flow was simplified without changing public CLI behavior, watcher policy, or merge output contracts.
- Dependency audit is strict in CI through `pip-audit`; secret scanning is enforced with Gitleaks; Dependabot is configured for GitHub Actions and uv.
- Coverage is enforced through pytest-cov. Start at the current baseline and ratchet up only when tests improve.
- No deprecation or migration is required for this release; old diarization flags remain supported and live mode is an additive optional path.
- Rollback path is git-based: revert the release commit to restore the previous behavior and docs.

## Conventions

- Keep runtime behavior stable unless the task explicitly asks for transcription, watcher, translation, or subtitle behavior changes.
- Prefer small edits to existing files over new abstractions.
- Keep CLI options backward compatible; visible CLI behavior is a public interface.
- Keep live mode isolated from the file transcription/watch pipeline except for CLI entry points and shared prompt/glossary parsing.
- Add or update tests when changing logic.
- Do not add GPU, WhisperX model download, ffmpeg, or Hugging Face token requirements to unit tests or CI.
- Do not add WhisperLiveKit, PyAudioWPatch, or Windows audio hardware requirements to base install or unit tests.
- Update `CONTRIBUTING.md`, `SECURITY.md`, or `docs/architecture.md` when their contracts change.

## Boundaries

- Always run the validation commands after code changes.
- Always keep `HF_TOKEN.txt` local and ignored.
- Ask first before changing watcher destination paths, episode numbering policy, or batch launcher behavior.
- Never commit local tokens, generated logs, bytecode caches, virtual environments, or transcription outputs.
