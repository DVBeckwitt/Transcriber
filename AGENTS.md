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
- `tools/evaluate_live_transcript.py`: Pure-Python CER/WER evaluator for committed live bilingual transcript files.
- `tests/test_helpers.py`: Unit tests for CLI config, watcher policy, translation helpers, confidence cleanup, and transcript merging.
- `tests/test_live_cli.py`, `tests/test_live_audio.py`, `tests/test_live_wlk_messages.py`, `tests/test_live_eval.py`: Live-mode unit coverage for CLI config, launchers, PCM/audio helpers, WLK protocol parsing, transcript semantics, and evaluator metrics.
- `README.md`: User setup, runtime behavior, CLI examples, and troubleshooting.
- `CONTRIBUTING.md`: Branch, review, validation, and rollback workflow.
- `SECURITY.md`: Vulnerability reporting, secret handling, and dependency audit policy.
- `docs/architecture.md`: Runtime flow, module boundaries, stable interfaces, and change guide.
- `docs/decisions/`: Accepted architecture and process decisions.
- `*.bat`: Windows launchers for one-off transcription, watcher processes, and live low-latency or quality caption modes.
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

- Status date: 2026-06-08.
- Optional live caption mode is implemented for Windows PC speaker output through WhisperLiveKit and ready for local manual validation. Live dependencies stay behind the `live` extra and require Python 3.11+; the base package remains Python 3.10-compatible. Unit tests cover CLI dispatch, launcher contracts, PCM conversion, audio diagnostics, queue drop policy, WLK message parsing, direct/cascade transcript semantics, bilingual transcript formatting, quality preset flags, window formatting, missing dependency diagnostics, executable resolution, WLK startup cleanup, and live transcript evaluator metrics.
- Live launcher status: `live_translate.bat` remains the low-latency path. It runs from the repository root, prefers `VIRTUAL_ENV`, repo-local `.uv-venv`, repo-local `.venv`, then `%USERPROFILE%\.venv`, and writes committed direct-mode English captions to `logs\live_english_transcript.txt` by default. `live_translate_quality.bat` is the accuracy-first path. It uses quality/cascade mode, medium model by default, 750 ms chunks, frame threshold 45, 5 beams, Faster-Whisper, beam decoding, CTranslate2 NLLB, valid 1.0-45 second audio buffer bounds, optional local glossary injection when `transcriber_glossary.txt` exists, English and bilingual transcript logs, and live audio diagnostics. The direct-mode bilingual log defect is fixed by rejecting `--live-save-bilingual-transcript` unless cascade mode is selected.
- Live prompt status: automatic Spanish static prompts are mode-aware. Direct mode asks WhisperLiveKit for natural English translation; cascade mode asks for accurate Spanish ASR and explicitly says not to translate. Explicit `--live-static-prompt` still overrides the automatic prompts.
- Live evaluator status: `tools/evaluate_live_transcript.py` is a source-repo developer tool for deterministic CER/WER checks against committed bilingual transcript text. It has no external dependencies, is covered by unit tests, and is included in mypy validation. It is not a packaged console script.
- Dependency status: `uv.lock` is updated to WhisperLiveKit `0.2.21`; `pyproject.toml` still allows `whisperlivekit[translation]>=0.2.20.post1` behind the optional `live` extra. Local file post-translation uses Python standard-library HTTP and does not require an HTTP client extra; automatic server startup expects a `vllm` executable in the active Python environment, on `PATH`, or in the default WSL2 distribution on Windows.
- Agentic legibility hardening is ready for release: reproducible uv environment, `Makefile` validation, coverage gate, CI quality gates, agent guide, governance docs, security policy, architecture docs, and generated-artifact cleanup are in place.
- Watcher move failure handling is fixed and covered by regression tests. Missing or locked `.srt` files no longer strand moved media without a retryable source.
- Transcript merge secret filtering is fixed and covered by regression tests. Token files are skipped case-insensitively, and generated/cache directories are excluded from recursive scans.
- Speaker label option is ready for release and covered by regression tests. Interactive one-off runs prompt for speaker labels after language and quality/fast mode. `--speaker-labels` / `--no-speaker-labels` control SRT `SPEAKER_00:` labels; disabling labels skips diarization and Hugging Face token loading. Existing `--diarize` / `--no-diarize` remain backward-compatible aliases.
- German language selection is ready for release. `--lang de`, legacy `g`/`de` tokens, and the interactive language prompt select German transcription without enabling Spanish post-translation by default.
- File English conversion is ready for release. `english_output_mode` is selectable through CLI and interactive settings, supports `off` / `direct` / `post` / `auto`, preserves source Spanish/German SRTs, writes English `.en.srt` plus compatibility `.srt` for successful post-translation, logs the selected mode in status, auto-starts and stops a local vLLM server when no post-translation URL is supplied, and rejects non-loopback translation server URLs by default. Local post-translation defaults to one cue per request and 1024 max generated tokens; `--translation-batch-size` and `--translation-max-new-tokens` are exposed for tuning. Explicit `post` fails when server startup, translation, or the post-translation quality gate fails; `auto` warns and keeps source output when post-translation is unavailable or rejected. The default local model is `utter-project/EuroLLM-1.7B-Instruct`; WSL2 vLLM serving that model through the OpenAI-compatible localhost API has passed local smoke testing. Malformed model indexes recover by response order, malformed multi-cue batches retry as single-cue requests, residual Spanish/German or dropped-content cues retry once individually, and recovery/quality metadata is written to the translation report without transcript text.
- Code simplification is complete for the current refactor slice. The speaker-label prompt/config flow, live static-prompt selection, live translation-mode helper/parser cleanup, and evaluator parser cleanup were simplified without changing public CLI behavior, watcher policy, live direct/cascade semantics, merge output contracts, or evaluator output format.
- Dependency audit is strict in CI through `pip-audit`; secret scanning is enforced with Gitleaks; Dependabot is configured for GitHub Actions and uv.
- Coverage is enforced through pytest-cov. Start at the current baseline and ratchet up only when tests improve.
- No deprecation or migration is required for this release; old diarization flags and `--translate-to-english` remain supported, the latency live launcher remains unchanged, file post-translation is additive, and the quality launcher/evaluator are additive optional paths.
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
