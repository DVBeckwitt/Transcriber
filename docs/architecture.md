# Architecture

Transcriber is a local Python CLI around WhisperX. Most behavior lives in `transcriber/__main__.py`; supporting docs and tests explain the intended boundaries until the module is split further.

## Runtime Flow

```text
CLI args / batch files
  -> parse_args
  -> build_config
  -> resolve input or watch targets
  -> transcribe_file
  -> WhisperX / cleanup
  -> output files next to source media
```

Watch mode follows a polling flow:

```text
build_watch_targets
  -> run_watch_loop
  -> iter_watch_candidates
  -> needs_transcription
  -> transcribe_file
  -> move_completed_outputs_for_target
  -> move_completed_watch_outputs
```

Live mode is a separate runtime path:

```text
CLI live args / live_translate*.bat
  -> transcriber.live.build_live_config
  -> transcriber.live_audio WASAPI loopback capture
  -> downmixed/resampled 16 kHz mono signed 16-bit PCM chunks
  -> local WhisperLiveKit subprocess with configured live quality flags
  -> ws://127.0.0.1:<port>/asr?language=es&mode=full
  -> transcriber.live_wlk.CaptionState
  -> transcriber.live_window caption popup and optional text transcripts
```

Live transcript evaluation is a source-repo developer tool:

```text
reference Spanish text + committed bilingual transcript
  -> tools/evaluate_live_transcript.py
  -> local Levenshtein CER/WER metrics
```

## Module Map

- `transcriber/__main__.py`: CLI surface, `RunConfig`, output path contracts, WhisperX execution, subtitle cleanup, watch mode, and completed-file movement.
- `transcriber/translation.py`: legacy local-server post-translation contracts and helpers retained outside the current command surface. The file/watch pipeline no longer calls this module for English conversion; keep heavy model/server imports lazy while this code remains.
- `transcriber/live.py`: live-mode configuration and coordinator. It starts/stops capture, WhisperLiveKit streaming, caption updates, translation-mode validation, and optional English-only or bilingual transcript saving.
- `transcriber/live_audio.py`: Windows WASAPI loopback device discovery, loopback test WAV writing, and PCM conversion to 16 kHz mono signed 16-bit little-endian audio.
- `transcriber/live_wlk.py`: WhisperLiveKit subprocess command construction, readiness polling, WebSocket protocol handling, and caption-state extraction.
- `transcriber/live_window.py`: Tkinter always-on-top caption window fed through a thread-safe queue.
- `merge_transcripts.py`: standalone text utility for recursively merging transcript `.txt` files while skipping token files and generated/cache directories.
- `tools/evaluate_live_transcript.py`: source-repo developer tool for deterministic CER/WER checks against committed live bilingual transcript text. It is not a packaged console script.
- `tests/test_helpers.py`: unit tests for config, prompts, watcher policy, file movement failure handling, WhisperX direct translation orchestration, confidence cleanup, and transcript merging.
- `tests/test_translation.py`: unit tests for legacy file post-translation helpers, report metadata, local server batching, URL safety, and response validation.
- `*.bat`: Windows launchers. They should stay thin wrappers around `python -m transcriber`; `live_translate.bat` is the latency/direct launcher and `live_translate_quality.bat` is the accuracy/cascade launcher.
- `.github/workflows/ci.yml`: CI contract for tests with coverage, lint, format, type checking, CLI startup, package build, dependency audit, pre-commit hooks, and secret scanning.
- `docs/decisions/`: ADRs for decisions that future agents should not re-decide from scratch.

## Stable Interfaces

- CLI options documented in `README.md` are user-facing. Preserve them unless a change explicitly deprecates behavior.
- `RunConfig` is the internal configuration object passed through transcription and watcher flows. `speaker_labels` records the user-facing SRT display choice; `diarize` records whether WhisperX diarization should run.
- `OutputPaths` defines where compatibility `.srt`, legacy source/English/translation-report paths, `*_llm.txt`, logs, and lock files are expected.
- File English output mode is explicit. `off` keeps source-language output only. `direct` uses WhisperX translation and is the only visible English conversion path. Legacy `post` / `auto` values and old post-translation server flags are hidden from help and normalize to direct translation when parsed for old launchers; they do not start vLLM, WSL2, or a local OpenAI-compatible server.
- `WatchTarget` defines per-folder watcher policy, including allowed extensions, destination moves, and rename strategy.
- Live mode is only entered through `--live`, `--live-list-devices`, or `--live-loopback-test`. It does not call `transcribe_file`, WhisperX alignment, PyAnnote diarization, SRT cue generation, or legacy file post-translation helpers.
- `LiveConfig` is the live-mode boundary between CLI parsing and runtime orchestration. New live quality controls should be added there first, then passed through to `live_wlk.build_wlk_command` without changing file/watch `RunConfig`.
- `CaptionState` is the live UI and live transcript contract. Full-mode WhisperLiveKit updates replace the partial caption line instead of appending it. In direct mode, committed pairs intentionally have empty source text because WLK committed `text` is already English; in cascade mode, committed pairs preserve each line's source-language text with its English translation.
- `tools/evaluate_live_transcript.py` parses the committed Spanish bilingual transcript text format written by cascade mode: numbered entries with `ES:` and `EN:` lines. Keep it dependency-free and deterministic so it remains usable without WhisperLiveKit, audio hardware, network, GPU, or model downloads.
- The CI workflow is a repository contract. Update `README.md`, `AGENTS.md`, and `CONTRIBUTING.md` when changing validation commands.

## Release Notes

- SRT speaker label control is a user-facing CLI feature, not a pipeline migration. Interactive one-off runs prompt for speaker labels after language and quality/fast mode. `--speaker-labels` and `--no-speaker-labels` are the preferred flag names; `--diarize` and `--no-diarize` remain supported aliases.
- Disabling speaker labels skips diarization, Hugging Face token loading, speaker smoothing, and `SPEAKER_00:` rendering while preserving subtitle timing, cleanup, translation, watcher, and movement behavior.
- The speaker-label prompt/config simplification is an internal refactor only. It does not change CLI options, prompt wording, defaults, watcher behavior, CI gates, or migration posture.
- Live mode is an optional Windows-only feature path with Python 3.11+ runtime guard and optional `live` dependencies. Base package compatibility remains Python 3.10+. The mode is unit-tested without Windows audio hardware or model downloads; manual ship validation still requires `uv sync --extra live` and a WASAPI loopback device. The launcher resolves WhisperLiveKit's `wlk` or `whisperlivekit-server` executable from the active Python environment's Scripts directory before falling back to `PATH`, starts it with the configured live backend policy, and writes committed direct-mode English captions to `logs\live_english_transcript.txt` by default.
- Live translation mode is explicit. `direct` uses WhisperLiveKit `--direct-english-translation` and rejects `--live-save-bilingual-transcript`; `cascade` uses `--target-language en` and allows real Spanish/English transcript pairs.
- Live quality controls are additive. The latency preset keeps direct mode, model `small`, 250 ms chunks, greedy decoding, and a drop-oldest bounded queue policy. The quality preset uses cascade mode, model `medium` unless overridden, 500 ms chunks, Faster-Whisper backend, beam decoding, CTranslate2 NLLB translation, validated audio buffer lengths, and an unbounded no-intentional-drop queue policy. The quality launcher chooses a slower accuracy profile on top of that preset: 750 ms chunks, frame threshold 45, 5 beams, 1.0-45 second audio buffer bounds, diagnostics, and optional local glossary injection.
- Live static prompts are mode-aware for Spanish quality mode. Direct mode prompts for natural English translation; cascade mode prompts for Spanish ASR and explicitly says not to translate because WLK handles English output through `--target-language en`.
- Live audio diagnostics are observational only. They report sample rate, channels, output chunk bytes, RMS, peak, queue depth, dropped chunks, estimated queue delay, and WhisperLiveKit lag without changing transcript semantics.
- The live translation-mode helper/parser cleanup is an internal refactor only. It does not change CLI options, defaults, direct/cascade transcript semantics, WLK command flags, CI gates, or migration posture.
- Live-mode error status: missing optional live dependencies now report a clear install message instead of a traceback, and WLK subprocess startup failures terminate the child process before returning an error.
- Live-mode rollout status: additive local beta. Startup smoke has passed with the `live` extra installed, but full Windows loopback audio validation is still pending. No existing file transcription/watch behavior is migrated or deprecated, and `live_translate.bat` remains the fast path.
- File English conversion rollout status: the local vLLM/WSL2 post-translation command path is retired from the visible CLI. No existing direct WhisperX translation path is deprecated; `--translate-to-english` remains the compatibility alias for `direct`. Legacy `post` / `auto` inputs remain parseable only to avoid breaking saved commands, and now resolve to direct WhisperX translation.
- Rollback is git-based: revert the release commit and rerun the full validation gate.

## Change Guide

- CLI argument or config behavior: update `parse_args`, `build_config`, README CLI options, and tests.
- File English conversion behavior: update `build_config`, `run_whisperx_direct`, the finalization branch in `transcribe_file`, README output-mode docs, and helper tests. Do not reintroduce heavyweight model/server imports at module import time.
- Live-mode behavior: update `transcriber/live.py`, `transcriber/live_audio.py`, `transcriber/live_wlk.py`, `transcriber/live_window.py`, README live commands, and live tests.
- Live transcript evaluation: update `tools/evaluate_live_transcript.py` and `tests/test_live_eval.py`.
- Transcription execution: update `transcribe_file`, WhisperX helpers, and tests that mock execution.
- Legacy translation helper behavior: update `transcriber/translation.py` and `tests/test_translation.py`. This module is not part of the current file/watch command surface.
- Watch folder policy: update `build_watch_targets`, `run_watch_loop`, watcher docs, and tests.
- Completed-file movement: update `move_completed_watch_outputs` and regression tests for missing or locked `.srt` files.
- Transcript merging: update `merge_transcripts.py` and merge utility tests.
- Quality gates: update `Makefile`, `.github/workflows/ci.yml`, `pyproject.toml`, `AGENTS.md`, `CONTRIBUTING.md`, and ADRs when the contract changes.

## Generated And Local Artifacts

Generated files are intentionally outside source control:

- `logs/`, `*.log`, `*.srt`, `*_llm.txt`
- `.tmp_transcriber_temp/`
- `build/`, `dist/`, `*.egg-info/`
- `.venv/`, `.uv-venv/`, caches, and bytecode
- `HF_TOKEN.txt`, `.env`, and local media inputs

If a future change needs to commit a generated artifact, document why in `CHANGELOG.md` and add an ADR if the decision is durable.
