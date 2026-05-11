# Architecture

Transcriber is a local Python CLI around WhisperX. Most behavior lives in `transcriber/__main__.py`; supporting docs and tests explain the intended boundaries until the module is split further.

## Runtime Flow

```text
CLI args / batch files
  -> parse_args
  -> build_config
  -> resolve input or watch targets
  -> transcribe_file
  -> WhisperX / translation / cleanup
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

## Module Map

- `transcriber/__main__.py`: CLI surface, `RunConfig`, output path contracts, WhisperX execution, subtitle cleanup, Spanish-to-English translation, watch mode, and completed-file movement.
- `merge_transcripts.py`: standalone text utility for recursively merging transcript `.txt` files while skipping token files and generated/cache directories.
- `tests/test_helpers.py`: unit tests for config, prompts, watcher policy, file movement failure handling, translation helpers, confidence cleanup, and transcript merging.
- `*.bat`: Windows launchers. They should stay thin wrappers around `python -m transcriber`.
- `.github/workflows/ci.yml`: CI contract for tests with coverage, lint, format, type checking, CLI startup, package build, dependency audit, pre-commit hooks, and secret scanning.
- `docs/decisions/`: ADRs for decisions that future agents should not re-decide from scratch.

## Stable Interfaces

- CLI options documented in `README.md` are user-facing. Preserve them unless a change explicitly deprecates behavior.
- `RunConfig` is the internal configuration object passed through transcription and watcher flows.
- `OutputPaths` defines where `.srt`, `*_llm.txt`, logs, and lock files are expected.
- `WatchTarget` defines per-folder watcher policy, including allowed extensions, destination moves, and rename strategy.
- The CI workflow is a repository contract. Update `README.md`, `AGENTS.md`, and `CONTRIBUTING.md` when changing validation commands.

## Change Guide

- CLI argument or config behavior: update `parse_args`, `build_config`, README CLI options, and tests.
- Transcription execution: update `transcribe_file`, WhisperX helpers, and tests that mock execution.
- Translation behavior: update translation helpers and tests around prompts/context/beam settings.
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
