# ADR-001: Agentic Legibility and CI Quality Gates

## Status
Accepted

## Date
2026-05-11

## Context
The repository is a local WhisperX transcription launcher. Before this change, future agents and contributors had limited machine-readable project context and no committed quality gate that consistently ran tests, coverage, lint, type checking, CLI smoke checks, package build validation, dependency audit, hook checks, and secret scanning.

The repository also had generated artifacts committed, which made diffs noisier and increased the chance of committing local outputs.

## Decision
Use `pyproject.toml` plus `uv.lock` as the canonical local and CI dependency/tooling entrypoint. Add a `Makefile` as the command index for agents and contributors. Add GitHub Actions CI to run pytest with coverage, Ruff lint, Ruff format check, mypy, CLI help, `uv build`, dependency audit, pre-commit hooks, and Gitleaks.

Keep generated runtime artifacts out of source control through `.gitignore`, and document agent workflow, boundaries, current ship status, and rollback notes in `AGENTS.md`.

## Alternatives Considered

### Keep ad hoc local commands only
- Pros: No new CI configuration.
- Cons: Agents and contributors would continue to rely on local memory and manual convention.
- Rejected: The project needs repeatable checks before merge.

### Use requirements files only
- Pros: Familiar Python workflow.
- Cons: Does not capture grouped dev tooling as cleanly and does not provide the same lockfile workflow used by the new CI.
- Rejected: `uv` gives a single repeatable path for local and CI validation.

### Leave generated artifacts tracked
- Pros: No cleanup required.
- Cons: Bytecode, logs, and sample outputs create noisy diffs and are not source.
- Rejected: Generated artifacts should be recreated locally, not versioned.

## Consequences
- Contributors and agents have one documented validation path.
- CI blocks regressions in tests, coverage, style, typing, CLI startup, packaging, dependency audit, hook hygiene, and secret scanning.
- Reverting the release commit restores the previous behavior if the local CLI release needs rollback.
- Future tooling changes should update this ADR only if the core quality-gate decision changes.
