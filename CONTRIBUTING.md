# Contributing

This is a local CLI project. Keep changes small, reversible, and verified before handoff.

## Workflow

1. Work from a short-lived branch. Use `codex/<short-description>` for agent work.
2. Keep each change focused on one concern: runtime behavior, tests, docs, or tooling.
3. Prefer modifying existing files over adding new files. Add a file only when it creates a stable repo contract, such as CI, security policy, or architecture docs.
4. Update `README.md`, `AGENTS.md`, `CHANGELOG.md`, or `docs/` when behavior, commands, architecture, or release status changes.
5. Use a clear commit message:

```text
<type>: <short reason>
```

Use `feat`, `fix`, `docs`, `test`, `refactor`, or `chore`.

## Required Validation

Run the full gate before review:

```powershell
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m transcriber --help
uv build
```

If OneDrive or another process locks the default `.venv`, keep this set for the shell session:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".uv-venv"
```

## Review Checklist

- Source behavior is covered by tests when logic changes.
- CLI options remain backward compatible unless the change explicitly deprecates them.
- Docs describe user-visible behavior and rollback notes.
- Generated files are absent from the diff.
- Secrets are absent from the diff.
- CI passes before merge.

## Generated Files And Secrets

Never commit:

- `HF_TOKEN.txt` or `.env` files.
- `logs/`, `*.log`, `*.srt`, or `*_llm.txt`.
- `build/`, `dist/`, `*.egg-info/`, virtual environments, or bytecode caches.
- Local media inputs or transcription outputs.

Keep `HF_TOKEN.example.txt` committed as the safe template.

## Rollback

This project ships as local CLI source. Roll back a bad release with `git revert <commit>`, then rerun the full validation gate.
