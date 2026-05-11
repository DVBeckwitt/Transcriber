## Summary

-

## Validation

- [ ] `uv sync --locked`
- [ ] `uv run pytest -q`
- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run mypy`
- [ ] `uv run python -m transcriber --help`
- [ ] `uv build`

## Change Review

- [ ] Tests added or updated for behavior changes
- [ ] Docs updated for user-visible behavior, commands, architecture, or release status
- [ ] No secrets, local media, logs, subtitles, bytecode, virtualenvs, or build artifacts included
- [ ] CLI behavior remains backward compatible, or breaking change is explicitly documented

## Risk And Rollback

- Risk:
- Rollback:
