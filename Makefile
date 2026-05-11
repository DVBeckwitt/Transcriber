.PHONY: sync test coverage lint format-check typecheck cli build audit hooks quality validate clean

sync:
	uv sync --locked

test: coverage

coverage:
	uv run pytest -q

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

cli:
	uv run python -m transcriber --help

build:
	uv build

audit:
	uv run --with pip-audit pip-audit .

hooks:
	uv run pre-commit run --all-files

quality: sync coverage lint format-check typecheck cli build

validate: quality audit hooks

clean:
	python -c "import shutil; from pathlib import Path; [shutil.rmtree(path, ignore_errors=True) for path in [Path('.uv-venv'), Path('build'), Path('dist'), Path('htmlcov'), Path('transcriber_local.egg-info'), Path('.pytest_cache'), Path('.mypy_cache'), Path('.ruff_cache')]]; [path.unlink(missing_ok=True) for path in [Path('.coverage'), Path('coverage.xml')]]"
