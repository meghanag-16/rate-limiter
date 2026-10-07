# Contributing

Thanks for considering a contribution to Limivault. The project supports Python 3.14 and uses Ruff, mypy, pytest, and Bandit for local checks.

## Development setup

```sh
python -m venv .venv
. .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Redis-backed and end-to-end tests use Docker and testcontainers. The in-memory test lane does not require Docker.

## Checks

Run these from the repository root:

```sh
ruff check src tests benchmarks
mypy src tests benchmarks
bandit -c pyproject.toml -r src
python -m coverage run -m pytest -m "not docker and not e2e"
python -m coverage report -m
pytest -m "docker or e2e"
```

Please keep changes focused, preserve sync/async API parity where applicable, and add or update tests for behavior changes. Redis Lua decisions should remain atomic and use Redis server time.
