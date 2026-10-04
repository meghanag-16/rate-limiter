# src/rlimit/__main__.py
"""Allows `python -m rlimit ...` as an alternative to the installed
`rlimit` console script (see the [project.scripts] table in
pyproject.toml for the entry point that provides the latter)."""

from __future__ import annotations

from rlimit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
