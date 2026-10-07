# src/limivault/__main__.py
"""Allows `python -m limivault ...` as an alternative to the installed
`limivault` console script (see the [project.scripts] table in
pyproject.toml for the entry point that provides the latter)."""

from __future__ import annotations

from limivault.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
