# tests/test_public_api.py
"""The package-root and rlimit.algorithms public API.

`from rlimit import FixedWindow` / `from rlimit.algorithms import
RedisGcraTokenBucket` work through lazy (PEP 562) exports, so these
tests check three things:

  - every name in `__all__` actually resolves, and the two packages
    agree on the object for every algorithm class;
  - unknown names still raise AttributeError (the lazy `__getattr__`
    does not swallow typos);
  - importing `rlimit` does NOT import any algorithm module. Each
    algorithm module binds its structlog logger at import time, so an
    eager import here would break the "configure_logging() first"
    ordering used by the CLI and the benchmarks. The last test runs
    `rlimit demo` in a subprocess and asserts no per-call DEBUG
    decision lines leak into its output.

The tests always exercise THIS checkout's src/ tree: pyproject.toml
puts src/ on pytest's import path, and the subprocess-based tests below
set PYTHONPATH to it explicitly. Without that, a stale non-editable
`pip install .` of an older rlimit sitting in the virtualenv's
site-packages would shadow the working tree and these tests would
silently check old code.

NO DOCKER REQUIRED (no Redis is contacted; the Lua classes are only
imported, never constructed).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import rlimit
import rlimit.algorithms

_SRC = Path(__file__).resolve().parents[1] / "src"


def _subprocess_env() -> dict[str, str]:
    """Environment that makes a child Python import rlimit from this
    checkout's src/ tree, ahead of anything installed in site-packages."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SRC), existing] if existing else [str(_SRC)]
    )
    return env


def test_rlimit_is_imported_from_this_checkout() -> None:
    assert Path(rlimit.__file__).resolve().is_relative_to(_SRC), (
        f"rlimit was imported from {rlimit.__file__}, not from {_SRC}. A stale "
        "non-editable install is probably shadowing the working tree; "
        "run `pip install -e .` in the project root."
    )


def test_every_name_in_rlimit_all_resolves() -> None:
    for name in rlimit.__all__:
        assert getattr(rlimit, name) is not None, name


def test_root_exports_supporting_public_api() -> None:
    expected = {
        "BackendUnavailableError",
        "UnsatisfiableRequestError",
        "AllowedEvent",
        "DeniedEvent",
        "BackendErrorEvent",
        "MetricsHook",
        "NoOpMetricsHook",
        "rate_limit",
        "KeyedLimiter",
        "async_rate_limit",
        "AsyncKeyedLimiter",
        "SimulationClock",
        "SimulationRecorder",
        "TrafficStep",
        "run_simulation",
        "async_run_simulation",
        "LuaSimulationRecorder",
        "run_lua_simulation",
    }
    assert expected <= set(rlimit.__all__)
    for name in expected:
        assert getattr(rlimit, name) is not None


def test_every_name_in_algorithms_all_resolves() -> None:
    for name in rlimit.algorithms.__all__:
        assert getattr(rlimit.algorithms, name) is not None, name


def test_algorithm_classes_are_the_same_object_from_both_packages() -> None:
    for name in rlimit.algorithms.__all__:
        assert getattr(rlimit, name) is getattr(rlimit.algorithms, name), name


def test_all_has_no_duplicates() -> None:
    assert len(rlimit.__all__) == len(set(rlimit.__all__))
    assert len(rlimit.algorithms.__all__) == len(set(rlimit.algorithms.__all__))


def test_there_are_24_algorithm_classes() -> None:
    """6 algorithms x {in-memory, Redis Lua/GCRA} x {sync, async}."""
    assert len(rlimit.algorithms.__all__) == 24


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        rlimit.definitely_not_a_real_name  # noqa: B018
    with pytest.raises(AttributeError):
        rlimit.algorithms.definitely_not_a_real_name  # noqa: B018


def test_dir_lists_lazy_names() -> None:
    assert "FixedWindow" in dir(rlimit)
    assert "RedisGcraTokenBucket" in dir(rlimit.algorithms)


def test_import_rlimit_does_not_import_any_algorithm_module() -> None:
    code = (
        "import sys, rlimit, rlimit.logging, rlimit.algorithms\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.startswith('rlimit.algorithms.')\n"
        ")\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr


def test_cli_demo_output_has_no_debug_decision_lines() -> None:
    """Regression guard for the logging-order trap: `rlimit demo` must
    print only its own ALLOWED/DENIED lines, not one structlog DEBUG
    line per allow() call."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlimit",
            "demo",
            "--algorithm",
            "fixed_window",
            "--param1",
            "2",
            "--param2",
            "60",
            "--requests",
            "4",
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "ALLOWED" in combined
    assert "DENIED" in combined
    assert "fixed_window_decision" not in combined
