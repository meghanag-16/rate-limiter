"""CLI unit checks run in child processes to isolate logging setup.

Importing ``rlimit.cli`` configures structlog for the command-line app.
Keep that import out of pytest's process: cached loggers would otherwise
interfere with ``structlog.testing.capture_logs`` in other test modules.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    source_dir = str(_PROJECT_ROOT / "src")
    previous_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [source_dir, previous_path] if previous_path else [source_dir]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_direct_library_use_is_quiet_by_default() -> None:
    result = _run_isolated(
        "from rlimit import FixedWindow; "
        "limiter = FixedWindow(limit=1, period=60); "
        "limiter.allow('quiet-check'); limiter.allow('quiet-check')"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_demo_dispatch_runs_without_parent_process_import() -> None:
    code = """
from rlimit.cli import main
raise SystemExit(main([
    "demo", "--algorithm", "fixed_window", "--param1", "2",
    "--param2", "60", "--requests", "4",
]))
"""
    result = _run_isolated(code)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("ALLOWED") == 2
    assert result.stdout.count("DENIED") == 2


@pytest.mark.parametrize("command", ["benchmark", "memory", "lua-benchmark"])
def test_benchmark_wrappers_report_missing_directory(command: str) -> None:
    code = f"""
from pathlib import Path
from rlimit import cli, lua_cli
missing = Path.cwd() / "no-such-benchmarks-directory"
cli._find_repo_root = lambda: missing
lua_cli._find_repo_root = lambda: missing
raise SystemExit(cli.main([{command!r}]))
"""
    result = _run_isolated(code)

    assert result.returncode == 1
    assert "Could not find the benchmarks/ directory" in result.stderr


def test_benchmark_wrapper_forwards_arguments() -> None:
    code = """
from pathlib import Path
from types import SimpleNamespace
from rlimit import cli
root = Path.cwd()
calls = []
cli._find_repo_root = lambda: root
cli.subprocess.run = lambda args, cwd: (
    calls.append((args, cwd)) or SimpleNamespace(returncode=7)
)
assert cli.main(["benchmark", "--", "--concurrency", "1"]) == 7
assert calls[0][0][-2:] == ["--concurrency", "1"]
assert calls[0][1] == str(root)
"""
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr


def test_memory_wrapper_forwards_key_counts() -> None:
    code = """
from pathlib import Path
from types import SimpleNamespace
from rlimit import cli
root = Path.cwd()
calls = []
cli._find_repo_root = lambda: root
cli.subprocess.run = lambda args, cwd: (
    calls.append(args) or SimpleNamespace(returncode=0)
)
assert cli.main(["memory", "--n-keys", "10", "20"]) == 0
assert calls[0][-3:] == ["--n-keys", "10", "20"]
"""
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr


def test_lua_cli_reports_connection_failure_without_network() -> None:
    code = """
from types import SimpleNamespace
import rlimit.lua_cli as lua_cli
class BrokenRedis:
    def ping(self):
        raise ConnectionError("offline")
lua_cli.redis.Redis = lambda **kwargs: BrokenRedis()
args = SimpleNamespace(
    redis_host="localhost", redis_port=6379, algorithm="fixed_window",
    param1=2, param2=60, requests=1, interval=0, cost=1, key="test",
)
raise SystemExit(lua_cli._cmd_lua_demo(args))
"""
    result = _run_isolated(code)

    assert result.returncode == 1
    assert "Could not reach Redis" in result.stderr


def test_lua_benchmark_wrapper_forwards_arguments() -> None:
    code = """
from pathlib import Path
from types import SimpleNamespace
from rlimit import cli, lua_cli
root = Path.cwd()
calls = []
lua_cli._find_repo_root = lambda: root
lua_cli.subprocess.run = lambda args, cwd: (
    calls.append((args, cwd)) or SimpleNamespace(returncode=0)
)
assert cli.main(["lua-benchmark", "--", "--concurrency", "2"]) == 0
assert calls[0][0][-2:] == ["--concurrency", "2"]
assert calls[0][1] == str(root)
"""
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
