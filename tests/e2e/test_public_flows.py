from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import redis
from testcontainers.community.redis import RedisContainer

import rlimit
from rlimit import (
    FixedWindow,
    LuaSimulationRecorder,
    RedisLuaFixedWindow,
    SimulationClock,
    SimulationRecorder,
    constant_rate_traffic,
    run_lua_simulation,
    run_simulation,
)

pytestmark = pytest.mark.e2e
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_from_checkout() -> dict[str, str]:
    env = dict(os.environ)
    source_dir = str(_PROJECT_ROOT / "src")
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [source_dir, old_path] if old_path else [source_dir]
    )
    return env


def test_in_memory_public_api_drives_metrics_and_decisions() -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=2, period=60, clock=clock, metrics=recorder)

    run_simulation(limiter, clock, constant_rate_traffic("e2e", 4, 0))

    assert [row["decision"] for row in recorder.rows] == [
        "allowed",
        "allowed",
        "denied",
        "denied",
    ]
    assert limiter.remaining("e2e") == 0


def test_simulation_csv_round_trip(tmp_path: Path) -> None:
    clock = SimulationClock()
    recorder = SimulationRecorder()
    limiter = FixedWindow(limit=2, period=60, clock=clock, metrics=recorder)
    output = tmp_path / "simulation.csv"

    run_simulation(limiter, clock, constant_rate_traffic("csv", 4, 0))
    recorder.flush_to_csv(output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["decision"] for row in rows} == {"allowed", "denied"}
    assert sum(row["decision"] == "allowed" for row in rows) == 2
    assert sum(row["decision"] == "denied" for row in rows) == 2


def test_cli_demo_has_exact_decision_counts() -> None:
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
        cwd=_PROJECT_ROOT,
        env=_env_from_checkout(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("ALLOWED") == 2
    assert result.stdout.count("DENIED") == 2


@pytest.mark.parametrize(
    "command",
    [
        "benchmark",
        "lua-benchmark",
    ],
)
def test_benchmark_cli_help(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rlimit", command, "--", "--help"],
        cwd=_PROJECT_ROOT,
        env=_env_from_checkout(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--concurrency" in result.stdout


@pytest.mark.docker
def test_lua_public_api_matches_in_memory_for_burst(
    redis_client: redis.Redis,
) -> None:
    memory = FixedWindow(limit=2, period=60)
    distributed = RedisLuaFixedWindow(redis_client, limit=2, period=60)

    memory_results = [memory.allow("same-burst") for _ in range(4)]
    redis_results = [distributed.allow("same-burst") for _ in range(4)]

    assert memory_results == redis_results == [True, True, False, False]


@pytest.mark.docker
def test_lua_cli_demo_against_test_redis(
    redis_container: RedisContainer,
) -> None:
    host = redis_container.get_container_host_ip()
    port = str(redis_container.get_exposed_port(redis_container.port))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlimit",
            "lua-demo",
            "--algorithm",
            "fixed_window",
            "--param1",
            "2",
            "--param2",
            "60",
            "--requests",
            "4",
            "--redis-host",
            host,
            "--redis-port",
            port,
        ],
        cwd=_PROJECT_ROOT,
        env=_env_from_checkout(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("ALLOWED") == 2
    assert result.stdout.count("DENIED") == 2


def test_lua_cli_reports_unreachable_redis() -> None:
    code = "\n".join(
        [
            "from types import SimpleNamespace",
            "import rlimit.lua_cli as lua_cli",
            "class BrokenRedis:",
            "    def ping(self):",
            "        raise ConnectionError('offline')",
            "lua_cli.redis.Redis = lambda **kwargs: BrokenRedis()",
            "args = SimpleNamespace(",
            "    redis_host='127.0.0.1', redis_port=1,",
            "    algorithm='fixed_window', param1=2, param2=60,",
            "    requests=1, interval=0, cost=1, key='e2e',",
            ")",
            "raise SystemExit(lua_cli._cmd_lua_demo(args))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT,
        env=_env_from_checkout(),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 1
    assert "Could not reach Redis" in result.stderr


def test_root_exports_are_resolvable_from_black_box() -> None:
    assert rlimit.FixedWindow is FixedWindow
    assert rlimit.RedisLuaFixedWindow is RedisLuaFixedWindow


@pytest.mark.docker
def test_lua_simulation_csv_round_trip(
    redis_client: redis.Redis, tmp_path: Path
) -> None:
    recorder = LuaSimulationRecorder()
    limiter = RedisLuaFixedWindow(
        redis_client, limit=2, period=60, metrics=recorder
    )
    sleep_calls: list[float] = []
    steps = constant_rate_traffic("lua-csv", num_requests=4, interval=0.25)

    run_lua_simulation(limiter, steps, recorder, sleep=sleep_calls.append)
    output = tmp_path / "lua-simulation.csv"
    recorder.flush_to_csv(output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert sleep_calls == [0.25, 0.25, 0.25]
    assert len(rows) == 4
    assert sum(row["decision"] == "allowed" for row in rows) == 2
    assert sum(row["decision"] == "denied" for row in rows) == 2
