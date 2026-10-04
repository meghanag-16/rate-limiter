# tests/test_debug_logging.py
"""Test structured decision events when DEBUG logging is explicitly enabled."""

from __future__ import annotations

import logging

import pytest
from structlog.testing import capture_logs

from rlimit.algorithms.async_fixed_window import AsyncFixedWindow
from rlimit.algorithms.async_leaky_bucket import (
    AsyncLeakyBucketMeter,
    AsyncLeakyBucketQueue,
)
from rlimit.algorithms.async_sliding_window_counter import AsyncSlidingWindowCounter
from rlimit.algorithms.async_sliding_window_log import AsyncSlidingWindowLog
from rlimit.algorithms.async_token_bucket import AsyncTokenBucket
from rlimit.algorithms.fixed_window import FixedWindow
from rlimit.algorithms.leaky_bucket import LeakyBucketMeter, LeakyBucketQueue
from rlimit.algorithms.sliding_window_counter import SlidingWindowCounter
from rlimit.algorithms.sliding_window_log import SlidingWindowLog
from rlimit.algorithms.token_bucket import TokenBucket
from rlimit.logging import configure_logging

configure_logging(level=logging.DEBUG)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# --- Sync: one test per concrete class, exact event name + fields -----


def test_fixed_window_emits_debug_event_with_documented_fields() -> None:
    limiter = FixedWindow(limit=3, period=60.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "fixed_window_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["key"] == "k"
    assert e["allowed"] is True
    assert e["cost"] == 2
    assert e["count_before"] == 0
    assert e["count"] == 2
    assert e["limit"] == 3
    assert e["rolled_over"] is True  # first-ever call for this key
    assert "window_start" in e


def test_fixed_window_debug_event_shows_the_rollover_transition() -> None:
    """Transition-level check (not just final-state presence): a
    second call within the SAME window must report rolled_over=False
    and count_before reflecting the first call's admitted count."""
    clock = FakeClock()
    limiter = FixedWindow(limit=5, period=60.0, clock=clock)
    limiter.allow("k", cost=2)  # establishes count=2 in the window

    with capture_logs() as cap:
        limiter.allow("k", cost=1)

    events = [e for e in cap if e["event"] == "fixed_window_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["rolled_over"] is False
    assert e["count_before"] == 2
    assert e["cost"] == 1
    assert e["count"] == 3


def test_token_bucket_emits_debug_event_with_documented_fields() -> None:
    limiter = TokenBucket(capacity=5, refill_rate=1.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k")

    events = [e for e in cap if e["event"] == "token_bucket_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["key"] == "k"
    assert e["allowed"] is True
    assert e["cost"] == 1
    assert e["tokens_before"] == 5.0  # bucket starts full
    assert e["elapsed"] == 0.0
    assert e["refilled"] == 0.0
    assert e["tokens_after_refill"] == 5.0
    assert e["tokens"] == 4
    assert e["capacity"] == 5


def test_token_bucket_debug_event_shows_the_refill_transition() -> None:
    """Transition-level check: draining the bucket, then letting time
    pass, must show a nonzero `refilled` amount distinct from the
    final `tokens` value -- proving this is genuinely before/after
    refill data, not just the post-decision token count repeated."""
    clock = FakeClock()
    limiter = TokenBucket(capacity=10, refill_rate=2.0, clock=clock)
    limiter.allow("k", cost=10)  # fully drain
    clock.advance(3.0)  # 3s * 2/s = 6 tokens should refill

    with capture_logs() as cap:
        limiter.allow("k", cost=1)

    events = [e for e in cap if e["event"] == "token_bucket_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["tokens_before"] == 0.0
    assert e["elapsed"] == pytest.approx(3.0)
    assert e["refilled"] == pytest.approx(6.0)
    assert e["tokens_after_refill"] == pytest.approx(6.0)
    assert e["cost"] == 1
    assert e["tokens"] == pytest.approx(5.0)  # 6 refilled - 1 consumed


def test_sliding_window_log_emits_debug_event_with_documented_fields() -> None:
    limiter = SlidingWindowLog(limit=5, period=60.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "sliding_window_log_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["key"] == "k"
    assert e["allowed"] is True
    assert e["cost"] == 2
    assert e["entries_before"] == 0
    assert e["expired"] == 0
    assert e["entries_count"] == 1
    assert e["total_cost_before"] == 0
    assert e["total_cost"] == 2
    assert e["limit"] == 5


def test_sliding_window_log_debug_event_shows_the_expiry_transition() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLog(limit=5, period=10.0, clock=clock)
    limiter.allow("k")  # one entry at t=0
    clock.advance(10.01)  # that entry now expires

    with capture_logs() as cap:
        limiter.allow("k")

    events = [e for e in cap if e["event"] == "sliding_window_log_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["entries_before"] == 1
    assert e["expired"] == 1
    assert e["entries_count"] == 1  # the one new entry just added


def test_sliding_window_counter_emits_debug_event_with_documented_fields() -> None:
    limiter = SlidingWindowCounter(limit=10, period=10.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k", cost=3)

    events = [e for e in cap if e["event"] == "sliding_window_counter_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["key"] == "k"
    assert e["allowed"] is True
    assert e["cost"] == 3
    assert e["count"] == 0
    assert e["prev_count"] == 0
    assert e["weighted_before"] == 0
    assert e["weighted"] == 3
    assert e["limit"] == 10


def test_sliding_window_counter_debug_event_shows_the_decay_transition() -> None:
    clock = FakeClock()
    limiter = SlidingWindowCounter(limit=10, period=10.0, clock=clock)
    for _ in range(10):
        limiter.allow("k")
    # Advance past the full period (rolling into window [10, 20)) plus
    # 5 more seconds, so we're halfway through the NEW window with the
    # exhausted window as prev_count at overlap_fraction 0.5.
    clock.advance(15.0)

    with capture_logs() as cap:
        limiter.allow("k")

    events = [e for e in cap if e["event"] == "sliding_window_counter_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["prev_count"] == 10
    assert e["count"] == 0
    assert e["weighted_before"] == pytest.approx(5.0)  # 10 * 0.5 overlap


def test_leaky_bucket_meter_emits_debug_event_with_documented_fields() -> None:
    limiter = LeakyBucketMeter(capacity=5, leak_rate=1.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "leaky_bucket_meter_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["key"] == "k"
    assert e["allowed"] is True
    assert e["cost"] == 2
    assert e["volume_before"] == 0.0
    assert e["elapsed"] == 0.0
    assert e["leaked"] == 0.0
    assert e["volume_after_leak"] == 0.0
    assert e["volume"] == 2
    assert e["capacity"] == 5


def test_leaky_bucket_meter_debug_event_shows_the_leak_transition() -> None:
    clock = FakeClock()
    limiter = LeakyBucketMeter(capacity=10, leak_rate=2.0, clock=clock)
    limiter.allow("k", cost=10)  # fill fully
    clock.advance(3.0)  # 3s * 2/s = 6 units should leak out

    with capture_logs() as cap:
        limiter.allow("k", cost=1)

    events = [e for e in cap if e["event"] == "leaky_bucket_meter_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["volume_before"] == pytest.approx(10.0)
    assert e["elapsed"] == pytest.approx(3.0)
    assert e["leaked"] == pytest.approx(6.0)
    assert e["volume_after_leak"] == pytest.approx(4.0)
    assert e["volume"] == pytest.approx(5.0)  # 4 after leak + 1 consumed


def test_leaky_bucket_queue_emits_debug_event_with_documented_fields() -> None:
    limiter = LeakyBucketQueue(capacity=5, leak_rate=1.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "leaky_bucket_queue_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["key"] == "k"
    assert e["allowed"] is True
    assert e["cost"] == 2
    assert e["depth_before"] == 0
    assert e["elapsed"] == 0.0
    assert e["drained"] == 0
    assert e["depth_after_drain"] == 0
    assert e["depth"] == 2
    assert e["capacity"] == 5


def test_leaky_bucket_queue_debug_event_shows_the_drain_transition() -> None:
    clock = FakeClock()
    limiter = LeakyBucketQueue(capacity=5, leak_rate=1.0, clock=clock)  # 1 item/s
    limiter.allow("k", cost=5)  # fill fully
    clock.advance(3.0)  # 3 whole items should drain

    with capture_logs() as cap:
        limiter.allow("k", cost=1)

    events = [e for e in cap if e["event"] == "leaky_bucket_queue_decision"]
    assert len(events) == 1
    e = events[0]
    assert e["depth_before"] == 5
    assert e["elapsed"] == pytest.approx(3.0)
    assert e["drained"] == 3
    assert e["depth_after_drain"] == 2
    assert e["depth"] == 3  # 2 after drain + 1 consumed


# --- Async: one test per concrete class, exact event name + fields ----


@pytest.mark.asyncio
async def test_async_fixed_window_emits_debug_event_with_documented_fields() -> None:
    limiter = AsyncFixedWindow(limit=3, period=60.0, clock=FakeClock())
    with capture_logs() as cap:
        await limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "async_fixed_window_decision"]
    assert len(events) == 1
    assert events[0]["cost"] == 2
    assert events[0]["count_before"] == 0
    assert events[0]["count"] == 2
    assert events[0]["limit"] == 3
    assert events[0]["rolled_over"] is True


@pytest.mark.asyncio
async def test_async_token_bucket_emits_debug_event_with_documented_fields() -> None:
    limiter = AsyncTokenBucket(capacity=5, refill_rate=1.0, clock=FakeClock())
    with capture_logs() as cap:
        await limiter.allow("k")

    events = [e for e in cap if e["event"] == "async_token_bucket_decision"]
    assert len(events) == 1
    assert events[0]["cost"] == 1
    assert events[0]["tokens_before"] == 5.0
    assert events[0]["elapsed"] == 0.0
    assert events[0]["refilled"] == 0.0
    assert events[0]["tokens"] == 4
    assert events[0]["capacity"] == 5


@pytest.mark.asyncio
async def test_async_token_bucket_debug_event_shows_the_refill_transition() -> None:
    clock = FakeClock()
    limiter = AsyncTokenBucket(capacity=10, refill_rate=2.0, clock=clock)
    await limiter.allow("k", cost=10)
    clock.advance(3.0)

    with capture_logs() as cap:
        await limiter.allow("k", cost=1)

    events = [e for e in cap if e["event"] == "async_token_bucket_decision"]
    assert len(events) == 1
    assert events[0]["tokens_before"] == 0.0
    assert events[0]["elapsed"] == pytest.approx(3.0)
    assert events[0]["refilled"] == pytest.approx(6.0)
    assert events[0]["tokens"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_async_sliding_window_log_emits_debug_event_with_documented_fields() -> (
    None
):
    limiter = AsyncSlidingWindowLog(limit=5, period=60.0, clock=FakeClock())
    with capture_logs() as cap:
        await limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "async_sliding_window_log_decision"]
    assert len(events) == 1
    assert events[0]["cost"] == 2
    assert events[0]["entries_before"] == 0
    assert events[0]["expired"] == 0
    assert events[0]["entries_count"] == 1
    assert events[0]["total_cost"] == 2


@pytest.mark.asyncio
async def test_async_sliding_window_counter_emits_debug_event_with_documented_fields() -> (  # noqa: E501
    None
):
    limiter = AsyncSlidingWindowCounter(limit=10, period=10.0, clock=FakeClock())
    with capture_logs() as cap:
        await limiter.allow("k", cost=3)

    events = [e for e in cap if e["event"] == "async_sliding_window_counter_decision"]
    assert len(events) == 1
    assert events[0]["cost"] == 3
    assert events[0]["count"] == 0
    assert events[0]["prev_count"] == 0
    assert events[0]["weighted_before"] == 0
    assert events[0]["weighted"] == 3
    assert events[0]["limit"] == 10


@pytest.mark.asyncio
async def test_async_leaky_bucket_meter_emits_debug_event_with_documented_fields() -> (
    None
):
    limiter = AsyncLeakyBucketMeter(capacity=5, leak_rate=1.0, clock=FakeClock())
    with capture_logs() as cap:
        await limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "async_leaky_bucket_meter_decision"]
    assert len(events) == 1
    assert events[0]["cost"] == 2
    assert events[0]["volume_before"] == 0.0
    assert events[0]["elapsed"] == 0.0
    assert events[0]["leaked"] == 0.0
    assert events[0]["volume"] == 2
    assert events[0]["capacity"] == 5


@pytest.mark.asyncio
async def test_async_leaky_bucket_queue_emits_debug_event_with_documented_fields() -> (
    None
):
    limiter = AsyncLeakyBucketQueue(capacity=5, leak_rate=1.0, clock=FakeClock())
    with capture_logs() as cap:
        await limiter.allow("k", cost=2)

    events = [e for e in cap if e["event"] == "async_leaky_bucket_queue_decision"]
    assert len(events) == 1
    assert events[0]["cost"] == 2
    assert events[0]["depth_before"] == 0
    assert events[0]["elapsed"] == 0.0
    assert events[0]["drained"] == 0
    assert events[0]["depth"] == 2
    assert events[0]["capacity"] == 5


# --- Scope boundary: cost==0 and invalid cost emit NO debug event -----
# (mirrors the same documented scope boundary already tested for
# metrics in test_metrics.py -- debug logging follows the same rule:
# it only fires on the actual lock-held decision path, per every
# algorithm file's early-return structure. Checked here for
# FixedWindow as the representative case.)


def test_cost_zero_emits_no_debug_event() -> None:
    limiter = FixedWindow(limit=3, period=60.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k", cost=0)

    assert [e for e in cap if e["event"] == "fixed_window_decision"] == []


def test_invalid_cost_emits_no_debug_event() -> None:
    limiter = FixedWindow(limit=3, period=60.0, clock=FakeClock())
    with capture_logs() as cap:
        limiter.allow("k", cost=-1)

    assert [e for e in cap if e["event"] == "fixed_window_decision"] == []


# --- Denied decisions also log (not just allowed) ----------------------


def test_denied_decision_still_emits_debug_event() -> None:
    limiter = FixedWindow(limit=1, period=60.0, clock=FakeClock())
    limiter.allow("k")  # consume the one slot

    with capture_logs() as cap:
        limiter.allow("k")

    events = [e for e in cap if e["event"] == "fixed_window_decision"]
    assert len(events) == 1
    assert events[0]["allowed"] is False
