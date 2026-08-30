"""The wall-timeout pool must not silently throttle a caller's concurrency.

``ModelClient`` routes non-Bedrock backends through one process-wide pool that
is eight slots wide by default. A caller running more workers than that used to
get eight-way concurrency while believing it had asked for more: the extra
threads queued, the server saw eight in flight, and the wait was charged to each
call's timeout budget. ``reserve_wall_pool`` is the declaration that fixes it,
and these tests pin the two properties the callers rely on — that it GROWS, and
that the width it reports is the concurrency actually delivered.
"""
from __future__ import annotations

import concurrent.futures as cf
import threading

import pytest

from indra_belief.model_client import ModelClient


@pytest.fixture
def restore_pool():
    """Give the width back. The pool is class-level, so leaking a widened one
    would let this module's order decide another test's concurrency."""
    pool, width = ModelClient._WALL_POOL, ModelClient._WALL_POOL_WORKERS
    yield
    ModelClient._WALL_POOL, ModelClient._WALL_POOL_WORKERS = pool, width


def test_default_width_is_eight():
    assert ModelClient._WALL_POOL_WORKERS == 8
    assert ModelClient._WALL_POOL._max_workers == 8


def test_reservation_grows_and_never_shrinks(restore_pool):
    assert ModelClient.reserve_wall_pool(4) == 8, "a smaller ask must not shrink"
    assert ModelClient.reserve_wall_pool(16) == 16
    assert ModelClient._WALL_POOL._max_workers == 16
    assert ModelClient.reserve_wall_pool(8) == 16, "a later smaller ask must not shrink"
    assert ModelClient.reserve_wall_pool(24) == 24


def test_reserved_width_is_the_concurrency_actually_delivered(restore_pool):
    """The point of the reservation, asserted by a barrier that only trips if
    all 16 tasks are running at once. Without the reservation this deadlocks
    until the timeout, which is exactly the throttle being fixed."""
    workers = 16
    ModelClient.reserve_wall_pool(workers)
    barrier = threading.Barrier(workers, timeout=10)
    client = ModelClient.__new__(ModelClient)  # no network, no config needed

    def hold() -> int:
        return barrier.wait()

    with cf.ThreadPoolExecutor(max_workers=workers) as callers:
        futures = [callers.submit(client._invoke_with_wall_timeout, hold, 30)
                   for _ in range(workers)]
        assert sorted(f.result() for f in futures) == list(range(workers))


def test_eight_concurrent_calls_fit_without_any_reservation():
    """The default is a floor that works unreserved — a caller at or below eight
    never needs to know this pool exists."""
    barrier = threading.Barrier(8, timeout=10)
    client = ModelClient.__new__(ModelClient)

    with cf.ThreadPoolExecutor(max_workers=8) as callers:
        futures = [callers.submit(client._invoke_with_wall_timeout, barrier.wait, 30)
                   for _ in range(8)]
        assert sorted(f.result() for f in futures) == list(range(8))


def test_probe_path_finds_the_wall_timeout_by_its_current_name():
    """`src/indra_belief/probes/reader.py::read_probe` reaches the circuit
    breaker by duck-typing a PRIVATE name:

        wall_timeout = getattr(client, "_invoke_with_wall_timeout", None)
        ...
        if callable(wall_timeout): ...
        else: response = create(**request)

    A rename would make that getattr return None and take the else branch, which
    issues the call with no wall bound at all. The SDK's own `timeout` is
    per-connection and per-chunk and does not cap total wall time — that is why
    the wrapper exists — so the probe would lose its only real bound, silently
    and with every test still green. This pins the coupling until the private
    reach-through is replaced by a public one.
    """
    assert callable(getattr(ModelClient, "_invoke_with_wall_timeout", None)), (
        "probes/reader.py::read_probe looks this name up with getattr and falls "
        "back to an UNBOUNDED call when it is missing — rename it there too"
    )
