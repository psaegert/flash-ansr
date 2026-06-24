"""Tests for the persistent recoverable fork pool (inference-speed Step 3).

The load-bearing NEW behaviour the persistent pool introduces is recovery: a worker death poisons the
executor so the next submit raises BrokenProcessPool, which (unlike the per-call-fork design that got a
fresh pool every call) would abort a multi-hour run unless the pool recreates itself.
"""
import pytest

if "fork" not in __import__("multiprocessing").get_all_start_methods():  # pragma: no cover
    pytest.skip("RecoverableForkPool requires the 'fork' start method", allow_module_level=True)

from concurrent.futures.process import BrokenProcessPool  # noqa: E402

from flash_ansr._refine_pool import RecoverableForkPool  # noqa: E402


def _square(x):
    return x * x


def _identity(x):
    return x


def _suicide(x):
    import os
    import signal
    os.kill(os.getpid(), signal.SIGKILL)  # kill this worker -> poisons the pool every time
    return x  # unreachable


def test_map_ordered_returns_results_in_submission_order():
    pool = RecoverableForkPool(4)
    try:
        assert pool.map_ordered(_square, range(20)) == [i * i for i in range(20)]
        # order is preserved even with chunking
        assert pool.map_ordered(_identity, range(50), chunksize=7) == list(range(50))
        assert pool.map_ordered(_square, []) == []
    finally:
        pool.shutdown()


def test_force_fork_creates_live_workers_immediately():
    pool = RecoverableForkPool(3)
    try:
        procs = list(getattr(pool._pool, "_processes", {}).values())
        assert len(procs) == 3, "workers should be force-forked at construction (pre-CUDA)"
        assert all(p.is_alive() for p in procs)
    finally:
        pool.shutdown()


def test_recovers_after_all_workers_killed():
    """Killing every worker poisons the executor; the next call must recreate it and still succeed."""
    pool = RecoverableForkPool(4)
    try:
        assert pool.map_ordered(_square, range(8)) == [i * i for i in range(8)]
        old_pids = {p.pid for p in getattr(pool._pool, "_processes", {}).values()}

        pool._kill_workers()  # poison: next submit will raise BrokenProcessPool internally

        # map_ordered must catch the broken pool, recreate, retry, and return correct results.
        assert pool.map_ordered(_square, range(8)) == [i * i for i in range(8)]
        new_pids = {p.pid for p in getattr(pool._pool, "_processes", {}).values()}
        assert new_pids and new_pids.isdisjoint(old_pids), "pool should have been recreated with fresh workers"

        # and it stays healthy for subsequent calls
        assert pool.map_ordered(_identity, range(5)) == list(range(5))
    finally:
        pool.shutdown()


def test_raises_after_max_recreate_but_leaves_healthy_pool():
    """Invariant 4: a deterministically-poisonous batch raises after max_recreate, NOT an infinite
    loop, and the pool is left HEALTHY for the next caller."""
    pool = RecoverableForkPool(2, max_recreate=2)
    try:
        with pytest.raises(BrokenProcessPool):
            pool.map_ordered(_suicide, [1])  # recover=True default -> recreate+retry, then give up
        # the post-raise contract: a benign call must succeed on a fresh, healthy pool
        assert pool.map_ordered(_square, range(4)) == [0, 1, 4, 9]
    finally:
        pool.shutdown()


def test_recover_false_propagates_without_recreating():
    """recover=False (the CUDA-tainted-caller policy) must propagate and NOT re-fork the pool."""
    pool = RecoverableForkPool(2)
    try:
        assert pool.map_ordered(_square, range(4)) == [0, 1, 4, 9]
        pool._kill_workers()  # poison the pool
        with pytest.raises(BrokenProcessPool):
            pool.map_ordered(_square, range(4), recover=False)
        # NOT recreated -> still broken -> a second recover=False call raises again
        with pytest.raises(BrokenProcessPool):
            pool.map_ordered(_square, range(4), recover=False)
    finally:
        pool.shutdown()


def test_shutdown_is_idempotent():
    pool = RecoverableForkPool(2)
    pool.shutdown()
    pool.shutdown()  # must not raise
    assert pool._pool is None


def test_rejects_bad_worker_count():
    with pytest.raises(ValueError):
        RecoverableForkPool(0)
