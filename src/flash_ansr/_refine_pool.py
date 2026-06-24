"""Persistent recoverable fork pool for inference-speed Step 3 (pre-CUDA refine + simplify).

A single ``ProcessPoolExecutor`` over the ``fork`` start method, created ONCE *before* any CUDA init.
This is the structural mitigation for the fork-after-CUDA / nested-fork deadlock family the project
hit for ~9h: forking after a live CUDA context (or nesting two fork pools) inherits locked
CUDA/GIL/glibc-arena mutexes. The same pool serves BOTH the constant-refinement ``curve_fit`` jobs
and the parallel post-generation simplify pass, so NO fork ever happens after CUDA. The per-call fork
pool in :class:`~flash_ansr.flash_ansr.FlashANSR` stays the default; this pool is opt-in
(``FlashANSR.load(persistent_refine_pool=True)``).

**Recovery is the one failure mode the persistent pool introduces** that the per-call pool did not
have. A single worker death (a ``curve_fit`` segfault, an OOM-killed child) poisons the executor so
the *next* submit raises :class:`BrokenProcessPool` and would abort the whole (possibly multi-hour)
run. The per-call design recovered for free: a fresh pool every ``fit()``. The recovery POLICY is the
caller's, via :meth:`map_ordered`'s ``recover`` flag, because recreating the pool RE-FORKS workers:

- ``recover=True`` (generic / CPU / pre-CUDA caller): recreate the pool and re-run the whole batch
  (safe: refinement is per-candidate-seeded + order-independent, simplify is a pure function).
- ``recover=False`` (CUDA-tainted caller, e.g. ``FlashANSR._fit_refine`` which only ever runs after
  generation has initialized CUDA): do NOT recreate -- a re-fork from a live-CUDA parent would
  reintroduce the very fork-after-CUDA hazard this pool exists to avoid. Propagate the error so the
  caller degrades gracefully. ``FlashANSR`` catches it, disposes the persistent pool, and falls back
  to the LEGACY per-call fork pool (which already forks post-CUDA every call and is the deployed,
  proven-safe path) -- the most faithful "recover like the per-call pool did" without the re-fork.

The stall watchdog is plumbed but DISABLED by default (``stall_timeout=None``): a fixed per-call
deadline would false-positive-kill a legitimately long refine batch at large ``c`` / on a slow CPU,
and a true wedge would have wedged the per-call pool too (it is not a *new* Step-3 risk the way a
broken pool is). A per-job (not per-call) timeout is a later refinement.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Iterable


def _force_fork_probe(x: int) -> int:
    """Trivial picklable task that forces the pool to fork all its workers NOW (while pre-CUDA)."""
    return x


class RecoverableForkPool:
    """A persistent ``fork`` :class:`ProcessPoolExecutor` with broken-pool + stall recovery.

    Parameters
    ----------
    max_workers : int
        Number of worker processes (must be >= 1).
    initializer, initargs : callable and tuple, optional
        Forwarded to :class:`ProcessPoolExecutor`; run once per worker at fork time. Used to seed the
        per-worker engine globals (problem-independent), so per-problem data travels in each payload.
    stall_timeout : float or None, optional
        Per-call wall-clock deadline for :meth:`map_ordered`. ``None`` (default) disables the
        watchdog; a positive value force-kills the workers and recreates the pool if a call does not
        finish in time.
    max_recreate : int, optional
        Maximum number of pool recreations per :meth:`map_ordered` call before the error propagates.
    """

    def __init__(
            self,
            max_workers: int,
            *,
            initializer: Callable[..., None] | None = None,
            initargs: tuple = (),
            stall_timeout: float | None = None,
            max_recreate: int = 3) -> None:
        if int(max_workers) < 1:
            raise ValueError("RecoverableForkPool requires max_workers >= 1")
        if 'fork' not in mp.get_all_start_methods():
            raise RuntimeError("RecoverableForkPool requires the 'fork' start method")
        self._max_workers = int(max_workers)
        self._initializer = initializer
        self._initargs = initargs
        self._stall_timeout = stall_timeout
        self._max_recreate = int(max_recreate)
        self._ctx = mp.get_context('fork')
        self._pool: ProcessPoolExecutor | None = None
        self._closed = False
        self._build()

    def _build(self) -> None:
        self._pool = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=self._ctx,
            initializer=self._initializer,
            initargs=self._initargs,
        )
        # Force every worker to fork NOW, while we are guaranteed to be pre-CUDA. Without this the
        # workers fork lazily on the first real submit -- which, for the persistent pool, would be
        # AFTER .to('cuda'), reintroducing exactly the fork-after-CUDA hazard this design removes.
        list(self._pool.map(_force_fork_probe, range(self._max_workers * 2)))

    def _kill_workers(self) -> None:
        pool = self._pool
        if pool is None:
            return
        # CPython sets ProcessPoolExecutor._processes to None during shutdown/reset, so the {}-default
        # (which only guards a MISSING attribute) is not enough -- coerce a None value to {} too.
        for proc in list((getattr(pool, '_processes', None) or {}).values()):
            try:
                if proc.is_alive() and proc.pid is not None:
                    os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def _recreate(self, *, kill: bool = False) -> None:
        if kill:
            self._kill_workers()
        old = self._pool
        self._pool = None
        if old is not None:
            try:
                old.shutdown(wait=not kill, cancel_futures=True)
            except Exception:
                pass
        self._build()

    def map_ordered(
            self,
            fn: Callable[[Any], Any],
            items: Iterable[Any],
            *,
            chunksize: int = 1,
            wrap: Callable[[Iterable[Any]], Iterable[Any]] | None = None,
            recover: bool = True) -> list:
        """``list(executor.map(fn, items))``; optionally with broken-pool / stall recovery.

        Results are returned in submission order (deterministic). ``chunksize`` is forwarded to the
        underlying ``executor.map`` for IPC efficiency on large item lists. ``wrap`` may transform the
        result iterator (e.g. a progress bar) before it is materialized; it is the caller's hook so
        this module needs no dependency on the progress helper.

        ``recover`` controls the worker-death / stall policy:

        - ``True`` (default, for a CPU / generic / pre-CUDA caller): on :class:`BrokenProcessPool`
          (or ``TimeoutError`` when ``stall_timeout`` is set) the pool is recreated and the WHOLE call
          is retried (safe: refinement is per-candidate-seeded + order-independent, simplify is a pure
          function). After ``max_recreate`` failures the pool is recreated once more (so the next
          caller sees a healthy pool) and the error is re-raised. NB recreating RE-FORKS workers; the
          caller must guarantee that is safe (e.g. the parent has not initialized CUDA).

        - ``False`` (for a CUDA-tainted caller, e.g. ``FlashANSR._fit_refine`` after generation): do
          NOT recreate -- a re-fork from a live-CUDA parent would reintroduce the fork-after-CUDA
          hazard the persistent pool exists to avoid. The error is propagated immediately so the
          caller can degrade gracefully (FlashANSR falls back to the legacy per-call fork pool).
        """
        items = list(items)
        if not items:
            return []
        attempt = 0
        while True:
            assert self._pool is not None
            try:
                iterator: Iterable[Any] = self._pool.map(fn, items, chunksize=chunksize, timeout=self._stall_timeout)
                if wrap is not None:
                    iterator = wrap(iterator)
                return list(iterator)
            except (BrokenProcessPool, TimeoutError) as exc:
                if not recover:
                    raise
                attempt += 1
                killed = isinstance(exc, TimeoutError)
                if attempt > self._max_recreate:
                    # Leave a healthy pool behind for the next caller, then surface the failure.
                    self._recreate(kill=killed)
                    raise
                self._recreate(kill=killed)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        pool = self._pool
        self._pool = None
        if pool is not None:
            try:
                pool.shutdown(wait=True)
            except Exception:
                pass

    def __del__(self) -> None:  # best-effort; explicit shutdown() is preferred
        try:
            self.shutdown()
        except Exception:
            pass
