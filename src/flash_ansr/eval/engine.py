"""General-purpose evaluation runner tying data sources and model adapters together."""
from __future__ import annotations

import queue
import threading
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from tqdm import tqdm

from flash_ansr.eval.core import (
    EvaluationDataSource,
    EvaluationModelAdapter,
    EvaluationResult,
    EvaluationSample,
)
from flash_ansr.eval.result_store import ResultStore
from flash_ansr.utils.paths import substitute_root_path


class EvaluationEngine:
    """Drive evaluation loops with pluggable data sources and model adapters."""

    def __init__(
        self,
        data_source: EvaluationDataSource,
        model_adapter: EvaluationModelAdapter,
        *,
        result_store: ResultStore | None = None,
    ) -> None:
        self.data_source = data_source
        self.model_adapter = model_adapter
        self.result_store = result_store or ResultStore()

    def run(
        self,
        *,
        limit: Optional[int] = None,
        save_every: Optional[int] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
        progress: bool = True,
        log_placeholders: bool = True,
        summary_interval: Optional[int] = 50,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, list[Any]]:
        """Execute the evaluation loop and return accumulated results."""

        if save_every is not None and output_path is None:
            raise ValueError("output_path must be provided when save_every is configured")

        resolved_output: Optional[Path] = None
        if output_path is not None:
            resolved_output = Path(substitute_root_path(output_path))

        prepare_adapter = getattr(self.model_adapter, "prepare", None)
        if callable(prepare_adapter):
            prepare_adapter(data_source=self.data_source)

        prepare_source = getattr(self.data_source, "prepare", None)
        if callable(prepare_source):
            prepare_source(adapter=self.model_adapter)

        existing_results = self.result_store.size
        pending_target = limit if limit is not None else getattr(self.data_source, "size_hint", lambda: None)()
        if pending_target is not None:
            pending_target = max(0, int(pending_target))
        overall_target = None if pending_target is None else existing_results + pending_target

        iterator: Iterable[EvaluationSample] = self.data_source
        processed = 0

        progress_bar = None
        if progress and verbose:
            progress_bar = tqdm(total=pending_target, desc="Evaluating", smoothing=0.0)

        tracker: _EvaluationProgressTracker | None = None
        if log_placeholders or (summary_interval is not None and summary_interval > 0):
            tracker = _EvaluationProgressTracker(
                result_store=self.result_store,
                total_target=overall_target,
                logger=lambda message: self._log_message(message, progress_bar),
                log_placeholders=log_placeholders,
            )
            tracker.print_summary("Starting state")

        try:
            for sample in iterator:
                if limit is not None and processed >= limit:
                    break

                try:
                    if getattr(sample, "is_placeholder", False):
                        result = self._build_placeholder_record(sample)
                    else:
                        result = self._evaluate_sample(sample)
                except Exception as exc:  # pragma: no cover - defensive guard
                    result = self._handle_evaluation_exception(sample, exc)

                self.result_store.append(result)
                if tracker is not None:
                    tracker.record(result)

                processed += 1

                if progress_bar is not None:
                    progress_bar.update(1)

                if save_every is not None and processed % save_every == 0 and resolved_output is not None:
                    self.result_store.save(resolved_output, meta=meta)

                if (
                    tracker is not None
                    and summary_interval is not None
                    and summary_interval > 0
                    and processed % summary_interval == 0
                ):
                    tracker.print_summary(
                        f"Progress after {processed} new samples (overall {self.result_store.size})"
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        final_snapshot = self.result_store.snapshot()

        if resolved_output is not None:
            self.result_store.save(resolved_output, meta=meta)

        if tracker is not None:
            tracker.print_summary("Final evaluation summary")

        return final_snapshot

    def _evaluate_sample(self, sample: EvaluationSample) -> dict[str, Any]:
        result: EvaluationResult = self.model_adapter.evaluate_sample(sample)
        mapping = result.to_mapping() if hasattr(result, "to_mapping") else result
        if not isinstance(mapping, dict):
            raise TypeError("Model adapters must return dict-like results")
        return mapping

    def _build_placeholder_record(self, sample: EvaluationSample) -> dict[str, Any]:
        record = sample.clone_metadata()
        record["placeholder"] = True
        if sample.placeholder_reason is not None:
            record["placeholder_reason"] = sample.placeholder_reason
            record.setdefault("error", sample.placeholder_reason)
        else:
            record.setdefault("placeholder_reason", None)
        record.setdefault("prediction_success", False)
        return record

    def _handle_evaluation_exception(self, sample: EvaluationSample, exc: Exception) -> dict[str, Any]:
        warnings.warn(
            f"Evaluation sample failed with an unexpected error: {exc}. Recording placeholder result.",
            RuntimeWarning,
        )
        record = sample.clone_metadata()
        record["placeholder"] = True
        record.setdefault("placeholder_reason", "adapter_exception")
        record["error"] = str(exc)
        record["prediction_success"] = False
        return record

    @staticmethod
    def _log_message(message: str, progress_bar: tqdm | None) -> None:
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message, flush=True)


class _EvaluationProgressTracker:
    """Track placeholder statistics and emit human-readable summaries."""

    def __init__(
        self,
        *,
        result_store: ResultStore,
        total_target: int | None,
        logger: Callable[[str], None],
        log_placeholders: bool,
    ) -> None:
        stats = result_store.statistics()
        self.result_store = result_store
        self.total_target = total_target
        self.logger = logger
        self.log_placeholders = log_placeholders
        self.placeholder_count = stats["placeholders"]
        self.valid_count = stats["valid"]
        self.placeholder_reasons: Counter[str] = Counter(stats.get("placeholder_reasons", {}))

    def record(self, record: Mapping[str, Any]) -> None:
        if record.get("placeholder"):
            reason = str(record.get("placeholder_reason") or "unspecified")
            self.placeholder_count += 1
            self.placeholder_reasons[reason] += 1
            if self.log_placeholders:
                self._log_placeholder_event(reason, record)
        else:
            self.valid_count += 1

    def pending(self) -> int | None:
        if self.total_target is None:
            return None
        return max(0, self.total_target - self.result_store.size)

    def print_summary(self, label: str) -> None:
        summary = self._format_summary()
        self.logger(f"[Evaluation] {label}: {summary}")

    def _format_summary(self) -> str:
        total = self.result_store.size
        pending = self.pending()
        parts = [
            f"total={total}",
            f"valid={self.valid_count}",
            f"placeholders={self.placeholder_count}",
        ]
        if pending is not None:
            parts.append(f"remaining={pending}")
        if self.placeholder_reasons:
            breakdown = ", ".join(
                f"{reason}:{count}" for reason, count in sorted(self.placeholder_reasons.items())
            )
            parts.append(f"reasons={breakdown}")
        return "; ".join(parts)

    def _log_placeholder_event(self, reason: str, record: Mapping[str, Any]) -> None:
        location = (
            record.get("benchmark_eq_id")
            or record.get("skeleton_hash")
            or record.get("skeleton")
            or record.get("sample_id")
        )
        location_text = f", location={location}" if location is not None else ""
        pending = self.pending()
        pending_text = f", remaining={pending}" if pending is not None else ""
        self.logger(
            "[Evaluation] Placeholder #{count} recorded (reason={reason}{location}). "
            "Valid={valid}, placeholders={count}{pending}.".format(
                count=self.placeholder_count,
                reason=reason,
                location=location_text,
                valid=self.valid_count,
                pending=pending_text,
            )
        )


class OverlappedEvaluationEngine(EvaluationEngine):
    """Cross-problem pipeline-overlap engine (inference-speed Regime B).

    Hides problem N's CPU constant-refinement behind problem N+1's GPU generation. A single
    GPU-owner producer thread runs ``adapter.generate_phase`` in sample order and hands each
    ``GenState`` to the main thread through a depth-1 queue; the main thread runs
    ``adapter.refine_extract_phase`` (which dispatches ``curve_fit`` to the model's *persistent
    pre-CUDA* fork pool and reads the result back) and commits results in sample order. While the
    main thread refines N, the producer is already generating N+1.

    Correctness rests on three structural facts, all enforced by :meth:`_overlap_supported`:

    1. **Single CUDA owner.** Only the producer touches the GPU. Under ``prune_constant_budget == 0``
       (every deployed eval config) ``_fit_refine`` is pure CPU, ``predict`` / ``get_expression`` are
       CPU, so the main thread issues no CUDA op -> no two-thread CUDA races.
    2. **No shared mutable model state across the threads.** The generate phase writes only
       ``model._prompt_prefix`` / ``_mcts_cache`` / ``_n_params`` (none read by the commit path); the
       commit path writes ``_results`` / ``variable_mapping`` / ``_input_dim`` / the timings (none read
       by the generate phase). The field sets are disjoint.
    3. **Safe concurrent use of the shared refine pool (Step 4b).** At ``choices >=
       _SIMPLIFY_PARALLEL_THRESHOLD`` (4096) the producer's post-generation simplify pass and the
       consumer's refine share ``model._refine_pool``. This is race-free (refine ships per-job X/y,
       simplify is a pure function, ``ProcessPoolExecutor.map`` is thread-safe) and contention-free in
       practice: generation dominates so heavily at high ``c`` that refine(N) finishes early in
       gen(N+1)'s window, so simplify(N+1) runs effectively uncontended (pipeline = gen+simplify = the
       ceiling). Under ``_overlap_mode`` BOTH the refine and simplify paths re-raise a
       ``BrokenProcessPool`` rather than close/re-fork the shared pool while the GPU producer thread is
       live, so a worker death aborts the run cleanly instead of forking after CUDA. (Pre-4b this was
       gated to ``choices < 4096``.)

    Results are committed strictly in sample order on the main thread, so ``result_store`` ordering and
    ``save_every`` checkpoints are byte-identical to the serial engine, and a save only ever sees
    contiguous committed problems (drain-before-save falls out for free). If any precondition fails the
    engine transparently falls back to the serial :class:`EvaluationEngine`.
    """

    def __init__(
        self,
        data_source: Any,
        model_adapter: Any,
        *,
        result_store: Any | None = None,
        queue_depth: int = 1,
    ) -> None:
        super().__init__(data_source, model_adapter, result_store=result_store)
        self._queue_depth = max(1, int(queue_depth))

    # ------------------------------------------------------------------ gate
    def _overlap_supported(self) -> tuple[bool, str]:
        """Return ``(ok, reason)``: whether the overlap preconditions hold, else why not."""
        adapter = self.model_adapter
        if not (hasattr(adapter, "generate_phase") and hasattr(adapter, "refine_extract_phase")):
            return False, "model adapter does not expose the generate/refine phase split"
        model = getattr(adapter, "model", None)
        if model is None:
            return False, "adapter has no .model"
        if getattr(model, "_refine_pool", None) is None:
            return False, "no persistent refine pool (load the model with persistent_refine_pool=True)"
        if float(getattr(model, "prune_constant_budget", 0.0) or 0.0) != 0.0:
            return False, "prune_constant_budget != 0 (constant pruning would issue a GPU rescore off the owner thread)"
        gen_cfg = getattr(model, "generation_config", None)
        method = getattr(gen_cfg, "method", None)
        if method != "softmax_sampling":
            return False, f"generation method {method!r} is not 'softmax_sampling'"

        # Step 4b: choices >= _SIMPLIFY_PARALLEL_THRESHOLD is NO LONGER gated off. At high c the
        # producer's post-generation simplify pass and the consumer's refine share model._refine_pool,
        # but generation dominates so heavily there that refine(N) finishes early in gen(N+1)'s window
        # and simplify(N+1) then runs effectively uncontended (pipeline = gen+simplify = the ceiling).
        # Concurrent pool use is race-free: refine ships per-job X/y (no shared _GLOBAL_REFINEMENT_DATA),
        # simplify is a pure function, and ProcessPoolExecutor.map is thread-safe across threads. Under
        # _overlap_mode BOTH the refine (_fit_refine) and simplify (_parallel_build_simplify_map) paths
        # RE-RAISE a BrokenProcessPool instead of closing/re-forking the shared pool while the GPU
        # producer thread is live, so a worker death aborts the run cleanly (no fork-after-CUDA).
        return True, ""

    # -------------------------------------------------------------- dispatch
    def run(
        self,
        *,
        limit: Optional[int] = None,
        save_every: Optional[int] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
        progress: bool = True,
        log_placeholders: bool = True,
        summary_interval: Optional[int] = 50,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, list[Any]]:
        ok, reason = self._overlap_supported()
        if not ok:
            warnings.warn(
                f"OverlappedEvaluationEngine: overlap disabled ({reason}); running the serial engine.",
                RuntimeWarning,
            )
            return super().run(
                limit=limit, save_every=save_every, output_path=output_path, verbose=verbose,
                progress=progress, log_placeholders=log_placeholders, summary_interval=summary_interval,
                meta=meta,
            )
        return self._run_overlapped(
            limit=limit, save_every=save_every, output_path=output_path, verbose=verbose,
            progress=progress, log_placeholders=log_placeholders, summary_interval=summary_interval,
            meta=meta,
        )

    # ---------------------------------------------------------- overlap loop
    def _run_overlapped(
        self,
        *,
        limit: Optional[int],
        save_every: Optional[int],
        output_path: Optional[str],
        verbose: bool,
        progress: bool,
        log_placeholders: bool,
        summary_interval: Optional[int],
        meta: Optional[Mapping[str, Any]],
    ) -> dict[str, list[Any]]:
        if save_every is not None and output_path is None:
            raise ValueError("output_path must be provided when save_every is configured")

        resolved_output: Optional[Path] = None
        if output_path is not None:
            resolved_output = Path(substitute_root_path(output_path))

        prepare_adapter = getattr(self.model_adapter, "prepare", None)
        if callable(prepare_adapter):
            prepare_adapter(data_source=self.data_source)
        prepare_source = getattr(self.data_source, "prepare", None)
        if callable(prepare_source):
            prepare_source(adapter=self.model_adapter)

        existing_results = self.result_store.size
        pending_target = limit if limit is not None else getattr(self.data_source, "size_hint", lambda: None)()
        if pending_target is not None:
            pending_target = max(0, int(pending_target))
        overall_target = None if pending_target is None else existing_results + pending_target

        progress_bar = None
        if progress and verbose:
            progress_bar = tqdm(total=pending_target, desc="Evaluating (overlapped)", smoothing=0.0)

        tracker: _EvaluationProgressTracker | None = None
        if log_placeholders or (summary_interval is not None and summary_interval > 0):
            tracker = _EvaluationProgressTracker(
                result_store=self.result_store,
                total_target=overall_target,
                logger=lambda message: self._log_message(message, progress_bar),
                log_placeholders=log_placeholders,
            )
            tracker.print_summary("Starting state")

        out_queue: queue.Queue = queue.Queue(maxsize=self._queue_depth)
        abort_event = threading.Event()
        exc_box: list[BaseException] = []

        # Signal the model that a GPU-owner thread is live: _fit_refine then keeps all per-candidate
        # work in forked pool workers (RNG-isolated) and re-raises a pool break instead of forking a
        # fresh pool on this thread (fork-after-CUDA-while-a-thread-is-live hazard). Reset in `finally`.
        model = getattr(self.model_adapter, "model", None)
        prior_overlap_mode = getattr(model, "_overlap_mode", False) if model is not None else False
        if model is not None:
            model._overlap_mode = True

        producer = threading.Thread(
            target=self._producer_loop,
            args=(out_queue, abort_event, exc_box, limit),
            name="overlap-gpu-producer",
            daemon=True,
        )
        producer.start()

        processed = 0
        pool_broke: BrokenProcessPool | None = None
        try:
            while True:
                item = out_queue.get()
                if item is None:  # end-of-stream sentinel
                    break

                kind = item[0]
                sample = item[1]
                if kind == "placeholder":
                    result: Any = self._build_placeholder_record(sample)
                elif kind == "gen_error":
                    result = self._handle_evaluation_exception(sample, item[2])
                else:  # "item"
                    _, sample, record, gen_state, fit_t0 = item
                    try:
                        eval_result = self.model_adapter.refine_extract_phase(sample, record, gen_state, fit_t0)
                        result = eval_result.to_mapping() if hasattr(eval_result, "to_mapping") else eval_result
                        if not isinstance(result, dict):
                            raise TypeError("Model adapters must return dict-like results")
                    except BrokenProcessPool as exc:
                        # A refine worker died. We must NOT fork a replacement pool while the producer
                        # thread is live; stop consuming and tear the overlap down (see post-loop).
                        pool_broke = exc
                        break
                    except Exception as exc:  # pragma: no cover - mirrors the serial defensive guard
                        result = self._handle_evaluation_exception(sample, exc)

                self.result_store.append(result)
                if tracker is not None:
                    tracker.record(result)
                processed += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                if save_every is not None and processed % save_every == 0 and resolved_output is not None:
                    self.result_store.save(resolved_output, meta=meta)
                if (
                    tracker is not None
                    and summary_interval is not None
                    and summary_interval > 0
                    and processed % summary_interval == 0
                ):
                    tracker.print_summary(
                        f"Progress after {processed} new samples (overall {self.result_store.size})"
                    )
        finally:
            # Tell the producer to stop and unblock it if it is parked on a full queue, then join until
            # it is actually dead: draining each iteration keeps its put() from wedging (so this cannot
            # deadlock), and waiting for true death guarantees no live GPU thread survives run().
            abort_event.set()
            while producer.is_alive():
                self._drain_queue(out_queue)
                producer.join(timeout=0.1)
            if model is not None:
                model._overlap_mode = prior_overlap_mode
            if progress_bar is not None:
                progress_bar.close()

        if pool_broke is not None:
            # Checkpoint the contiguous committed results, then surface an actionable error: the overlap
            # engine deliberately does not auto-recover a broken pool (unlike the serial per-call-fork
            # path) because that would fork while the GPU producer thread was live.
            if resolved_output is not None:
                self.result_store.save(resolved_output, meta=meta)
            raise RuntimeError(
                "OverlappedEvaluationEngine: the persistent refine pool broke mid-run (a refine worker "
                f"died) after committing {processed} problem(s)"
                + (" (checkpoint saved)" if resolved_output is not None else "")
                + ". The overlap engine does not fork a replacement pool while the GPU producer thread "
                "is live; re-run the remainder with the serial EvaluationEngine or load with "
                "persistent_refine_pool=False."
            ) from pool_broke

        if exc_box:
            # A producer-side failure (CUDA OOM in generate, or a producer-side BrokenProcessPool from
            # the overlap simplify pass -- the dominant high-c abort timing Step 4b enables). Checkpoint
            # the contiguous committed results FIRST, symmetric with the pool_broke path above
            # (:486-503): without this, save_every=None + output_path set would discard the entire
            # committed multi-hour run on this path. Safe + no double-save: the producer is joined
            # (:478-484) and the consumer has exited its loop, so the main thread is the sole
            # result_store accessor, and pool_broke (:486) and exc_box reaches are mutually exclusive.
            if resolved_output is not None:
                self.result_store.save(resolved_output, meta=meta)
            raise exc_box[0]

        final_snapshot = self.result_store.snapshot()
        if resolved_output is not None:
            self.result_store.save(resolved_output, meta=meta)
        if tracker is not None:
            tracker.print_summary("Final evaluation summary")
        return final_snapshot

    # ----------------------------------------------------------- GPU producer
    def _producer_loop(
        self,
        out_queue: queue.Queue,
        abort_event: threading.Event,
        exc_box: list[BaseException],
        limit: Optional[int],
    ) -> None:
        """Run the GPU generation phase in sample order; push GenStates to the consumer."""
        adapter = self.model_adapter
        produced = 0
        try:
            for sample in self.data_source:
                if abort_event.is_set():
                    break
                if limit is not None and produced >= limit:
                    break

                if getattr(sample, "is_placeholder", False):
                    item: Any = ("placeholder", sample)
                else:
                    try:
                        record, gen_state, fit_t0 = adapter.generate_phase(sample)
                    except BrokenProcessPool:
                        # Step 4b: the shared pool died during the producer's simplify pass. This is a
                        # run-level abort (propagate to the outer handler -> exc_box -> surfaced after a
                        # clean teardown + checkpoint), NOT a per-problem gen_error -- otherwise the
                        # producer would keep hammering the dead pool and checkpoint a run of spurious
                        # failures (acute on a multi-hour c=262144 run).
                        raise
                    except Exception as exc:  # noqa: BLE001 - unexpected gen error (expected ones are caught inside)
                        item = ("gen_error", sample, exc)
                    else:
                        if gen_state is not None:
                            # Bound GPU residency: under the prune==0 gate the scoring memory is unused
                            # by refinement, so drop it before queueing so gen(N+1) does not keep two
                            # problems' memory tensors resident at once (WSL2 spill guard).
                            gen_state.memory_for_scoring = None
                        item = ("item", sample, record, gen_state, fit_t0)

                produced += 1
                self._put_with_abort(out_queue, item, abort_event)
                if abort_event.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - propagate any producer failure to the consumer
            exc_box.append(exc)
        finally:
            self._put_with_abort(out_queue, None, abort_event)  # best-effort end-of-stream sentinel

    @staticmethod
    def _put_with_abort(
        out_queue: queue.Queue,
        item: Any,
        abort_event: threading.Event,
        poll: float = 0.2,
    ) -> None:
        """Put ``item`` on the queue, polling so a full queue never blocks past an abort.

        A payload is dropped if the consumer has already aborted (it is gone); the ``None`` sentinel is
        still attempted so a live consumer's ``get()`` always unblocks.
        """
        while True:
            if abort_event.is_set() and item is not None:
                return
            try:
                out_queue.put(item, timeout=poll)
                return
            except queue.Full:
                continue

    @staticmethod
    def _drain_queue(out_queue: queue.Queue) -> None:
        try:
            while True:
                out_queue.get_nowait()
        except queue.Empty:
            pass


__all__ = ["EvaluationEngine", "OverlappedEvaluationEngine"]
