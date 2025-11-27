"""General-purpose evaluation runner tying data sources and model adapters together."""
from __future__ import annotations

from collections import Counter
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
                    self.result_store.save(resolved_output)

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
            self.result_store.save(resolved_output)

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


__all__ = ["EvaluationEngine"]
