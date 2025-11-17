"""General-purpose evaluation runner tying data sources and model adapters together."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm import tqdm

from flash_ansr.eval.core import EvaluationDataSource, EvaluationModelAdapter, EvaluationResult, EvaluationSample
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

        total_target = limit if limit is not None else getattr(self.data_source, "size_hint", lambda: None)()

        iterator: Iterable[EvaluationSample] = self.data_source
        processed = 0

        progress_bar = None
        if progress and verbose:
            progress_bar = tqdm(total=total_target, desc="Evaluating", smoothing=0.0)

        try:
            for sample in iterator:
                if limit is not None and processed >= limit:
                    break

                if getattr(sample, "is_placeholder", False):
                    result = self._build_placeholder_record(sample)
                else:
                    result = self._evaluate_sample(sample)
                self.result_store.append(result)
                processed += 1

                if progress_bar is not None:
                    progress_bar.update(1)

                if save_every is not None and processed % save_every == 0 and resolved_output is not None:
                    self.result_store.save(resolved_output)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        final_snapshot = self.result_store.snapshot()

        if resolved_output is not None:
            self.result_store.save(resolved_output)

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


__all__ = ["EvaluationEngine"]
