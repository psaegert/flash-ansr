"""Evaluation data source implementations."""
from __future__ import annotations

import math
import random
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, TypeVar

import numpy as np
from simplipy.utils import numbers_to_constant

from flash_ansr.benchmarks import FastSRBBenchmark
from flash_ansr.data import FlashANSRDataset, FlashANSRPreprocessor
from flash_ansr.eval.core import EvaluationDataSource, EvaluationSample
from flash_ansr.eval.sample_metadata import build_base_metadata
from flash_ansr.expressions.skeleton_pool import NoValidSampleFoundError, SkeletonPool
from flash_ansr.expressions.token_ops import substitute_constants
from flash_ansr.expressions.normalization import (
    normalize_expression,
    normalize_skeleton,
)
from flash_ansr.utils.tensor_ops import mask_unused_variable_columns


T = TypeVar("T")


def _repeat_items_with_placeholders(
    items: Sequence[T],
    repeats: int,
    slots: int,
    *,
    start_offset: int = 0,
) -> list[T | None]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if slots <= 0:
        return []

    required = slots + max(0, start_offset)
    schedule: list[T | None] = []

    if items:
        for item in items:
            schedule.extend([item] * repeats)
            if len(schedule) >= required:
                break

    if len(schedule) < required:
        schedule.extend([None] * (required - len(schedule)))

    start = min(start_offset, len(schedule))
    trimmed = schedule[start:]
    if len(trimmed) > slots:
        trimmed = trimmed[:slots]
    return trimmed


class SkeletonDatasetSource(EvaluationDataSource):
    """Stream evaluation samples from a ``FlashANSRDataset`` skeleton pool."""

    def __init__(
        self,
        dataset: FlashANSRDataset,
        *,
        target_size: int | None = None,
        n_support: int | None = None,
        noise_level: float = 0.0,
        preprocess: bool = False,
        device: str = "cpu",
        iterator_buffer: int = 2,
        tokenizer_oov: str = "unk",
        datasets_per_expression: int | None = None,
        datasets_random_seed: int | None = None,
        max_trials: int | None = None,
        skip: int = 0,
        resume_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.n_support = n_support
        self.noise_level = noise_level
        self.preprocess = preprocess
        self.device = device
        self.iterator_buffer = max(1, iterator_buffer)
        self.tokenizer_oov = tokenizer_oov

        if datasets_per_expression is not None and datasets_per_expression < 1:
            raise ValueError("datasets_per_expression must be positive when provided")
        self.datasets_per_expression = datasets_per_expression or 1
        self.datasets_random_seed = datasets_random_seed
        self._skip_successes = max(0, skip)
        self._noise_rng: np.random.Generator | None = None
        if max_trials is None:
            self._max_trials = 100
        else:
            if max_trials < 1:
                raise ValueError("max_trials must be positive when provided")
            self._max_trials = int(max_trials)

        pool_size = len(self.dataset.skeleton_pool)
        per_expression = self.datasets_per_expression
        total_available = pool_size * per_expression if pool_size > 0 else None
        default_target = total_available if total_available is not None else 0
        raw_target = target_size if target_size is not None else default_target
        if total_available is None:
            self._target_size = max(0, raw_target)
        else:
            self._target_size = max(0, min(raw_target, total_available))
        self._total_available: int | None = total_available

        self._prepared = False
        self._max_n_support = self._resolve_max_n_support()
        self._skeleton_sequence: list[tuple[str, ...]] | None = None
        self._resume_expression_index = 0
        self._resume_dataset_offset = 0
        if resume_state is not None:
            self.load_state_dict(resume_state)
        elif self._skip_successes:
            self._apply_skip_offset(self._skip_successes)

    def size_hint(self) -> int:
        return self._target_size

    def prepare(self, *, adapter: Any | None = None) -> None:  # type: ignore[override]
        if self._prepared:
            return

        simplipy_engine = None
        if adapter is not None:
            get_engine = getattr(adapter, "get_simplipy_engine", None)
            if callable(get_engine):
                simplipy_engine = get_engine()
            elif hasattr(adapter, "simplipy_engine"):
                simplipy_engine = getattr(adapter, "simplipy_engine")

        if simplipy_engine is not None:
            self.dataset.skeleton_pool.simplipy_engine = simplipy_engine
            self.dataset.skeleton_pool.skeleton_codes = self.dataset.skeleton_pool.compile_codes(verbose=False)

            resolved_holdouts: list[SkeletonPool] = []
            for idx, holdout_pool in enumerate(self.dataset.skeleton_pool.holdout_pools):
                if isinstance(holdout_pool, str):
                    _, loaded_pool = SkeletonPool.load(holdout_pool)
                    holdout_pool = loaded_pool
                    self.dataset.skeleton_pool.holdout_pools[idx] = holdout_pool
                resolved_holdouts.append(holdout_pool)

            for holdout_pool in resolved_holdouts:
                holdout_pool.simplipy_engine = simplipy_engine
                holdout_pool.skeleton_codes = holdout_pool.compile_codes(verbose=False)
                self._apply_sample_strategy(holdout_pool)

        for holdout_pool in self.dataset.skeleton_pool.holdout_pools:
            if isinstance(holdout_pool, SkeletonPool):
                self._apply_sample_strategy(holdout_pool)

        self._apply_sample_strategy(self.dataset.skeleton_pool)

        if self.preprocess:
            self.dataset.preprocessor = FlashANSRPreprocessor(
                simplipy_engine=self.dataset.skeleton_pool.simplipy_engine,
                tokenizer=self.dataset.tokenizer,
            )

        self._ensure_skeleton_sequence()

        self._prepared = True

    def _apply_sample_strategy(self, pool: SkeletonPool | None) -> None:
        if pool is None:
            return
        strategy = getattr(pool, "sample_strategy", None)
        if isinstance(strategy, dict):
            strategy["max_tries"] = self._max_trials

    def __iter__(self) -> Iterator[EvaluationSample]:
        yield from self._iter_sequential()

    def _iter_sequential(self) -> Iterator[EvaluationSample]:
        per_expression = self.datasets_per_expression
        if self._target_size <= 0:
            return

        self._ensure_skeleton_sequence()
        if not self._skeleton_sequence:
            warnings.warn("No skeletons available for deterministic evaluation.", RuntimeWarning)
            placeholder_count = self._target_size
            placeholder_skeleton: tuple[str, ...] = tuple()
            for _ in range(placeholder_count):
                yield self._build_placeholder_sample(placeholder_skeleton, reason="source_exhausted")
            self._resume_expression_index = 0
            self._resume_dataset_offset = 0
            return

        if self.preprocess:
            warnings.warn(
                "Deterministic skeleton evaluation does not currently support preprocessing; skipping preprocess step.",
                RuntimeWarning,
            )

        start_expression = self._resume_expression_index
        start_rep = self._resume_dataset_offset
        start_offset = (start_expression * per_expression) + start_rep

        schedule = _repeat_items_with_placeholders(
            self._skeleton_sequence,
            per_expression,
            self._target_size,
            start_offset=start_offset,
        )

        schedule_placeholders = 0
        last_skeleton: tuple[str, ...] | None = None
        if start_expression < len(self._skeleton_sequence):
            last_skeleton = self._skeleton_sequence[start_expression]

        for skeleton in schedule:
            if skeleton is None:
                schedule_placeholders += 1
                placeholder_basis = last_skeleton or (self._skeleton_sequence[-1] if self._skeleton_sequence else tuple())
                yield self._build_placeholder_sample(placeholder_basis, reason="source_exhausted")
                continue

            last_skeleton = skeleton
            yield self._build_deterministic_sample(skeleton)

        if schedule_placeholders > 0:
            produced = self._target_size - schedule_placeholders
            warnings.warn(
                f"Deterministic SkeletonDatasetSource only yielded {produced} / {self._target_size} samples.",
                RuntimeWarning,
            )
            self._resume_expression_index = len(self._skeleton_sequence)
            self._resume_dataset_offset = 0
        else:
            total_processed = start_offset + self._target_size
            self._resume_expression_index = total_processed // per_expression
            self._resume_dataset_offset = total_processed % per_expression

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "type": "skeleton_dataset",
            "state": {
                "expression_index": self._resume_expression_index,
                "dataset_offset": self._resume_dataset_offset,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expression_index = int(state.get("expression_index", 0))
        dataset_offset = int(state.get("dataset_offset", 0))
        if dataset_offset < 0 or dataset_offset >= self.datasets_per_expression:
            dataset_offset = 0
        self._resume_expression_index = max(0, expression_index)
        self._resume_dataset_offset = max(0, dataset_offset)

    def _apply_skip_offset(self, skip: int) -> None:
        per_expression = self.datasets_per_expression
        self._resume_expression_index = skip // per_expression
        self._resume_dataset_offset = skip % per_expression

    def _build_deterministic_sample(self, skeleton: tuple[str, ...]) -> EvaluationSample:
        pool = self.dataset.skeleton_pool
        if not pool.skeleton_codes:
            pool.skeleton_codes = pool.compile_codes(verbose=False)

        if skeleton not in pool.skeleton_codes:
            pool.skeleton_codes = pool.compile_codes(verbose=False)
            if skeleton not in pool.skeleton_codes:
                warnings.warn("Skeleton missing from pool codes; skipping sample.", RuntimeWarning)
                return self._build_placeholder_sample(skeleton, reason="skeleton_missing")

        code, constants_tokens = pool.skeleton_codes[skeleton]
        n_constants = len(constants_tokens)
        n_points = self.n_support * 2 if self.n_support is not None else None

        for _ in range(8):
            try:
                x_all, y_all, literals = pool.sample_data(code, n_constants, n_support=n_points)
            except NoValidSampleFoundError:
                continue

            if x_all.size == 0 or y_all.size == 0:
                continue

            X_support, X_val, y_support, y_val = self._split_support_and_validation(x_all, y_all)
            if X_support.size == 0:
                continue

            y_support_noisy, y_val_noisy = self._apply_noise(y_support, y_val)

            mask_unused_variable_columns(
                arrays=(X_support, X_val),
                variables=self.dataset.skeleton_pool.variables,
                skeleton_tokens=skeleton,
                padding=getattr(self.dataset, "padding", None),
            )

            metadata = self._build_metadata_for_skeleton(
                skeleton=skeleton,
                literals=literals,
                X_support=X_support,
                y_support=y_support,
                y_support_noisy=y_support_noisy,
                X_val=X_val,
                y_val=y_val,
                y_val_noisy=y_val_noisy,
            )

            return EvaluationSample(
                x_support=X_support,
                y_support=y_support,
                x_validation=X_val,
                y_validation=y_val,
                y_support_noisy=y_support_noisy,
                y_validation_noisy=y_val_noisy,
                metadata=metadata,
            )

        warnings.warn("Failed to sample deterministic skeleton after multiple attempts; skipping.", RuntimeWarning)
        return self._build_placeholder_sample(skeleton, reason="max_trials_exhausted")

    def _ensure_skeleton_sequence(self) -> None:
        if self._skeleton_sequence is not None:
            return

        per_expression = self.datasets_per_expression
        pool = self.dataset.skeleton_pool
        processed_samples = (self._resume_expression_index * per_expression) + self._resume_dataset_offset
        required_samples = processed_samples + self._target_size
        required_expressions = math.ceil(required_samples / per_expression) if required_samples > 0 else 0

        skeletons = list(pool.skeletons)
        if required_expressions > len(skeletons):
            self._populate_skeleton_pool(required_expressions - len(skeletons))
            skeletons = list(pool.skeletons)

        skeletons.sort()

        self._skeleton_sequence = skeletons
        total_available = len(skeletons) * per_expression
        if self._total_available is None or self._total_available < total_available:
            self._total_available = total_available

    def _populate_skeleton_pool(self, needed: int) -> None:
        if needed <= 0:
            return

        pool = self.dataset.skeleton_pool
        np_state = None
        random_state = None
        if self.datasets_random_seed is not None:
            np_state = np.random.get_state()
            random_state = random.getstate()
            np.random.seed(self.datasets_random_seed)
            random.seed(self.datasets_random_seed)

        try:
            generated = 0
            while generated < needed:
                try:
                    skeleton, code, constants = pool.sample_skeleton(new=True)
                except NoValidSampleFoundError:
                    warnings.warn(
                        "Unable to sample additional skeletons for deterministic evaluation.",
                        RuntimeWarning,
                    )
                    break

                if skeleton in pool.skeletons:
                    continue

                pool.skeletons.add(skeleton)
                pool.skeleton_codes[skeleton] = (code, constants)
                generated += 1
        finally:
            if np_state is not None:
                np.random.set_state(np_state)
            if random_state is not None:
                random.setstate(random_state)

    def _split_support_and_validation(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        total_points = x_all.shape[0]
        if total_points == 0:
            return (
                np.empty((0, x_all.shape[1]), dtype=np.float32),
                np.empty((0, x_all.shape[1]), dtype=np.float32),
                np.empty((0, y_all.shape[1]), dtype=np.float32),
                np.empty((0, y_all.shape[1]), dtype=np.float32),
            )

        support_count = self.n_support if self.n_support is not None else total_points // 2
        support_count = max(1, min(support_count, total_points))
        if support_count == total_points and total_points > 1:
            support_count = total_points // 2

        X_support = x_all[:support_count].astype(np.float32, copy=True)
        y_support = y_all[:support_count].astype(np.float32, copy=True)

        if support_count < total_points:
            X_val = x_all[support_count:].astype(np.float32, copy=True)
            y_val = y_all[support_count:].astype(np.float32, copy=True)
        else:
            X_val = np.empty((0, x_all.shape[1]), dtype=np.float32)
            y_val = np.empty((0, y_all.shape[1]), dtype=np.float32)

        return X_support, X_val, y_support, y_val

    def _apply_noise(
        self,
        y_support: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.noise_level <= 0.0:
            return y_support.copy(), y_val.copy()

        rng = self._noise_rng
        if rng is None:
            rng = np.random.default_rng(self.datasets_random_seed)
            self._noise_rng = rng

        def _inject_noise(array: np.ndarray) -> np.ndarray:
            if array.size == 0:
                return array.copy()
            noisy = array.copy()
            y_std = float(np.std(noisy))
            if np.isfinite(y_std) and y_std > 0:
                noise = rng.standard_normal(size=noisy.shape).astype(np.float32)
                noisy = noisy + (self.noise_level * y_std * noise)
            return noisy

        return _inject_noise(y_support), _inject_noise(y_val)

    def _build_metadata_for_skeleton(
        self,
        *,
        skeleton: tuple[str, ...],
        literals: np.ndarray,
        X_support: np.ndarray,
        y_support: np.ndarray,
        y_support_noisy: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_val_noisy: np.ndarray,
    ) -> Dict[str, Any]:
        skeleton_list = normalize_skeleton(skeleton)
        if skeleton_list is None:
            raise ValueError("Skeleton tokens must be provided for metadata building")
        expression_tokens = substitute_constants(list(skeleton_list), values=literals, inplace=False)
        normalized_expression = normalize_expression(expression_tokens)
        input_ids = self._encode_input_ids(skeleton_list)
        tokenizer = self.dataset.tokenizer

        metadata = build_base_metadata(
            skeleton=skeleton_list,
            expression=normalized_expression,
            variables=list(self.dataset.skeleton_pool.variables),
            x_support=X_support,
            y_support=y_support,
            x_validation=X_val,
            y_validation=y_val,
            y_support_noisy=y_support_noisy,
            y_validation_noisy=y_val_noisy,
            noise_level=self.noise_level,
            skeleton_hash=skeleton_list,
            labels_decoded=tokenizer.decode(input_ids, special_tokens="<constant>"),
            complexity=len(expression_tokens) if expression_tokens else None,
        )

        metadata.update(
            {
                "input_ids": np.asarray(input_ids, dtype=np.int64),
                "labels": np.asarray(input_ids[1:], dtype=np.int64),
                "constants": self._format_constants(literals),
            }
        )
        return metadata

    def _build_placeholder_sample(self, skeleton: tuple[str, ...], reason: str) -> EvaluationSample:
        variables = list(self.dataset.skeleton_pool.variables)
        feature_dim = max(1, len(variables) if variables else 1)
        X_empty = np.empty((0, feature_dim), dtype=np.float32)
        y_empty = np.empty((0, 1), dtype=np.float32)
        skeleton_list = normalize_skeleton(skeleton) or list(skeleton) or None
        expression_tokens = list(skeleton_list) if skeleton_list is not None else None

        metadata = build_base_metadata(
            skeleton=skeleton_list,
            expression=expression_tokens,
            variables=variables or None,
            x_support=X_empty,
            y_support=y_empty,
            x_validation=X_empty.copy(),
            y_validation=y_empty.copy(),
            y_support_noisy=y_empty.copy(),
            y_validation_noisy=y_empty.copy(),
            noise_level=self.noise_level,
            skeleton_hash=skeleton_list,
            labels_decoded=None,
            complexity=None,
        )

        input_ids = None
        if skeleton_list:
            try:
                input_ids = self._encode_input_ids(list(skeleton_list))
            except Exception:
                input_ids = None

        metadata.update(
            {
                "input_ids": np.asarray(input_ids, dtype=np.int64) if input_ids is not None else None,
                "labels": np.asarray(input_ids[1:], dtype=np.int64) if input_ids is not None else None,
                "placeholder": True,
                "placeholder_reason": reason,
                "error": reason,
                "prediction_success": False,
            }
        )

        return EvaluationSample(
            x_support=X_empty,
            y_support=y_empty,
            x_validation=X_empty.copy(),
            y_validation=y_empty.copy(),
            y_support_noisy=y_empty.copy(),
            y_validation_noisy=y_empty.copy(),
            metadata=metadata,
            is_placeholder=True,
            placeholder_reason=reason,
        )

    def _encode_input_ids(self, skeleton_tokens: list[str]) -> list[int]:
        tokenizer = self.dataset.tokenizer
        body_tokens = skeleton_tokens
        if "<expression>" in tokenizer and "</expression>" in tokenizer:
            body_tokens = ["<expression>", *skeleton_tokens, "</expression>"]

        body_ids = tokenizer.encode(body_tokens, oov=self.tokenizer_oov)
        bos = tokenizer["<bos>"]
        eos = tokenizer["<eos>"]
        return [int(bos), *map(int, body_ids), int(eos)]

    def _format_constants(self, literals: np.ndarray) -> list[np.ndarray]:
        if literals is None or literals.size == 0:
            return []
        literals_array = np.asarray(literals, dtype=np.float32)
        return [literals_array.copy()]

    # ---------------------------------------------------------------------
    # Helper methods
    def _resolve_max_n_support(self) -> int:
        sampler = self.dataset.skeleton_pool.support_sampler
        max_support = sampler.configured_max_n_support
        if max_support is None and self.n_support is None:
            raise ValueError(
                "Support sampler configuration must define a maximum support size when evaluation does not "
                "override 'n_support'."
            )
        if self.n_support is None:
            return max_support * 2  # type: ignore[operator]
        return self.n_support * 2

    def _infer_sampling_support(self) -> int | None:
        return self.n_support * 2 if self.n_support is not None else None


class FastSRBSource(EvaluationDataSource):
    """Yield evaluation samples from the FastSRB benchmark specification."""

    def __init__(
        self,
        benchmark: FastSRBBenchmark,
        *,
        target_size: int,
        skip: int = 0,
        eq_ids: Sequence[str] | None = None,
        datasets_per_expression: int = 1,
        support_points: int = 100,
        sample_points: int | None = None,
        n_support_override: int | None = None,
        method: str = "random",
        max_trials: int = 100,
        incremental: bool = False,
        random_state: int | None = None,
        noise_level: float = 0.0,
    ) -> None:
        if support_points < 1:
            raise ValueError("support_points must be positive")
        if datasets_per_expression < 1:
            raise ValueError("datasets_per_expression must be positive")
        self.benchmark = benchmark
        self.eq_ids = list(eq_ids) if eq_ids is not None else None
        self.datasets_per_expression = int(datasets_per_expression)
        self.support_points = int(support_points)
        self.sample_points = int(sample_points or (self.support_points * 2))
        self.n_support_override = n_support_override
        self.method = method
        self.max_trials = max_trials
        self.incremental = incremental
        self.random_state = random_state
        self.noise_level = noise_level
        self._resample_rng: np.random.Generator | None = None
        self._resample_attempts = max(1, min(64, self.max_trials))
        self.skipped_expressions: dict[str, int] = {}

        self._target_size = max(0, int(target_size))
        self._skip = max(0, int(skip))

        base_eq_ids = self.eq_ids if self.eq_ids is not None else benchmark.equation_ids()
        self._eq_order = list(base_eq_ids)
        self._total_available = len(self._eq_order) * self.datasets_per_expression

        self._simplipy_engine = None

    def size_hint(self) -> int:
        available = max(0, self._total_available - self._skip)
        return min(self._target_size, available)

    def prepare(self, *, adapter: Any | None = None) -> None:  # type: ignore[override]
        simplipy_engine = None
        if adapter is not None:
            get_engine = getattr(adapter, "get_simplipy_engine", None)
            if callable(get_engine):
                simplipy_engine = get_engine()
            elif hasattr(adapter, "simplipy_engine"):
                simplipy_engine = getattr(adapter, "simplipy_engine")
        self._simplipy_engine = simplipy_engine

    def __iter__(self) -> Iterator[EvaluationSample]:
        iterate_rng = None
        if self.random_state is not None:
            iterate_rng = np.random.default_rng(self.random_state)

        self._resample_rng = iterate_rng

        schedule = self._build_schedule()
        if not schedule:
            return

        per_eq_counts: dict[str, int] = defaultdict(int)
        skip = min(self._skip, len(schedule))
        for eq_id in schedule[:skip]:
            if eq_id is not None:
                per_eq_counts[eq_id] += 1

        pending_slots = schedule[skip: skip + self._target_size]
        produced = 0
        noise_rng = None
        if self.noise_level > 0:
            if self.random_state is not None:
                noise_rng = np.random.default_rng(self.random_state)
            else:
                noise_rng = np.random.default_rng()

        for eq_id in pending_slots:
            if produced >= self._target_size:
                break

            if eq_id is None:
                yield self._build_placeholder_sample("__placeholder__", sample_index=-1, reason="source_exhausted")
                produced += 1
                continue

            sample_index = per_eq_counts[eq_id]
            per_eq_counts[eq_id] = sample_index + 1

            record = self._generate_sample(eq_id, sample_index, noise_rng)
            yield record
            produced += 1

        if produced < self._target_size:
            warnings.warn(
                f"FastSRBSource only yielded {produced} / {self._target_size} samples.",
                RuntimeWarning,
            )
            while produced < self._target_size:
                yield self._build_placeholder_sample("__placeholder__", sample_index=-1, reason="source_exhausted")
                produced += 1

    def _build_schedule(self) -> list[str | None]:
        slots_needed = max(0, self._target_size + self._skip)
        if slots_needed == 0:
            return []
        eq_order = list(self._eq_order) or list(self.benchmark.equation_ids())
        return _repeat_items_with_placeholders(
            eq_order,
            self.datasets_per_expression,
            slots_needed,
        )

    def _generate_sample(
        self,
        eq_id: str,
        sample_index: int,
        noise_rng: np.random.Generator | None,
    ) -> EvaluationSample:
        attempts = 0
        rng = self._resample_rng if self._resample_rng is not None else self.random_state
        while attempts < self._resample_attempts:
            try:
                candidate = self.benchmark.sample(
                    eq_id,
                    n_points=self.sample_points,
                    method=self.method,
                    max_trials=self.max_trials,
                    incremental=self.incremental,
                    random_state=rng,
                )
            except Exception as exc:  # pragma: no cover - defensive against benchmark issues
                warnings.warn(
                    f"FastSRBSource failed to sample {eq_id}: {exc}.", RuntimeWarning
                )
                break

            result = self._build_sample(eq_id, sample_index, candidate, noise_rng)
            if result is not None:
                return result
            attempts += 1

        total_attempts = attempts or self._resample_attempts
        warnings.warn(
            f"Skipping FastSRB equation {eq_id} after {total_attempts} invalid datasets.",
            RuntimeWarning,
        )
        self.skipped_expressions[eq_id] = self.skipped_expressions.get(eq_id, 0) + 1
        return self._build_placeholder_sample(eq_id, sample_index, reason="max_trials_exhausted")

    def _build_sample(
        self,
        eq_id: str,
        sample_index: int,
        sample: Mapping[str, Any],
        noise_rng: np.random.Generator | None,
    ) -> EvaluationSample | None:
        metadata_block = sample.get("metadata", {})
        data_block = sample.get("data", {})
        inputs = np.asarray(data_block.get("X"), dtype=np.float64)
        targets = np.asarray(data_block.get("y"), dtype=np.float64)

        if not self._is_valid_numeric_array(inputs) or not self._is_valid_numeric_array(targets):
            warnings.warn(
                f"FastSRB sample {eq_id} contains non-finite or out-of-range values. Resampling dataset.",
                RuntimeWarning,
            )
            return None

        inputs = inputs.astype(np.float32, copy=False)
        targets = targets.astype(np.float32, copy=False)

        if inputs.ndim != 2:
            raise ValueError("FastSRB sample inputs must be 2D")

        total_points = inputs.shape[0]
        if total_points == 0:
            warnings.warn(f"Sample for {eq_id} has no data. Skipping.")
            return None

        desired_support = min(self.support_points, total_points)
        if self.n_support_override is not None:
            desired_support = min(desired_support, self.n_support_override, total_points)

        n_support = max(1, desired_support)

        X_support = inputs[:n_support].copy()
        y_support = targets[:n_support].copy()

        X_val = inputs[n_support:].copy()
        y_val = targets[n_support:].copy()

        if self.noise_level > 0 and noise_rng is not None:
            y_std = float(np.std(y_support))
            if np.isfinite(y_std) and y_std > 0:
                noise = noise_rng.normal(size=y_support.shape)
                y_support_noisy = y_support + self.noise_level * y_std * noise.astype(np.float32)
            else:
                y_support_noisy = y_support.copy()
        else:
            y_support_noisy = y_support.copy()

        if not np.all(np.isfinite(y_support_noisy)):
            warnings.warn("Noisy targets contain non-finite values. Skipping sample.")
            return None

        prepared_prefix = metadata_block.get("prepared_prefix")
        if isinstance(prepared_prefix, tuple):
            ground_truth_prefix = list(prepared_prefix)
        elif isinstance(prepared_prefix, list):
            ground_truth_prefix = list(prepared_prefix)
        else:
            prepared_expr = metadata_block.get("prepared_normalized")
            fallback_expr = metadata_block.get("prepared") or metadata_block.get("raw")
            ground_truth_expr_candidate = prepared_expr or fallback_expr
            if not ground_truth_expr_candidate:
                raise ValueError("FastSRB metadata is missing a prepared expression.")
            ground_truth_prefix = self._parse_ground_truth(ground_truth_expr_candidate)
        ground_truth_expr = metadata_block.get("prepared_normalized") or metadata_block.get("prepared") or metadata_block.get("raw")
        if ground_truth_expr is None:
            if self._simplipy_engine is None:
                raise RuntimeError("FastSRBSource needs a SimpliPy engine to reconstruct prepared expressions.")
            ground_truth_expr = self._simplipy_engine.prefix_to_infix(ground_truth_prefix, realization=False)

        variables_block = sample.get("variables", {})
        variable_names = self._extract_variable_names(variables_block)
        skeleton_tokens = normalize_skeleton(self._build_skeleton_from_prefix(ground_truth_prefix))
        normalized_expression = normalize_expression(ground_truth_prefix)

        fallback_variables = variable_names or [f"x{idx + 1}" for idx in range(X_support.shape[1])]
        metadata = build_base_metadata(
            skeleton=skeleton_tokens,
            expression=normalized_expression,
            variables=fallback_variables,
            x_support=X_support,
            y_support=y_support,
            x_validation=X_val,
            y_validation=y_val,
            y_support_noisy=y_support_noisy,
            y_validation_noisy=y_val.copy(),
            noise_level=self.noise_level,
            labels_decoded=normalized_expression,
            complexity=len(normalized_expression) if normalized_expression else None,
        )

        metadata.update(
            {
                "input_ids": None,
                "labels": None,
                "constants": [],
                "benchmark_eq_id": eq_id,
                "benchmark_sample_index": int(sample_index),
                "benchmark_metadata": metadata_block,
                "benchmark_n_points": int(total_points),
                "benchmark_support_points": int(self.support_points),
                "benchmark_method": self.method,
                "ground_truth_infix": ground_truth_expr,
                "ground_truth_prefix": ground_truth_prefix.copy() if ground_truth_prefix else None,
                "variable_names": variable_names or fallback_variables,
            }
        )

        return EvaluationSample(
            x_support=X_support,
            y_support=y_support,
            x_validation=X_val,
            y_validation=y_val,
            y_support_noisy=y_support_noisy,
            y_validation_noisy=y_val.copy(),
            metadata=metadata,
        )

    def _build_placeholder_sample(self, eq_id: str, sample_index: int, reason: str) -> EvaluationSample:
        feature_dim = 1
        X_empty = np.empty((0, feature_dim), dtype=np.float32)
        y_empty = np.empty((0, 1), dtype=np.float32)
        metadata = build_base_metadata(
            skeleton=None,
            expression=None,
            variables=None,
            x_support=X_empty,
            y_support=y_empty,
            x_validation=X_empty.copy(),
            y_validation=y_empty.copy(),
            y_support_noisy=y_empty.copy(),
            y_validation_noisy=y_empty.copy(),
            noise_level=self.noise_level,
            skeleton_hash=None,
            labels_decoded=None,
            complexity=None,
        )

        metadata.update(
            {
                "input_ids": None,
                "labels": None,
                "constants": [],
                "benchmark_eq_id": eq_id,
                "benchmark_sample_index": int(sample_index),
                "benchmark_metadata": None,
                "benchmark_n_points": 0,
                "benchmark_support_points": int(self.support_points),
                "benchmark_method": self.method,
                "ground_truth_infix": None,
                "ground_truth_prefix": None,
                "variable_names": None,
                "placeholder": True,
                "placeholder_reason": reason,
                "error": reason,
                "prediction_success": False,
            }
        )

        return EvaluationSample(
            x_support=X_empty,
            y_support=y_empty,
            x_validation=X_empty.copy(),
            y_validation=y_empty.copy(),
            y_support_noisy=y_empty.copy(),
            y_validation_noisy=y_empty.copy(),
            metadata=metadata,
            is_placeholder=True,
            placeholder_reason=reason,
        )

    @staticmethod
    def _build_skeleton_from_prefix(prefix: Sequence[str] | None) -> list[str] | None:
        if prefix is None:
            return None
        prefix_list = list(prefix)
        if not prefix_list:
            return None
        try:
            normalized = numbers_to_constant(prefix_list)
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.warn(
                f"Failed to convert FastSRB prefix to skeleton via numbers_to_constant: {exc}. Falling back to prefix tokens."
            )
            return prefix_list
        return list(normalized)

    def _parse_ground_truth(self, expression: Any) -> list[str]:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("FastSRB ground truth expression is missing or empty.")
        normalized_expression = expression.replace("^", "**")
        normalized = re.sub(r"\bv(\d+)\b", lambda match: f"x{match.group(1)}", normalized_expression)
        if self._simplipy_engine is None:
            raise RuntimeError("FastSRBSource requires a SimpliPy engine to parse ground truth expressions.")
        parsed = self._simplipy_engine.parse(normalized, mask_numbers=True)
        return list(parsed)

    @staticmethod
    def _extract_variable_names(variables_block: Mapping[str, Any] | None) -> list[str] | None:
        if not isinstance(variables_block, Mapping):
            return None
        inputs_meta = variables_block.get("inputs")
        if not isinstance(inputs_meta, Iterable):
            return None
        names: list[str] = []
        for idx, meta in enumerate(inputs_meta):
            if isinstance(meta, Mapping):
                names.append(str(meta.get("name", f"x{idx + 1}")))
            else:
                names.append(f"x{idx + 1}")
        return names or None

    @staticmethod
    def _is_valid_numeric_array(array: np.ndarray) -> bool:
        if array.size == 0:
            return True
        if not np.all(np.isfinite(array)):
            return False
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        float32_info = np.finfo(np.float32)
        return bool((min_val >= float32_info.min) and (max_val <= float32_info.max))


__all__ = ["SkeletonDatasetSource", "FastSRBSource"]
