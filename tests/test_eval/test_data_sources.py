import itertools
from collections import Counter
import warnings
from pathlib import Path

import numpy as np
import pytest

from flash_ansr import get_path
from flash_ansr.benchmarks import FastSRBBenchmark
from flash_ansr.data import FlashANSRDataset
from flash_ansr.eval.data_sources import FastSRBSource, SkeletonDatasetSource
from flash_ansr.expressions.normalization import normalize_expression, normalize_skeleton
from flash_ansr.expressions.skeleton_pool import NoValidSampleFoundError


DATASET_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "test" / "dataset_val.yaml"
FASTSRB_BENCHMARK_PATH = get_path("data", "ansr-data", "test_set", "fastsrb", "expressions.yaml")


pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:Failed to sample deterministic skeleton after multiple attempts; skipping\.:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:FastSRB sample .* contains non-finite or out-of-range values\\. Resampling dataset\.:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:Skipping FastSRB equation .* after .* invalid datasets\.:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:FastSRBSource only yielded .* samples\.:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:overflow encountered in cast:RuntimeWarning"
    ),
]


def _make_dataset():
    return FlashANSRDataset.from_config(str(DATASET_CONFIG))


def test_deterministic_source_iterates_per_expression():
    dataset = _make_dataset()
    try:
        source = SkeletonDatasetSource(
            dataset,
            n_support=16,
            datasets_per_expression=2,
            target_size=3,
            device="cpu",
            datasets_random_seed=0,
        )
        source.prepare()
        samples = list(itertools.islice(iter(source), 2))
        assert len(samples) == 2
        for sample in samples:
            assert sample.n_support == 16
            assert sample.metadata["prediction_success"] is False
            np.testing.assert_equal(sample.x_support.shape[1], sample.x_validation.shape[1])
    finally:
        dataset.shutdown()


def test_deterministic_source_respects_skip():
    dataset = _make_dataset()
    try:
        base_source = SkeletonDatasetSource(
            dataset,
            n_support=8,
            datasets_per_expression=1,
            target_size=2,
            device="cpu",
        )
        base_source.prepare()
        base_first = next(iter(base_source)).metadata["skeleton_hash"]

        skip_source = SkeletonDatasetSource(
            dataset,
            n_support=8,
            datasets_per_expression=1,
            target_size=1,
            device="cpu",
            skip=1,
        )
        skip_source.prepare()
        skipped_first = next(iter(skip_source)).metadata["skeleton_hash"]

        if len(dataset.skeleton_pool) > 1:
            assert skipped_first != base_first
    finally:
        dataset.shutdown()


def test_skeleton_source_resume_state_advances_progress():
    dataset = _make_dataset()
    try:
        if len(dataset.skeleton_pool) < 3:
            pytest.skip("test dataset does not include enough skeletons")
        source = SkeletonDatasetSource(
            dataset,
            n_support=8,
            datasets_per_expression=1,
            target_size=3,
            device="cpu",
        )
        source.prepare()
        iterator = iter(source)
        first = next(iterator)
        second = next(iterator)
        state_payload = source.state_dict()["state"]
        expected_third = next(iterator)

        resume_source = SkeletonDatasetSource(
            dataset,
            n_support=8,
            datasets_per_expression=1,
            target_size=1,
            device="cpu",
            resume_state=state_payload,
        )
        resume_source.prepare()
        resumed_sample = next(iter(resume_source))

        hashes = (
            first.metadata["skeleton_hash"],
            second.metadata["skeleton_hash"],
            expected_third.metadata["skeleton_hash"],
        )
        resumed_hash = resumed_sample.metadata["skeleton_hash"]

        assert resumed_hash not in hashes[:2]
        assert resumed_hash == hashes[2]
    finally:
        dataset.shutdown()


def test_fastsrb_source_builds_skeleton_from_prefix():
    prefix = ["+", "3.0", "x1"]
    skeleton = FastSRBSource._build_skeleton_from_prefix(prefix)
    assert skeleton == ["+", "<constant>", "x1"]


def test_normalize_skeleton_standardizes_variables_and_constants():
    tokens = ["*", "v1", "+", "3.0", "v2", "<constant>"]
    assert normalize_skeleton(tokens) == ["*", "x1", "+", "<constant>", "x2", "<constant>"]


def test_normalize_expression_standardizes_variables_only():
    tokens = ["*", "v1", "+", "3.0", "v2"]
    assert normalize_expression(tokens) == ["*", "x1", "+", "3.0", "x2"]


def test_fastsrb_source_samples_all_expressions_without_invalid_power_warning():
    benchmark = FastSRBBenchmark(FASTSRB_BENCHMARK_PATH, random_state=1234)
    eq_ids = benchmark.equation_ids()
    repeats = 10
    target_size = len(eq_ids) * repeats

    source = FastSRBSource(
        benchmark,
        target_size=target_size,
        eq_ids=eq_ids,
        datasets_per_expression=repeats,
        support_points=4,
        sample_points=4,
        method="random",
        incremental=True,
        max_trials=4096,
        n_support_override=4,
    )
    source.prepare()

    counts = {eq_id: 0 for eq_id in eq_ids}
    placeholder_total = 0
    placeholder_reasons: Counter[str] = Counter()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=".*invalid value encountered in power.*",
            category=RuntimeWarning,
        )
        for sample in source:
            eq_id = sample.metadata["benchmark_eq_id"]
            if sample.metadata.get("placeholder"):
                placeholder_total += 1
                placeholder_reasons[sample.metadata.get("placeholder_reason")] += 1
            else:
                counts[eq_id] += 1

    skipped_total = sum(source.skipped_expressions.values())
    assert sum(counts.values()) + placeholder_total == target_size
    assert placeholder_reasons.get("max_trials_exhausted", 0) == skipped_total
    assert placeholder_reasons.get("source_exhausted", 0) == 0
    for eq_id, count in counts.items():
        skipped = source.skipped_expressions.get(eq_id, 0)
        assert count + skipped == repeats


def test_fastsrb_source_metadata_includes_variables():
    benchmark = FastSRBBenchmark(FASTSRB_BENCHMARK_PATH, random_state=0)
    source = FastSRBSource(
        benchmark,
        target_size=1,
        support_points=8,
        datasets_per_expression=1,
        method="random",
    )
    source.prepare()
    sample = next(iter(source))
    metadata = sample.metadata
    assert metadata["variables"] is not None
    assert metadata["variable_names"] is not None
    assert len(metadata["variables"]) == sample.x_support.shape[1]


def test_skeleton_dataset_metadata_uses_shared_builder():
    dataset = _make_dataset()
    try:
        source = SkeletonDatasetSource(dataset, n_support=8, target_size=1)
        source.prepare()
        sample = next(iter(source))
        metadata = sample.metadata
        assert metadata["variables"]
        assert metadata["variable_names"]
        assert metadata["skeleton"]
    finally:
        dataset.shutdown()


def test_skeleton_source_masks_unused_variables_when_zero_padding():
    dataset = _make_dataset()
    try:
        assert dataset.padding == "zero"
        source = SkeletonDatasetSource(dataset, n_support=8, target_size=1)
        source.prepare()
        sample = next(iter(source))

        skeleton = sample.metadata["skeleton"] or []
        pool_variables = list(dataset.skeleton_pool.variables)
        skeleton_vars = {token for token in skeleton if token in pool_variables}
        unused_variables = [var for var in pool_variables if var not in skeleton_vars]
        if not unused_variables:
            pytest.skip("Sample used all available variables; masking cannot be asserted")

        column_idx = pool_variables.index(unused_variables[0])
        assert np.all(sample.x_support[:, column_idx] == 0)
        if sample.x_validation.size:
            assert np.all(sample.x_validation[:, column_idx] == 0)
    finally:
        dataset.shutdown()


def test_skeleton_source_emits_placeholder_on_sampling_failure(monkeypatch):
    dataset = _make_dataset()
    try:
        source = SkeletonDatasetSource(
            dataset,
            n_support=8,
            target_size=1,
            device="cpu",
            datasets_per_expression=1,
        )
        source.prepare()

        def _always_fail(*args, **kwargs):  # noqa: ARG001
            raise NoValidSampleFoundError("forced failure")

        monkeypatch.setattr(dataset.skeleton_pool, "sample_data", _always_fail)

        sample = next(iter(source))
        assert sample.is_placeholder is True
        assert sample.metadata["placeholder"] is True
        assert sample.metadata["prediction_success"] is False
        assert sample.metadata["placeholder_reason"] in {"max_trials_exhausted", "skeleton_missing"}
    finally:
        dataset.shutdown()


def test_skeleton_source_shortfall_inserts_source_exhausted_placeholders(monkeypatch):
    dataset = _make_dataset()
    try:
        pool = dataset.skeleton_pool
        skeletons = sorted(list(pool.skeletons))
        if not skeletons:
            pytest.skip("test dataset does not include skeletons")

        limited = skeletons[:1]
        pool.skeletons = set(limited)
        pool.skeleton_codes = {s: pool.skeleton_codes[s] for s in limited}

        monkeypatch.setattr(SkeletonDatasetSource, "_populate_skeleton_pool", lambda self, needed: None)

        target_size = len(limited) + 2
        source = SkeletonDatasetSource(
            dataset,
            n_support=8,
            target_size=target_size,
            device="cpu",
            datasets_per_expression=1,
        )
        source.prepare()

        samples = list(source)
        placeholders = [sample for sample in samples if sample.metadata.get("placeholder")]

        assert len(samples) == target_size
        assert len(placeholders) == target_size - len(limited)
        assert {p.metadata.get("placeholder_reason") for p in placeholders} == {"source_exhausted"}
    finally:
        dataset.shutdown()


def test_fastsrb_source_emits_placeholder_when_resampling_exhausted(monkeypatch):
    benchmark = FastSRBBenchmark(FASTSRB_BENCHMARK_PATH, random_state=0)
    source = FastSRBSource(
        benchmark,
        target_size=1,
        datasets_per_expression=1,
        support_points=4,
        sample_points=4,
        method="random",
        max_trials=1,
    )
    source.prepare()

    monkeypatch.setattr(source, "_build_sample", lambda *args, **kwargs: None)

    placeholder = next(iter(source))
    assert placeholder.is_placeholder is True
    assert placeholder.metadata["placeholder"] is True
    assert placeholder.metadata["benchmark_eq_id"] is not None


def test_fastsrb_source_fills_shortfall_with_placeholders():
    benchmark = FastSRBBenchmark(FASTSRB_BENCHMARK_PATH, random_state=0)
    eq_ids = benchmark.equation_ids()[:2]
    target_size = len(eq_ids) + 3

    source = FastSRBSource(
        benchmark,
        target_size=target_size,
        eq_ids=eq_ids,
        datasets_per_expression=1,
        support_points=4,
        sample_points=4,
        method="random",
        incremental=True,
        max_trials=1024,
    )
    source.prepare()

    samples = list(source)
    placeholders = [s for s in samples if s.metadata.get("placeholder")]
    reasons = {p.metadata.get("placeholder_reason") for p in placeholders}

    assert len(samples) == target_size
    assert len(placeholders) >= target_size - len(eq_ids)
    assert reasons == {"source_exhausted"}


def test_fastsrb_shortfall_does_not_exceed_per_expression(monkeypatch):
    benchmark = FastSRBBenchmark(FASTSRB_BENCHMARK_PATH, random_state=0)
    eq_ids = benchmark.equation_ids()[:3]
    repeats = 2
    target_size = len(eq_ids) * repeats

    source = FastSRBSource(
        benchmark,
        target_size=target_size,
        eq_ids=eq_ids,
        datasets_per_expression=repeats,
        support_points=4,
        sample_points=4,
        method="random",
        incremental=True,
        max_trials=256,
    )
    source.prepare()

    provided = [
        (eq_ids[0], 0, {}),
        (eq_ids[0], 1, {}),
        (eq_ids[1], 0, {}),
        (eq_ids[1], 1, {}),
        (eq_ids[2], 0, {}),
    ]

    def fake_iter_samples(**kwargs):  # noqa: ARG001
        yield from provided

    def fake_build_sample(eq_id, sample_index, sample, noise_rng):  # noqa: ARG001
        record = source._build_placeholder_sample(eq_id, sample_index, reason="synthetic")
        record.metadata["placeholder"] = False
        record.metadata.pop("placeholder_reason", None)
        record.metadata["prediction_success"] = True
        record.is_placeholder = False
        record.placeholder_reason = None
        return record

    monkeypatch.setattr(source.benchmark, "iter_samples", fake_iter_samples)
    monkeypatch.setattr(source, "_build_sample", fake_build_sample)

    samples = list(source)
    counts = Counter(sample.metadata["benchmark_eq_id"] for sample in samples)

    assert len(samples) == target_size
    assert all(count <= repeats for count in counts.values())
    assert counts[eq_ids[2]] == repeats
