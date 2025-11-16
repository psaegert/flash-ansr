import itertools
import warnings
from pathlib import Path

import numpy as np

from flash_ansr import get_path
from flash_ansr.benchmarks import FastSRBBenchmark
from flash_ansr.data import FlashANSRDataset
from flash_ansr.eval.data_sources import FastSRBSource, SkeletonDatasetSource
from flash_ansr.expressions.normalization import normalize_expression, normalize_skeleton


DATASET_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "test" / "dataset_val.yaml"
FASTSRB_BENCHMARK_PATH = get_path("data", "ansr-data", "test_set", "fastsrb", "expressions.yaml")


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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=".*invalid value encountered in power.*",
            category=RuntimeWarning,
        )
        for sample in source:
            eq_id = sample.metadata["benchmark_eq_id"]
            counts[eq_id] += 1

    skipped_total = sum(source.skipped_expressions.values())
    assert sum(counts.values()) + skipped_total == target_size
    for eq_id, count in counts.items():
        skipped = source.skipped_expressions.get(eq_id, 0)
        assert count + skipped == repeats
