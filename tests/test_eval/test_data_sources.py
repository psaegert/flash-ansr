import itertools
from pathlib import Path

import numpy as np

from flash_ansr.data import FlashANSRDataset
from flash_ansr.eval.data_sources import SkeletonDatasetSource


DATASET_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "test" / "dataset_val.yaml"


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
