"""A drained stream can keep its worker pool alive for the next identical iterate.

Building the pool makes every worker parse the source's catalogs, which dominates a
short run -- validation pays it on every pass otherwise. Reuse is guarded two ways: an
abandoned generator still shuts down (jobs are in flight), and a pool built for other
settings refuses to serve them.
"""
import pytest
from symbolic_data import ProblemSource, load_config

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.model.tokenizer import Tokenizer


def _dataset() -> FlashANSRDataset:
    catalog = load_config(get_path('configs', 'test', 'catalog_val.yaml'))
    source = ProblemSource({
        'catalog': catalog,
        'sampling': {'n_support': 3, 'n_validation': 0, 'noise': 0.0},
    })
    tokenizer = Tokenizer.from_config(get_path('configs', 'test', 'tokenizer.yaml'))
    return FlashANSRDataset(source=source, tokenizer=tokenizer, padding='zero')


def test_a_drained_stream_keeps_its_pool() -> None:
    dataset = _dataset()
    try:
        for _ in dataset.iterate(steps=2, batch_size=2, num_workers=1, keep_alive=True):
            pass
        assert dataset._stream.is_initialized, "a drained keep_alive stream shut its pool down"
        # The second pass reuses it and still yields batches.
        batches = list(dataset.iterate(steps=2, batch_size=2, num_workers=1, keep_alive=True))
        assert len(batches) == 2
        assert dataset._stream.is_initialized
    finally:
        dataset.shutdown()
    assert not dataset._stream.is_initialized


def test_an_abandoned_stream_shuts_down_anyway() -> None:
    dataset = _dataset()
    try:
        for _ in dataset.iterate(steps=4, batch_size=2, num_workers=1, keep_alive=True):
            break  # jobs are still in flight -- this stream is not reusable
        assert not dataset._stream.is_initialized
    finally:
        dataset.shutdown()


def test_a_live_pool_refuses_different_settings() -> None:
    dataset = _dataset()
    try:
        for _ in dataset.iterate(steps=2, batch_size=2, num_workers=1, keep_alive=True):
            pass
        with pytest.raises(RuntimeError, match="different settings"):
            for _ in dataset.iterate(steps=2, batch_size=8, num_workers=1, keep_alive=True):
                pass
    finally:
        dataset.shutdown()


def test_the_default_still_shuts_down() -> None:
    dataset = _dataset()
    for _ in dataset.iterate(steps=2, batch_size=2, num_workers=1):
        pass
    assert not dataset._stream.is_initialized
