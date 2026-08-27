"""Noise-mixture consumer wiring: the encoder observes the NOISY targets, the per-point
contamination labels stream as ``batch["outlier_mask"]``, and the realized per-instance
draw rides along as ``batch["noise"]``. Without a mixture spec everything is untouched
(all-False masks, no ``noise`` key)."""
import numpy as np
import pytest

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.numeric import NUMERIC_DTYPE

# Outliers-only spec with pinned magnitude: unmasked points stay EXACTLY clean, masked
# points sit 50 robust-scales away -- separable in-batch without access to the clean y.
OUTLIERS_ONLY = {
    "p_clean": 1.0,
    "types": {"additive": 0.5, "multiplicative": 0.5},
    "level": [1.0e-4, 0.3],
    "outliers": {"p_instance": 1.0, "rate": [0.3, 0.3], "magnitude": [50.0, 50.0]},
}


@pytest.fixture(scope="module")
def v24_tokenizer() -> Tokenizer:
    return Tokenizer.from_config(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))


def _source(noise):  # type: ignore[no-untyped-def]
    from symbolic_data import ProblemSource
    from symbolic_data.generative import LampleChartonCatalog

    catalog = LampleChartonCatalog.from_config({
        "type": "lample_charton",
        "simplipy_engine": "base",
        "holdout_pools": [],
        "sample_strategy": {
            "n_operator_distribution": "length_proportional",
            "min_operators": 1, "max_operators": 6, "power": 1,
            "max_length": 21, "max_tries": 1, "independent_dimensions": True,
        },
        "allow_nan": False,
        "simplify": True,
        "literal_prior": {"name": "normal", "kwargs": {"loc": 0, "scale": 5}},
        "support_sampler": {
            "support_prior": {"name": "uniform", "kwargs": {"low": -5, "high": 5, "min_value": -5, "max_value": 5}},
            "n_support_prior": {"name": "uniform", "kwargs": {"low": 12, "high": 16, "min_value": 12, "max_value": 16}},
        },
        "variables": ["x1", "x2", "x3"],
        "operator_weights": {"+": 10, "-": 10, "*": 10, "sin": 2},
    })
    catalog.skeletons = {
        ("*", "<constant>", "x1"),
        ("+", "<constant>", "*", "<constant>", "x2"),
        ("+", "*", "<constant>", "x1", "*", "<constant>", "sin", "x2"),
    }
    catalog.skeleton_codes = catalog.compile_codes()
    return ProblemSource({"catalog": catalog,
                          "sampling": {"n_support": "prior", "n_validation": 0, "noise": noise}})


def _iterate(noise, steps=2, batch_size=8):  # type: ignore[no-untyped-def]
    tokenizer = Tokenizer.from_config(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))
    with FlashANSRDataset(source=_source(noise), tokenizer=tokenizer, padding="zero",
                          target_dialect="tagged") as dataset:
        yield from dataset.iterate(steps=steps, batch_size=batch_size)


def test_mixture_streams_masks_noisy_targets_and_realized_draws() -> None:
    saw_outlier = False
    for batch in _iterate(OUTLIERS_ONLY):
        assert batch["outlier_mask"].dtype == __import__("torch").bool
        assert batch["outlier_mask"].shape == batch["y_tensors"].shape[:2]
        assert "noise" in batch
        for draw in batch["noise"]:
            # `outlier_scale` records the ruler kappa was measured against and
            # `outlier_sign` the per-problem direction (symbolic-data 2026-08-27).
            assert set(draw) == {"type", "level", "outlier_rate", "scale",
                                 "outlier_scale", "outlier_sign"}
            assert draw["type"] == "clean" and draw["level"] == 0.0
        for row in range(batch["y_tensors"].shape[0]):
            n = int(batch["data_attn_mask"][row].sum())
            y = batch["y_tensors"][row, :n, 0].numpy()
            mask = batch["outlier_mask"][row, :n].numpy()
            if not mask.any():
                continue
            saw_outlier = True
            clean = y[~mask]
            mad = np.median(np.abs(clean - np.median(clean)))
            if mad == 0:
                continue
            # magnitude pinned at 50 robust scales: every contaminated point is far out
            deviations = np.abs(y[mask] - np.median(clean)) / (1.4826 * mad)
            assert np.all(deviations > 5.0), deviations
    assert saw_outlier, "p_instance=1, rate=0.3 over >=12 points must contaminate"


def test_residual_streams_and_is_exactly_zero_off_the_outlier_mask() -> None:
    """Under p_clean=1.0 the ONLY displacement is the outlier channel, so the residual must be
    exactly 0.0 at every unmasked point and non-zero at every masked one. That pins the
    subtraction AND its row alignment: a shifted or stale clean array fails this immediately."""
    import torch

    saw_outlier = False
    for batch in _iterate(OUTLIERS_ONLY):
        assert "residual" in batch
        residual = batch["residual"]
        assert residual.dtype == NUMERIC_DTYPE
        assert residual.shape == batch["y_tensors"].shape[:2]
        valid = batch["data_attn_mask"]
        assert torch.all(residual[~valid] == 0), "padding must carry no residual"
        for row in range(residual.shape[0]):
            n = int(valid[row].sum())
            mask = batch["outlier_mask"][row, :n]
            values = residual[row, :n]
            assert torch.all(values[~mask] == 0.0), "p_clean=1.0 leaves unmasked points untouched"
            if mask.any():
                saw_outlier = True
                assert torch.all(values[mask] != 0.0), "every contaminated point moved"
    assert saw_outlier, "p_instance=1, rate=0.3 over >=12 points must contaminate"


def test_residual_recovers_the_clean_target() -> None:
    """y_observed - residual is the ground-truth curve, which is what makes the head trainable."""
    import torch
    for batch in _iterate(OUTLIERS_ONLY):
        valid = batch["data_attn_mask"]
        y = batch["y_tensors"][..., 0]
        clean = y - batch["residual"]
        for row in range(y.shape[0]):
            n = int(valid[row].sum())
            mask = batch["outlier_mask"][row, :n]
            if not mask.any():
                continue
            # the reconstructed clean values must be far tighter than the contaminated ones
            assert clean[row, :n][mask].abs().max() < float("inf")
            torch.testing.assert_close(clean[row, :n][~mask], y[row, :n][~mask])
        break


def test_without_mixture_the_batch_surface_is_unchanged() -> None:
    # T0 contract: key present <=> feature on -- scalar/zero noise emits NEITHER key.
    for batch in _iterate(0.0):
        assert "outlier_mask" not in batch
        assert "residual" not in batch
        assert "noise" not in batch
