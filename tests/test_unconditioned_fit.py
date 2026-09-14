"""The unconditioned decode -- ``SoftmaxSamplingConfig(guidance_weight=0.0)``, the prior-decode control
documented in Getting Started -- routes the learned ``null_memory`` into the decoder as ONE memory row
that the sampler broadcasts over its draws. The fit path used to expand it over the support points
instead (its input is ``(points, features)``, not a batch), so the decoder's cross-attention refused every
call with more than one data point (0.16.1)."""
import numpy as np
import pytest
import torch

from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing import CapabilityUnavailable
from flash_ansr.utils.config_io import load_config


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("base", install=True)


def _estimator(engine, tokenizer, optional_condition: bool, guidance_weight: float | None) -> FlashANSR:  # type: ignore[no-untyped-def]
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    cfg["optional_condition"] = optional_condition
    torch.manual_seed(0)
    model = FlashANSRModel.from_config(cfg)  # random weights: the memory routing is what is under test
    model.eval()
    return FlashANSR(simplipy_engine=engine, flash_ansr_model=model, tokenizer=tokenizer,
                     # the test vocabulary has no emission flags
                     generation_config=SoftmaxSamplingConfig(draws=4, emission='constants', guidance_weight=guidance_weight),
                     refine={'n_restarts': 1}, compute={'workers': 0}, model_directory=None)


def _problem(n_points: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(n_points, 2))
    return X, (X[:, 0] + 0.5 * X[:, 1]).reshape(-1, 1)


def test_unconditioned_fit_hands_the_sampler_one_memory_row(engine, tokenizer) -> None:  # type: ignore[no-untyped-def]
    nsr = _estimator(engine, tokenizer, optional_condition=True, guidance_weight=0.0)
    seen: list[tuple[int, ...]] = []
    original = nsr._sample

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(tuple(kwargs["memory"].shape))
        return original(*args, **kwargs)

    nsr._sample = spy  # type: ignore[method-assign]
    X, y = _problem(100)  # more points than draws: the shape the old expand produced
    result = nsr.fit(X, y)
    assert seen and seen[0][0] == 1, f"null_memory must reach the sampler as a batch of one, got {seen}"
    assert seen[0][1:] == tuple(nsr.flash_ansr_model.null_memory.shape[1:])
    assert result.ledger is not None  # the call completes; with random weights the pool may be all-invalid


def test_conditioned_fit_is_untouched(engine, tokenizer) -> None:  # type: ignore[no-untyped-def]
    nsr = _estimator(engine, tokenizer, optional_condition=True, guidance_weight=None)
    X, y = _problem(24)
    assert nsr.fit(X, y).ledger is not None


def test_guidance_refused_at_load_without_null_memory(engine, tokenizer) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(CapabilityUnavailable, match="optional_condition"):
        _estimator(engine, tokenizer, optional_condition=False, guidance_weight=0.0)
