"""The per-point residual head: IEEE nibble target, per-instance loss reduction, rulers."""
import numpy as np
import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.train.train import (
    _RESIDUAL_SCALES, _float32_to_nibbles_torch, _masked_median)
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import (
    IEEE754_N_NIBBLES, IEEE754_N_NIBBLE_SYMBOLS, NIBBLE_TOKENS,
    float32_to_nibble_tokens, float32_to_nibble_values, nibble_values_to_float32)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("base", install=True)


def _model(tokenizer, engine, **overrides):  # type: ignore[no-untyped-def]
    from flash_ansr.model.flash_ansr_model import FlashANSRModel
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    kwargs.update(overrides)
    torch.manual_seed(7)
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model, kwargs


# --- the codec ---------------------------------------------------------------------------

def test_vectorized_codec_round_trips_bit_exactly() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.normal(0, 1e8, 4000), rng.normal(0, 1e-8, 4000),
        np.array([0.0, -0.0, 1.0, -2.0, np.inf, -np.inf, np.nan]),
    ]).astype(np.float32)
    nibbles = float32_to_nibble_values(values)
    assert nibbles.shape == (values.size, IEEE754_N_NIBBLES) and nibbles.dtype == np.uint8
    assert nibbles.max() < IEEE754_N_NIBBLE_SYMBOLS
    # bit patterns, not float equality: -0.0 and nan must survive as themselves
    restored = nibble_values_to_float32(nibbles)
    assert np.array_equal(values.view(np.uint32), restored.view(np.uint32))


def test_vectorized_codec_agrees_with_the_scalar_token_encoder() -> None:
    rng = np.random.default_rng(1)
    for value in rng.normal(0, 100, 500).astype(np.float32):
        expected = float32_to_nibble_tokens(float(value))
        got = [NIBBLE_TOKENS[n] for n in float32_to_nibble_values(np.float32(value))]
        assert got == expected


def test_torch_codec_is_bit_identical_to_numpy() -> None:
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(0, 1e6, 3000),
                             np.array([0.0, -0.0, 1.0, -2.0])]).astype(np.float32)
    numpy_nibbles = float32_to_nibble_values(values).astype(np.int64)
    torch_nibbles = _float32_to_nibbles_torch(torch.from_numpy(values)).numpy()
    assert np.array_equal(numpy_nibbles, torch_nibbles)


def test_nibble_target_encodes_sign_and_zero_natively() -> None:
    """The reason the IEEE parameterization needs no zero class and no sign logit."""
    nibbles = _float32_to_nibbles_torch(torch.tensor([0.0, -1.5, 1.5]))
    assert torch.all(nibbles[0] == 0), "+0.0 is all-zero nibbles -- a clean point is representable"
    assert nibbles[1][0] >= 8 and nibbles[2][0] < 8, "sign is the top bit of nibble 0"
    assert torch.equal(nibbles[1][1:], nibbles[2][1:]), "+/-x differ ONLY in that bit"


# --- the ruler ---------------------------------------------------------------------------

def test_masked_median_matches_numpy_on_valid_entries_only() -> None:
    torch.manual_seed(0)
    y = torch.randn(32, 25)
    valid = torch.rand(32, 25) > 0.4
    valid[3] = False  # a row with nothing valid must not blow up
    got = _masked_median(y, valid).numpy()
    for i in range(32):
        want = np.median(y[i][valid[i]].numpy()) if valid[i].any() else 0.0
        assert np.isclose(got[i], want, atol=1e-6)


def test_residual_scale_rulers_are_declared_and_validated() -> None:
    assert _RESIDUAL_SCALES == ("none", "mad", "y_plus_mad")


# --- the head ----------------------------------------------------------------------------

def test_head_shape_and_same_pass_capture(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    model, kwargs = _model(tokenizer, engine, residual_head=True, outlier_head=False)
    assert model.residual_head is not None
    B, M = 3, 11
    data = torch.randn(B, M, kwargs["encoder_max_n_variables"])
    mask = torch.ones(B, M, dtype=torch.bool)
    mask[0, 8:] = False
    with torch.no_grad():
        model(torch.randint(0, len(tokenizer), (B, 5)), data, data_attn_mask=mask)
    points = model.point_representations
    assert points is not None and points.shape == (B, M, kwargs["encoder_dim"])
    logits = model.residual_head(points)
    assert logits.shape == (B, M, IEEE754_N_NIBBLES * IEEE754_N_NIBBLE_SYMBOLS)
    assert torch.all(points[0, 8:] == 0)


def test_residual_head_off_by_default(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    model, _ = _model(tokenizer, engine)
    assert model.residual_head is None


def test_point_representations_captured_for_either_head(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    """Residual head alone must still trigger the pre-pooling capture."""
    model, kwargs = _model(tokenizer, engine, outlier_head=False, residual_head=True)
    data = torch.randn(2, 6, kwargs["encoder_max_n_variables"])
    with torch.no_grad():
        model(torch.randint(0, len(tokenizer), (2, 4)), data)
    assert model.point_representations is not None


# --- the loss ----------------------------------------------------------------------------

def _loss(logits, labels, instance_index):  # type: ignore[no-untyped-def]
    from flash_ansr.train.train import Trainer
    return Trainer._residual_loss(None, (logits, labels, instance_index))  # type: ignore[arg-type]


def test_loss_reduces_per_instance_not_flat_across_the_batch() -> None:
    """A 512-point problem must not out-vote an 8-point one. Constructed so the two differ."""
    torch.manual_seed(0)
    big, small = 200, 4
    labels = torch.zeros(big + small, IEEE754_N_NIBBLES, dtype=torch.long)
    logits = torch.zeros(big + small, IEEE754_N_NIBBLES, IEEE754_N_NIBBLE_SYMBOLS)
    logits[:big, :, 0] = 10.0          # the big instance is fit perfectly
    logits[big:, :, 5] = 10.0          # the small one is confidently wrong
    index = torch.cat([torch.zeros(big, dtype=torch.long), torch.ones(small, dtype=torch.long)])

    per_instance = float(_loss(logits, labels, index))
    flat = float(nn_functional_cross_entropy(logits, labels))
    # flat pooling buries the small instance; per-instance weights it at 1/2
    assert per_instance > flat * 5, (per_instance, flat)
    assert 4.5 < per_instance < 5.5, "half of ~0 plus half of ~10"


def nn_functional_cross_entropy(logits, labels):  # type: ignore[no-untyped-def]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, IEEE754_N_NIBBLE_SYMBOLS), labels.reshape(-1))


def test_loss_is_ln16_at_uniform_logits() -> None:
    labels = torch.randint(0, 16, (64, IEEE754_N_NIBBLES))
    logits = torch.zeros(64, IEEE754_N_NIBBLES, IEEE754_N_NIBBLE_SYMBOLS)
    index = torch.arange(64) // 8
    assert abs(float(_loss(logits, labels, index)) - float(np.log(16))) < 1e-5


def test_empty_scoring_is_a_finite_zero() -> None:
    empty = (torch.zeros(0, IEEE754_N_NIBBLES, IEEE754_N_NIBBLE_SYMBOLS),
             torch.zeros(0, IEEE754_N_NIBBLES, dtype=torch.long),
             torch.zeros(0, dtype=torch.long))
    assert float(_loss(*empty)) == 0.0


# --- the public verb ---------------------------------------------------------------------

class _Estimator:
    """Minimal stand-in: predict_residuals needs the model and the X-variable count.

    n_variables counts the X columns only; encoder_max_n_variables includes the y column.
    """

    def __init__(self, model, n_variables: int) -> None:  # type: ignore[no-untyped-def]
        self.flash_ansr_model = model
        self.n_variables = n_variables


def test_predict_residuals_refuses_without_a_head(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.preprocessing import CapabilityUnavailable
    from flash_ansr.tasks import predict_residuals
    model, kwargs = _model(tokenizer, engine)
    estimator = _Estimator(model, kwargs["encoder_max_n_variables"] - 1)
    X, y = np.zeros((5, 2), np.float32), np.zeros(5, np.float32)
    with pytest.raises(CapabilityUnavailable, match="residual_head"):
        predict_residuals(estimator, X, y)


def test_predict_residuals_returns_one_distribution_per_point(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.tasks import ValueDistribution, predict_residuals
    model, kwargs = _model(tokenizer, engine, residual_head=True)
    estimator = _Estimator(model, kwargs["encoder_max_n_variables"] - 1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 2)).astype(np.float32)
    y = rng.normal(size=7).astype(np.float32)

    out = predict_residuals(estimator, X, y, n_samples=16, seed=0)
    assert len(out) == 7 and all(isinstance(d, ValueDistribution) for d in out)
    assert all(d.n == 16 for d in out)
    # a head emits exactly 8 nibble positions structurally -- it cannot go off-grammar
    assert all(d.off_grammar_steps == 0 and d.closed_cleanly_fraction == 1.0 for d in out)


def test_predict_residuals_is_deterministic_under_a_seed(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.tasks import predict_residuals
    model, kwargs = _model(tokenizer, engine, residual_head=True)
    estimator = _Estimator(model, kwargs["encoder_max_n_variables"] - 1)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 2)).astype(np.float32)
    y = rng.normal(size=4).astype(np.float32)
    first = predict_residuals(estimator, X, y, n_samples=8, seed=7)
    again = predict_residuals(estimator, X, y, n_samples=8, seed=7)
    # Compare BIT PATTERNS, not floats: an untrained head emits nan patterns freely and
    # nan != nan would fail this even on identical draws (which is what non_finite_fraction
    # is there to report).
    def bits(runs):  # type: ignore[no-untyped-def]
        return np.array([d.draws for d in runs], dtype=np.float32).view(np.uint32)
    assert np.array_equal(bits(first), bits(again))


def test_estimator_exposes_the_verb() -> None:
    from flash_ansr.flash_ansr import FlashANSR
    assert callable(getattr(FlashANSR, "predict_residuals", None))


def _scaling_trainer(scale: str):  # type: ignore[no-untyped-def]
    from flash_ansr.train.train import Trainer
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.residual_scale = scale
    return trainer


def test_degenerate_problem_falls_back_to_no_scaling() -> None:
    """Every y identical -> MAD 0. Clamping to float32 tiny would make an ordinary residual
    ~1e38: absurd, yet FINITE, so it would encode as a legitimate nibble target."""
    batch = {
        "residual": torch.tensor([[1.0, 2.0, 0.0, 0.0]]),
        "y_tensors": torch.ones(1, 4, 1),
        "data_attn_mask": torch.tensor([[True, True, True, False]]),
    }
    scaled = _scaling_trainer("mad")._scaled_residual(batch)
    torch.testing.assert_close(scaled[0, :3], torch.tensor([1.0, 2.0, 0.0]))


def test_masking_follows_the_scaled_value_not_the_raw_residual() -> None:
    """A genuinely tiny spread can push a finite residual to inf under the ruler."""
    # spread must be tiny yet NON-zero: near 1.0 the float32 grid is ~1e-7, far too coarse.
    y = torch.tensor([[1e-30, 2e-30, 3e-30, 0.0]]).unsqueeze(-1)
    batch = {
        "residual": torch.tensor([[1e20, 1e20, 0.0, 0.0]]),
        "y_tensors": y,
        "data_attn_mask": torch.tensor([[True, True, True, False]]),
    }
    scaled = _scaling_trainer("mad")._scaled_residual(batch)
    assert torch.isfinite(batch["residual"]).all(), "the RAW residual is entirely finite"
    assert not torch.isfinite(scaled[0, :2]).any(), "the premise: the ruler overflows them"
    valid = batch["data_attn_mask"] & torch.isfinite(scaled)
    assert valid.tolist() == [[False, False, True, False]], (
        "overflowed points must drop; masking the RAW residual would have kept them")
