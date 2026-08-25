"""The per-point outlier head: encoder attach point, same-pass capture, and AUROC math."""
import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.encoders.set_transformer import SetTransformer
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.train.train import _binary_auprc, _binary_auroc
from flash_ansr.utils.config_io import load_config


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


def test_set_transformer_returns_prepool_point_representations() -> None:
    torch.manual_seed(0)
    encoder = SetTransformer(input_dim=8, output_dim=None, model_dim=16, n_heads=2,
                             n_isab=2, n_sab=1, n_inducing_points=4, n_seeds=3).eval()
    x = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 6:] = False
    pooled, points = encoder(x, mask, return_point_representations=True)
    assert pooled.shape == (2, 3, 16)
    assert points.shape == (2, 10, 16)
    assert torch.all(points[1, 6:] == 0), "padded rows must stay zeroed at the attach point"
    # The flag must not change the pooled encoding itself.
    torch.testing.assert_close(encoder(x, mask), pooled)


def test_model_captures_point_representations_in_the_same_pass(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    model, kwargs = _model(tokenizer, engine, outlier_head=True)
    B, M = 2, 12
    data = torch.randn(B, M, kwargs["encoder_max_n_variables"])
    mask = torch.ones(B, M, dtype=torch.bool)
    mask[0, 9:] = False
    tokens = torch.randint(0, len(tokenizer), (B, 5))
    with torch.no_grad():
        model(tokens, data, data_attn_mask=mask)
    points = model.point_representations
    assert points is not None and points.shape == (B, M, kwargs["encoder_dim"])
    assert torch.all(points[0, 9:] == 0)
    logits = model.outlier_head(points).squeeze(-1)
    assert logits.shape == (B, M)


def test_disabled_head_stays_absent_and_checkpoint_compatible(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    with_head, kwargs = _model(tokenizer, engine, outlier_head=True)
    without_head, _ = _model(tokenizer, engine)
    assert without_head.outlier_head is None
    extra = set(with_head.state_dict()) - set(without_head.state_dict())
    assert extra and all(key.startswith("outlier_head.") for key in extra)
    B, M = 1, 6
    data = torch.randn(B, M, kwargs["encoder_max_n_variables"])
    with torch.no_grad():
        without_head(torch.randint(0, len(tokenizer), (B, 4)), data)
    assert without_head.point_representations is None


def test_binary_auroc() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert _binary_auroc(scores, torch.tensor([False, False, True, True])) == pytest.approx(1.0)
    assert _binary_auroc(scores, torch.tensor([True, True, False, False])) == pytest.approx(0.0)
    assert _binary_auroc(scores, torch.tensor([False, True, False, True])) == pytest.approx(0.75)


def test_binary_auprc() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    # perfect ranking: precision 1 at every positive
    assert _binary_auprc(scores, torch.tensor([False, False, True, True])) == pytest.approx(1.0)
    # inverted ranking: positives arrive at ranks 3 and 4 -> AP = (1/3 + 2/4) / 2
    assert _binary_auprc(scores, torch.tensor([True, True, False, False])) == pytest.approx((1 / 3 + 2 / 4) / 2)
    # alternating: positives at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
    assert _binary_auprc(scores, torch.tensor([False, True, False, True])) == pytest.approx((1.0 + 2 / 3) / 2)
    # the imbalance property AUROC hides: one positive ranked 11th of 1011 still has
    # AUROC 0.99 but AP 1/11
    scores = torch.cat([torch.zeros(1000), torch.linspace(1, 2, 11)])
    labels = torch.zeros(1011, dtype=torch.bool)
    labels[1000] = True                      # the lowest of the top-11 band
    assert _binary_auroc(scores, labels) > 0.99
    assert _binary_auprc(scores, labels) == pytest.approx(1 / 11)
