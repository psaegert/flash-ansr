"""``head_fp32``: the next-token head runs in float32 under mixed precision (default on), so the
logits the loss and the sampler see are not bf16-rounded; ``head_fp32: false`` restores the ambient
autocast dtype. Compute-only: no parameters change, every checkpoint loads under either setting."""
import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.numeric import NUMERIC_DTYPE


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("base", install=True)


def _model(tokenizer, engine, **overrides):  # type: ignore[no-untyped-def]
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    kwargs.update(overrides)
    torch.manual_seed(3)
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    model.eval()
    return model, kwargs


def _batch(tokenizer, kwargs):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    B, M = 2, 12
    data = torch.randn(B, M, kwargs["encoder_max_n_variables"], dtype=NUMERIC_DTYPE)
    mask = torch.ones(B, M, dtype=torch.bool)
    mask[0, 9:] = False
    tokens = torch.randint(0, len(tokenizer), (B, 5))
    return tokens, data, mask


def test_default_is_on_and_config_can_turn_it_off(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    model, _ = _model(tokenizer, engine)
    assert model.head_fp32 is True
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    cfg["head_fp32"] = False
    off = FlashANSRModel.from_config(cfg)
    assert off.head_fp32 is False
    cfg.pop("head_fp32")
    assert FlashANSRModel.from_config(cfg).head_fp32 is True


def test_logits_are_float32_under_bf16_autocast(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    model, kwargs = _model(tokenizer, engine, head_fp32=True)
    tokens, data, mask = _batch(tokenizer, kwargs)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(tokens, data, data_attn_mask=mask)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

    model.head_fp32 = False
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits_ambient = model(tokens, data, data_attn_mask=mask)
    assert logits_ambient.dtype == torch.bfloat16


def test_flag_is_compute_only(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    """Without autocast the two settings are the same computation, and the state dict is identical."""
    model, kwargs = _model(tokenizer, engine, head_fp32=True)
    tokens, data, mask = _batch(tokenizer, kwargs)
    with torch.no_grad():
        on = model(tokens, data, data_attn_mask=mask)
        model.head_fp32 = False
        off = model(tokens, data, data_attn_mask=mask)
    torch.testing.assert_close(on, off)
    other, _ = _model(tokenizer, engine, head_fp32=False)
    assert set(other.state_dict()) == set(model.state_dict())


def test_head_follows_its_own_weight_dtype(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    """A model cast to bf16 keeps a bf16 head: the flag never forces a dtype the weights do not have."""
    model, kwargs = _model(tokenizer, engine, head_fp32=True)
    tokens, data, mask = _batch(tokenizer, kwargs)
    head = model.next_token_head.to(torch.bfloat16)
    with torch.no_grad():
        logits = model._logits(torch.randn(2, 5, kwargs["decoder_model_dim"]))
    assert logits.dtype == torch.bfloat16
    assert head is model.next_token_head
