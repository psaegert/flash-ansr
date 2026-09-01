"""The two divergence mitigations from the T4 byte-row runaway (owner ruling 2026-09-01).

z-loss (`_z_loss`, train.yaml `z_loss_weight`): the restoring force on the loss-flat
shared-logit-offset direction that cross-entropy cannot see. Weight 0.0 must be the
pre-ruling trainer bit-for-bit: the term is only ever added when the weight is positive.

head_pre_logits_norm (model.yaml): the v26 structural fix -- a LayerNorm between the head
MLP and the logit projection, bounding the head-internal activations whose growth
(||h|| 22 -> 207 over 500k steps at byte positions) drove the runaway. Default False so
every existing checkpoint keeps its parameter names and loads unchanged; v26 runs the norm
and NO z-loss.
"""
import math

import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.train.train import _z_loss
from flash_ansr.utils.config_io import load_config


class TestZLoss:
    def test_uniform_logits_give_squared_log_vocab(self) -> None:
        logits = torch.zeros(7, 337)
        valid = torch.ones(7, dtype=torch.bool)
        expected = math.log(337.0) ** 2
        assert torch.isclose(_z_loss(logits, valid), torch.tensor(expected), rtol=1e-6)

    def test_shared_offset_is_penalized(self) -> None:
        # The exact pathology: a shared offset leaves CE unchanged but must move the z-loss.
        logits = torch.randn(11, 64)
        valid = torch.ones(11, dtype=torch.bool)
        shifted = logits - 100.0
        ce = torch.nn.functional.cross_entropy(logits, torch.zeros(11, dtype=torch.long))
        ce_shifted = torch.nn.functional.cross_entropy(shifted, torch.zeros(11, dtype=torch.long))
        assert torch.isclose(ce, ce_shifted, rtol=1e-5)
        assert _z_loss(shifted, valid) > _z_loss(logits, valid) + 1e3

    def test_only_supervised_positions_count(self) -> None:
        logits = torch.zeros(4, 16)
        logits[2] += 50.0  # a masked position must not contribute
        valid = torch.tensor([True, True, False, True])
        expected = math.log(16.0) ** 2
        assert torch.isclose(_z_loss(logits, valid), torch.tensor(expected), rtol=1e-6)

    def test_fp32_result_from_bf16_logits(self) -> None:
        # The offsets this loss exists to shrink sit exactly where bf16 resolution dies.
        logits = (torch.randn(5, 32) - 90.0).to(torch.bfloat16)
        valid = torch.ones(5, dtype=torch.bool)
        out = _z_loss(logits, valid)
        assert out.dtype == torch.float32
        assert torch.isfinite(out)


class TestHeadPreLogitsNorm:
    @pytest.fixture(scope="class")
    def model_kwargs(self):  # type: ignore[no-untyped-def]
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        return {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}

    @pytest.fixture(scope="class")
    def tokenizer(self) -> Tokenizer:
        return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))

    @pytest.fixture(scope="class")
    def engine(self):  # type: ignore[no-untyped-def]
        from simplipy import SimpliPyEngine
        return SimpliPyEngine.load("base", install=True)

    def test_default_head_is_unchanged(self, model_kwargs, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        from flash_ansr.model.flash_ansr_model import FlashANSRModel
        model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **model_kwargs)
        head = model.next_token_head
        assert len(head) == 4
        assert not any(isinstance(m, torch.nn.LayerNorm) for m in head)
        # Checkpoint compatibility: the logit projection keeps its historical name.
        assert "next_token_head.3.weight" in dict(model.named_parameters())

    def test_norm_sits_before_the_logit_projection(self, model_kwargs, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        from flash_ansr.model.flash_ansr_model import FlashANSRModel
        model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer,
                               head_pre_logits_norm=True, **model_kwargs)
        head = model.next_token_head
        assert len(head) == 5
        assert isinstance(head[-2], torch.nn.LayerNorm)
        assert isinstance(head[-1], torch.nn.Linear)
        d = head[0].in_features
        h = torch.randn(2, 9, d)
        out = head(h)
        assert out.shape == (2, 9, len(tokenizer))
