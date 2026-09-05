"""head_pre_logits_norm (model.yaml): a LayerNorm between the head MLP and the logit projection,
bounding the head-internal activations whose loss-flat growth drove the byte-row logit runaway.
Default False so every existing checkpoint keeps its parameter names and loads unchanged.
"""
import pytest
import torch

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config


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
