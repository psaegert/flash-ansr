"""The static decode path's absolute-position bounds check.

Preserved from the retired compaction suite: the guard is not about compaction, it is
about the static path indexing the RoPE table by absolute position at all.
"""
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
def model(tokenizer):  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    torch.manual_seed(0x24C2)
    return FlashANSRModel(simplipy_engine=SimpliPyEngine.load("base", install=True),
                          tokenizer=tokenizer, **kwargs).eval()


def test_static_decode_refuses_a_max_len_past_the_rope_table(model, tokenizer):
    """The dynamic path raises a clean ValueError from RotaryEmbedding.forward. The static
    path indexes the table directly, so without this the same mistake reaches the GPU as an
    out-of-bounds gather -- a device-side assert that poisons the CUDA context for the whole
    process instead of raising something the caller can act on."""
    limit = int(model.decoder_max_seq_len)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.sample_top_kp(
            torch.rand(13, 11, dtype=NUMERIC_DTYPE), choices=2, max_len=limit + 1,
            return_raw=True, initial_tokens=[tokenizer["<bos>"]], static_decode=True)
