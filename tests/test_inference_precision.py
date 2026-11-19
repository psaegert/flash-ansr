"""Inference precision (owner 2026-11-19: "Inference should run in bf16"): ComputeConfig.precision resolves to bf16
only where the device has native bf16, and the static decoder stores K/V in the layers' dtype, not the fp32 head's."""
import contextlib

import pytest
import torch

from flash_ansr import get_path
from flash_ansr.estimator_config import PRECISIONS, ComputeConfig, native_bf16
from flash_ansr.model import flash_ansr_model as fam
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.numeric import NUMERIC_DTYPE


def test_precision_defaults_to_auto_and_validates():
    assert ComputeConfig().precision == 'auto'
    assert PRECISIONS == ('auto', 'bf16', 'fp32')
    with pytest.raises(ValueError, match="precision"):
        ComputeConfig(precision='fp16')
    assert ComputeConfig.from_mapping({'device': 'cpu', 'precision': 'fp32'}).as_dict()['precision'] == 'fp32'


def test_auto_is_fp32_on_cpu_and_bf16_is_refused_there():
    assert native_bf16('cpu') is False
    assert ComputeConfig(device='cpu').resolved_precision == 'fp32'
    assert ComputeConfig(device='cpu', precision='fp32').resolved_precision == 'fp32'
    with pytest.raises(ValueError, match="native bf16"):
        _ = ComputeConfig(device='cpu', precision='bf16').resolved_precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_auto_follows_the_cards_compute_capability():
    major, _ = torch.cuda.get_device_capability(0)
    assert ComputeConfig(device='cuda').resolved_precision == ('bf16' if major >= 8 else 'fp32')


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


@pytest.fixture(scope="module")
def model_fp32_head(tokenizer):  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine

    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    kwargs['head_fp32'] = True   # as every released model: the logits stay fp32 under autocast
    torch.manual_seed(0x24C2)
    return FlashANSRModel(simplipy_engine=SimpliPyEngine.load("base", install=True),
                          tokenizer=tokenizer, **kwargs).eval()


def test_static_cache_takes_the_kv_dtype_not_the_heads(model_fp32_head, tokenizer, monkeypatch):
    """Under bf16 autocast the layers write bf16 K/V while the head returns fp32 logits. The cache used to be
    created in the logits' dtype, storing bf16 values in fp32 buffers (twice the memory); it must follow the K/V."""
    seen: list[torch.dtype] = []
    real = fam.StaticKVCache

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(kwargs['dtype'])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fam, "StaticKVCache", Recording)
    torch.manual_seed(1)
    data = torch.rand(13, 11, dtype=NUMERIC_DTYPE)
    with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        model_fp32_head.sample_top_kp(data, draws=2, max_len=12, return_raw=True,
                                      initial_tokens=[tokenizer["<bos>"]], static_decode=True)
    assert seen, "the static path did not run"
    assert all(d == torch.bfloat16 for d in seen), seen


def test_static_cache_stays_fp32_without_autocast(model_fp32_head, tokenizer, monkeypatch):
    seen: list[torch.dtype] = []
    real = fam.StaticKVCache

    class Recording(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(kwargs['dtype'])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fam, "StaticKVCache", Recording)
    torch.manual_seed(1)
    with torch.no_grad(), contextlib.nullcontext():
        model_fp32_head.sample_top_kp(torch.rand(13, 11, dtype=NUMERIC_DTYPE), draws=2, max_len=12, return_raw=True,
                                      initial_tokens=[tokenizer["<bos>"]], static_decode=True)
    assert seen and all(d == torch.float32 for d in seen), seen
