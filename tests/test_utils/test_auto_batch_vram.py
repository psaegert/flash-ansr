"""`batch_size='auto'` must size against the device it is actually running on.

``suggest_batch_size`` applies its measured caps only at ``>= _FULL_CAP_MIN_VRAM_GB`` and falls
back to ``_SMALL_CARD_BATCH_CAP`` below that. The resolver used to default to 24.0 GB on every
non-CUDA device, clearing that gate for hardware it had never measured.
"""
import torch

from flash_ansr.utils.generation import (
    _FULL_CAP_MIN_VRAM_GB,
    _SMALL_CARD_BATCH_CAP,
    suggest_batch_size,
)

_ONE_B_PARAMS = 955_000_000


def test_small_device_gets_the_conservative_cap() -> None:
    """An 8 GB device must not receive the caps measured on a 24 GiB card."""
    small = suggest_batch_size(4096, _ONE_B_PARAMS, vram_gb=8.0)
    full = suggest_batch_size(4096, _ONE_B_PARAMS, vram_gb=_FULL_CAP_MIN_VRAM_GB)
    assert small <= _SMALL_CARD_BATCH_CAP
    assert small < full, "the small-card gate did not bite"


def test_unqueryable_device_gets_the_safe_cap() -> None:
    """A device whose memory cannot be read must not earn the full caps."""
    assert suggest_batch_size(4096, _ONE_B_PARAMS, vram_gb=0.0) <= _SMALL_CARD_BATCH_CAP


def test_optimistic_default_would_clear_the_gate() -> None:
    """Pins why the old unconditional 24.0 was the root cause, not just its MPS symptom."""
    assert suggest_batch_size(4096, _ONE_B_PARAMS, vram_gb=24.0) > _SMALL_CARD_BATCH_CAP


def test_mps_reports_a_usable_memory_budget() -> None:
    """The resolver reads torch.mps.recommended_max_memory(); check the API it depends on."""
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available")
    gb = torch.mps.recommended_max_memory() / 1e9
    assert gb > 0, "recommended_max_memory() gave nothing to size against"
