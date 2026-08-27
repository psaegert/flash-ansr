"""The v24 decode switches on the public generation config.

`constrain_ieee754` (the T6/T7 grammar mask) existed only as a private `sample_top_kp`
keyword; a caller going through `FlashANSR` could not reach it. These pin it to the
config surface, where `to_kwargs()` is what the estimator actually forwards.

Beam search and MCTS were retired (2026-08-27), so `SoftmaxSamplingConfig` is the whole
surface -- including `compact_ieee754`, which moved onto the sampling loop with it.
See tests/test_sampling_compaction.py for the compaction behaviour itself.
"""
import pytest

from flash_ansr.utils.generation import SoftmaxSamplingConfig, create_generation_config


def test_constrain_defaults_off() -> None:
    kwargs = SoftmaxSamplingConfig().to_kwargs()
    assert kwargs['constrain_ieee754'] is False


def test_constrain_reaches_to_kwargs() -> None:
    kwargs = SoftmaxSamplingConfig(constrain_ieee754=True).to_kwargs()
    assert kwargs['constrain_ieee754'] is True


def test_the_factory_serves_softmax_sampling_and_names_the_retired_methods() -> None:
    assert isinstance(create_generation_config(), SoftmaxSamplingConfig)
    assert isinstance(create_generation_config(method='softmax_sampling'), SoftmaxSamplingConfig)
    for retired in ('beam_search', 'mcts'):
        with pytest.raises(ValueError, match='retired'):
            create_generation_config(method=retired)  # type: ignore[arg-type]
