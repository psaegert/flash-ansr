"""The v24 decoding options must be reachable from the PUBLIC config (owner ruling
2026-08-26: benchmarks run the code and API we would serve a user).

`constrain_ieee754` (T6/T7 grammar mask) and `compact_ieee754` (T9/T10 per-beam KV
compaction) existed only as private `FlashANSRModel.beam_search` / `sample_top_kp`
keyword arguments: `FlashANSR` never passed them, so no user -- and no benchmark --
could reach the production compaction loop without hand-rolling a decoder. These
tests pin the plumbing and the fail-early validation.
"""
import pytest

from flash_ansr.utils.generation import BeamSearchConfig, SoftmaxSamplingConfig


def test_beam_config_defaults_are_off() -> None:
    """Existing configs must be byte-identical in behaviour: both flags default False."""
    kwargs = BeamSearchConfig().to_kwargs()
    assert kwargs['constrain_ieee754'] is False
    assert kwargs['compact_ieee754'] is False


def test_beam_config_forwards_both_flags() -> None:
    kwargs = BeamSearchConfig(constrain_ieee754=True, compact_ieee754=True).to_kwargs()
    assert kwargs['constrain_ieee754'] is True
    assert kwargs['compact_ieee754'] is True
    assert kwargs['use_cache'] is True


def test_compaction_without_grammar_raises_at_config_time() -> None:
    with pytest.raises(ValueError, match="requires constrain_ieee754"):
        BeamSearchConfig(compact_ieee754=True)


def test_compaction_without_cache_raises_at_config_time() -> None:
    with pytest.raises(ValueError, match="requires use_cache"):
        BeamSearchConfig(constrain_ieee754=True, compact_ieee754=True, use_cache=False)


def test_softmax_config_carries_grammar_only() -> None:
    """Sampling can carry the grammar; compaction is beam-search-only, so the
    sampling config must not advertise a flag it cannot honour."""
    kwargs = SoftmaxSamplingConfig(constrain_ieee754=True).to_kwargs()
    assert kwargs['constrain_ieee754'] is True
    assert 'compact_ieee754' not in kwargs
    with pytest.raises(TypeError):
        SoftmaxSamplingConfig(compact_ieee754=True)  # type: ignore[call-arg]


def test_softmax_config_default_off() -> None:
    assert SoftmaxSamplingConfig().to_kwargs()['constrain_ieee754'] is False


@pytest.mark.parametrize("cfg", [
    BeamSearchConfig(constrain_ieee754=True),
    BeamSearchConfig(constrain_ieee754=True, compact_ieee754=True),
    SoftmaxSamplingConfig(constrain_ieee754=True),
])
def test_mapping_protocol_survives_the_new_fields(cfg) -> None:  # type: ignore[no-untyped-def]
    assert cfg['constrain_ieee754'] is True
    assert dict(cfg) == cfg.to_kwargs()
    assert cfg == cfg.__class__(**{k: v for k, v in cfg.to_kwargs().items()
                                   if k != 'method'})
