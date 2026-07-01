"""FlashANSRModel.from_config validates required keys up front with a clear, actionable error."""
import pytest

from flash_ansr import FlashANSRModel


def test_from_config_missing_keys_raises_clear_keyerror():
    # An incomplete model config must fail with one clear error naming the missing keys, BEFORE the
    # (slow) simplipy-engine load -- not an opaque bare KeyError from deep in the constructor.
    with pytest.raises(KeyError) as excinfo:
        FlashANSRModel.from_config({'model': {'encoder_dim': 256}})

    message = str(excinfo.value)
    assert 'missing required key' in message
    # names specific missing keys and lists what was present
    assert 'simplipy_engine' in message
    assert 'tokenizer' in message
    assert 'Present keys' in message


def test_from_config_missing_keys_reports_only_absent_ones():
    # A key that IS present must not be listed as missing.
    with pytest.raises(KeyError) as excinfo:
        FlashANSRModel.from_config({'encoder_dim': 256, 'decoder_n_layers': 4})

    message = str(excinfo.value)
    missing_list = message.split('missing required key(s): [', 1)[1].split(']', 1)[0]
    assert 'encoder_dim' not in missing_list
    assert 'decoder_n_layers' not in missing_list
    assert 'tokenizer' in missing_list  # a genuinely absent key still reported
