"""The nibble/byte lane split (Q3, owner ruling 2026-08-27).

Byte tokens are two hex digits, so a nibble-lane vocabulary shares NO content token with a
byte-lane one. Without an explicit check that mismatch surfaces far downstream, one
out-of-vocabulary token at a time. Two mechanisms, because there are two kinds of stale
config: those written since the split declare `constants_format`, and those written before
it -- notably the tokenizer.yaml inside a v24 checkpoint directory -- declare nothing and
are recognised by their alphabet.
"""
import copy

import pytest

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import BYTE_TOKENS, CONSTANTS_FORMAT, IEEE754_START_TOKEN

BYTE_LANE = ("test", "v24-template", "v24.0-T17", "v24.0-T18")
NIBBLE_LANE = ("v24.0-T13", "v24.0-T14", "v24.0-T14-base",
               "v24.0-T15", "v24.0-T15-base", "v24.0-T16")


@pytest.mark.parametrize("name", BYTE_LANE)
def test_live_configs_are_byte_lane(name: str) -> None:
    config = load_config(get_path("configs", name, "tokenizer.yaml"))
    assert config["constants_format"] == CONSTANTS_FORMAT
    tokenizer = Tokenizer.from_config(config)
    assert all(token in tokenizer for token in BYTE_TOKENS)
    assert "<h0>" not in tokenizer and "<b0>" not in tokenizer


@pytest.mark.parametrize("name", NIBBLE_LANE)
def test_frozen_configs_are_refused_by_name(name: str) -> None:
    """These runs happened on the retired lane and are frozen at the compat/v24-nibbles tag.
    Loading one under a byte build must fail saying so, not half-work."""
    with pytest.raises(ValueError, match="constants_format"):
        Tokenizer.from_config(get_path("configs", name, "tokenizer.yaml"))


def test_an_undeclared_nibble_vocabulary_is_caught_by_its_alphabet() -> None:
    """The checkpoint case: tokenizer.yaml files written before the key existed declare
    nothing, so the alphabet has to give them away."""
    config = load_config(get_path("configs", "v24.0-T16", "tokenizer.yaml"))
    del config["constants_format"]
    assert IEEE754_START_TOKEN in config["special_tokens"]
    with pytest.raises(ValueError, match="retired"):
        Tokenizer.from_config(config)


def test_a_vocabulary_without_spans_at_all_is_left_alone() -> None:
    """The check is about which alphabet serializes constants, not about requiring one. A
    vocabulary that never opens a span has no lane to be wrong about."""
    config = copy.deepcopy(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))
    del config["constants_format"]
    config["special_tokens"] = [t for t in config["special_tokens"]
                                if t != IEEE754_START_TOKEN and t not in BYTE_TOKENS]
    Tokenizer.from_config(config)
