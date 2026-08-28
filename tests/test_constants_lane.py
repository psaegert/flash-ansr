"""The constants-format check at tokenizer construction.

Byte tokens are two hex digits, so a vocabulary built for a different constants format
shares NO content token with this one. Without an explicit check that mismatch surfaces far
downstream, one out-of-vocabulary token at a time. Two mechanisms, because a configuration
may or may not declare its format: a declaration is compared directly, and a configuration
that declares nothing is recognised by its alphabet.
"""
import copy
import re

import pytest

from flash_ansr import get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import BYTE_TOKENS, CONSTANTS_FORMAT, IEEE754_START_TOKEN

#: Configurations this build serves, and configurations declaring a different format.
SERVED = ("test", "v24-template", "v25.0-T1")
FOREIGN = ("v24.0-T13", "v24.0-T14", "v24.0-T14-base",
           "v24.0-T15", "v24.0-T15-base", "v24.0-T16")


@pytest.mark.parametrize("name", SERVED)
def test_served_configs_carry_this_codecs_alphabet(name: str) -> None:
    config = load_config(get_path("configs", name, "tokenizer.yaml"))
    assert config["constants_format"] == CONSTANTS_FORMAT
    tokenizer = Tokenizer.from_config(config)
    assert all(token in tokenizer for token in BYTE_TOKENS)
    assert "<h0>" not in tokenizer and "<b0>" not in tokenizer


@pytest.mark.parametrize("name", FOREIGN)
def test_foreign_format_configs_are_refused_by_name(name: str) -> None:
    """A configuration that declares a different constants format must be refused saying so,
    not half-work."""
    with pytest.raises(ValueError, match="constants_format"):
        Tokenizer.from_config(get_path("configs", name, "tokenizer.yaml"))


def test_an_undeclared_foreign_vocabulary_is_caught_by_its_alphabet() -> None:
    """A vocabulary stored alongside a checkpoint may declare no format, so the alphabet has
    to give it away. Matched on the token it is missing rather than on the wording: the
    contract is that the message NAMES the gap, not that it phrases it a particular way."""
    config = load_config(get_path("configs", "v24.0-T16", "tokenizer.yaml"))
    del config["constants_format"]
    assert IEEE754_START_TOKEN in config["special_tokens"]
    with pytest.raises(ValueError, match=re.escape(BYTE_TOKENS[0])):
        Tokenizer.from_config(config)


def test_a_vocabulary_without_spans_at_all_is_left_alone() -> None:
    """The check is about which alphabet serializes constants, not about requiring one. A
    vocabulary that never opens a span has no lane to be wrong about."""
    config = copy.deepcopy(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))
    del config["constants_format"]
    config["special_tokens"] = [t for t in config["special_tokens"]
                                if t != IEEE754_START_TOKEN and t not in BYTE_TOKENS]
    Tokenizer.from_config(config)
