"""`emission=` is the public form of the monkeypatch every published T16 number went through.

The 2026-08-26 capability probes set `<mask_all>` by replacing `_prepare_prompt_prefix` on the
instance, because no public verb could. These tests pin the flag's POSITION (training put it
immediately after <bos>), its numeric-channel fill, and the refusal on a checkpoint that lacks it.
"""
import math

import pytest

from flash_ansr.preprocessing import (
    EMISSION_FLAGS,
    CapabilityUnavailable,
    PromptPrefix,
    apply_emission_flag,
)


class _Tok:
    """Minimal tokenizer stand-in: a name->id map with `in` and `[]`."""

    def __init__(self, mapping: dict[str, int]) -> None:
        self._m = mapping

    def __contains__(self, key: str) -> bool:
        return key in self._m

    def __getitem__(self, key: str) -> int:
        return self._m[key]


V24 = _Tok({"<bos>": 1, "<mask_all>": 14, "<mask_fittable>": 15})
#: A vocabulary that predates the flags -- the tokens simply are not there.
NO_FLAGS = _Tok({"<bos>": 1})


def _prefix() -> PromptPrefix:
    return PromptPrefix(tokens=[1, 9, 9], numeric=[float("nan"), 2.0, 3.0],
                        mask=[True, False, False], metadata={"k": []})


class TestEmissionFlag:
    @pytest.mark.parametrize("mode,token", [("skeleton", 14), ("fittable", 15)])
    def test_flag_lands_immediately_after_bos(self, mode: str, token: int) -> None:
        out = apply_emission_flag(_prefix(), mode, V24)
        assert out.tokens == [1, token, 9, 9]
        assert out.tokens[0] == 1, "the flag must follow <bos>, never precede it"

    def test_numeric_channel_is_nan_filled_at_the_flag(self) -> None:
        out = apply_emission_flag(_prefix(), "skeleton", V24)
        assert math.isnan(out.numeric[1])
        assert out.numeric[2:] == [2.0, 3.0]
        assert len(out.numeric) == len(out.tokens)

    def test_mask_grows_with_the_prefix(self) -> None:
        out = apply_emission_flag(_prefix(), "fittable", V24)
        assert len(out.mask) == len(out.tokens)
        assert out.mask[0] is True

    def test_constants_is_the_unflagged_identity(self) -> None:
        prefix = _prefix()
        assert apply_emission_flag(prefix, "constants", V24) is prefix

    def test_metadata_survives(self) -> None:
        assert apply_emission_flag(_prefix(), "skeleton", V24).metadata == {"k": []}

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="emission must be one of"):
            apply_emission_flag(_prefix(), "mask_everything", V24)

    def test_missing_flag_token_refuses_at_call_time(self) -> None:
        # Principle 4: raise BEFORE the encoder runs, not mid-decode.
        with pytest.raises(CapabilityUnavailable, match="<mask_all>"):
            apply_emission_flag(_prefix(), "skeleton", NO_FLAGS)

    def test_missing_flag_token_still_serves_the_default(self) -> None:
        prefix = _prefix()
        assert apply_emission_flag(prefix, "constants", NO_FLAGS) is prefix

    def test_every_declared_mode_is_reachable(self) -> None:
        for mode in EMISSION_FLAGS:
            apply_emission_flag(_prefix(), mode, V24)
