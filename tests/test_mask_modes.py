"""The promptable-mask feature (owner ruling 2026-08-24).

Three target formats behind harness-owned flags: ``<mask_all>`` (every numeric
literal masked, simplipy ``mask_all``), ``<mask_fittable>`` (fittable values
masked, structural literals kept, simplipy ``mask_fittable``), and no flag
(unmasked, the default mass). The flag is force-fed and never supervised; the
masked body is the model's own output under the requested policy.
"""
import math

import numpy as np
import pytest
import torch

from test_task_blocks import _iterate, _rows, _source, engine, tokenizer  # noqa: F401

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.data.serialization import (
    CONSTANT_REPRESENTATION_IEEE754_MIXED,
    MASK_ALL_TOKEN,
    MASK_FITTABLE_TOKEN,
    serialize_constant_tokens,
)
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import IEEE754_START_TOKEN

MASK_ALL_ONLY = {"p_mask_all": 1.0, "p_mask_fittable": 0.0}
MASK_FITTABLE_ONLY = {"p_mask_all": 0.0, "p_mask_fittable": 1.0}


def _expression_span(tokens: "list[str]") -> "list[str]":
    return tokens[tokens.index("<expression>") + 1:tokens.index("</expression>")]


class TestMaskedTargets:
    def test_mask_all_bodies_carry_placeholders_and_no_spans(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        for batch in _iterate(tokenizer, mask_block=MASK_ALL_ONLY):
            for row, tokens in _rows(batch, tokenizer):
                assert tokens[1] == MASK_ALL_TOKEN, "the flag leads the prompt"
                assert batch["mask_mode"][row] == "all"
                body = _expression_span(tokens)
                assert IEEE754_START_TOKEN not in body and "<float>" not in body, \
                    "mask_all leaves no serialized values in the body"
                assert "<constant>" in body or not any(
                    t == "<constant>" for t in body), "placeholders spell the constants"
                mask = batch["task_mask"][row].tolist()
                assert mask[1], "the flag itself is NEVER supervised (harness-only)"
                start = tokens.index("<expression>")
                end = tokens.index("</expression>")
                assert not any(mask[start:end + 1]), "the masked body carries loss"

    def test_the_flag_is_absent_from_unmasked_instances(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        cfg = {"p_mask_all": 0.0, "p_mask_fittable": 0.0}
        for batch in _iterate(tokenizer, mask_block=cfg):
            assert all(mode is None for mode in batch["mask_mode"])
            for _, tokens in _rows(batch, tokenizer):
                assert MASK_ALL_TOKEN not in tokens and MASK_FITTABLE_TOKEN not in tokens

    def test_t0_no_config_no_key(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        for batch in _iterate(tokenizer):
            assert "mask_mode" not in batch


class TestFittableKeepsStructure:
    def test_fittable_keeps_structural_literals_serialized(self, engine) -> None:  # type: ignore[no-untyped-def]
        # The worker's exact alignment recipe on a pow-carrying expression: the
        # exponent survives as a SERIALIZED value, the coefficient becomes the
        # placeholder.
        from flash_ansr.utils.skeleton import mask_literals_positional, mask_promptable

        concrete = engine.simplify(["+", "*", "2.5", "x1", "pow", "x2", "3"], mode="corpus")
        masked = mask_promptable(engine, list(concrete), "fittable")
        assert "3" in masked and "<constant>" in masked
        skeleton_m, kept = mask_literals_positional(engine, masked, keep_specials=True)
        constants_opt: "list[float | None]" = []
        vi = 0
        for original, slot in zip(masked, skeleton_m):
            if slot == "<constant>":
                if original == "<constant>":
                    constants_opt.append(None)
                else:
                    constants_opt.append(float(kept[vi]))
                    vi += 1
        assert vi == len(kept) == 1
        out, numeric = serialize_constant_tokens(
            skeleton_m, constants_opt,
            representation=CONSTANT_REPRESENTATION_IEEE754_MIXED,
            rng=np.random.default_rng(0))
        assert "<constant>" in out, "the placeholder survives serialization"
        assert IEEE754_START_TOKEN in out or "<float>" in out, \
            "the kept structural literal rides the ieee754 spelling"

    def test_serializer_none_entries_consume_no_draws(self) -> None:
        tokens = ["+", "<constant>", "<constant>"]
        out_a, num_a = serialize_constant_tokens(
            tokens, [None, 2.5], representation=CONSTANT_REPRESENTATION_IEEE754_MIXED,
            rng=np.random.default_rng(7))
        out_b, num_b = serialize_constant_tokens(
            ["+", "<constant>"], [2.5], representation=CONSTANT_REPRESENTATION_IEEE754_MIXED,
            rng=np.random.default_rng(7))
        assert out_a[1] == "<constant>" and math.isnan(num_a[1])
        # the None slot consumed no coin flip: the 2.5 serialization is identical
        assert out_a[2:] == out_b[1:]


class TestConfigValidation:
    @pytest.mark.parametrize("bad", [
        {"p_mask_all": 0.05},                                      # missing key
        {"p_mask_all": 0.05, "p_mask_fittable": 0.05, "x": 1},     # extra key
        {"p_mask_all": 0.7, "p_mask_fittable": 0.7},               # mass > 1
    ])
    def test_malformed_blocks_refuse(self, tokenizer, bad) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                             constant_representation="ieee754_mixed",
                             target_dialect="tagged", mask_block=bad)


class TestSplitsAndAnchor:
    def test_mask_splits_partition_and_anchor_excludes_masked(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        from flash_ansr.train.train import _ce_split_metrics

        for batch in _iterate(tokenizer, steps=1, batch_size=16, mask_block=MASK_ALL_ONLY):
            batch["labels"] = batch["input_ids"].clone()[..., 1:]
            torch.manual_seed(0)
            logits = torch.randn(batch["input_ids"].shape[0], batch["input_ids"].shape[1],
                                 len(tokenizer))
            parts = _ce_split_metrics(batch, logits, ignore_index=tokenizer["<pad>"])
            assert "expression/mask_all" in parts
            assert "expression/anchor" not in parts, \
                "every instance is masked, so nothing is base-shaped"
            assert "expression/unmasked" not in parts


class TestPromptSurface:
    def test_mask_prefix_composes_and_stands_alone(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
        from flash_ansr.model.flash_ansr_model import FlashANSRModel
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
        model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)

        tokens, numeric = model.complexity_prefix(mask="all")
        assert [tokenizer.vocab[i] for i in tokens] == ["<bos>", MASK_ALL_TOKEN]
        assert all(np.isnan(v) for v in numeric)

        tokens, numeric = model.complexity_prefix(76000, mask="fittable")
        names = [tokenizer.vocab[i] for i in tokens]
        assert names == ["<bos>", MASK_FITTABLE_TOKEN, "<complexity>", "<float>", "</complexity>"]
        assert numeric[3] == 76000.0

        with pytest.raises(ValueError, match="mask"):
            model.complexity_prefix(mask="everything")
        with pytest.raises(ValueError):
            model.complexity_prefix()
