"""The promptable-mask + constant-infilling feature (owner rulings 2026-08-24).

Three emission formats behind harness-owned flags (``<mask_all>``, ``<mask_fittable>``,
absence = unmasked), the unflagged per-slot PARTIAL circumstance, and the
``<predict_constants>`` block: one ``<ieee754>`` span per ``<masked_constant>``
placeholder, positional order. The flag is an emission-format directive -- under a flag
the placeholder pattern is policy-determined and supervised; in a partial instance it is
a random harness draw and context-only.
"""
import math

import numpy as np
import pytest
import torch

from test_task_blocks import _iterate, _rows, _source, engine, tokenizer  # noqa: F401

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.data.serialization import (
    MASK_ALL_TOKEN,
    MASK_FITTABLE_TOKEN,
    MASKED_CONSTANT_TOKEN,
    PREDICT_CONSTANTS_END_TOKEN,
    PREDICT_CONSTANTS_START_TOKEN,
    serialize_constant_tokens,
)
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import (
    IEEE754_N_NIBBLES,
    IEEE754_START_TOKEN,
    nibble_tokens_to_float32,
)


def _mask_cfg(**overrides):
    cfg = {"p_mask_all": 0.0, "p_mask_fittable": 0.0, "p_partial": 0.0, "p_placeheld": 0.5,
           "p_predict_constants_flagged": 0.0, "p_predict_constants_partial": 0.0}
    cfg.update(overrides)
    return cfg


def _expression_span(tokens: "list[str]") -> "list[str]":
    return tokens[tokens.index("<expression>") + 1:tokens.index("</expression>")]


class TestMaskedTargets:
    def test_mask_all_bodies_carry_placeholders_and_no_spans(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        for batch in _iterate(tokenizer, mask_block=_mask_cfg(p_mask_all=1.0)):
            for row, tokens in _rows(batch, tokenizer):
                assert tokens[1] == MASK_ALL_TOKEN, "single prefix element leads the prompt"
                assert batch["mask_mode"][row] == "all"
                body = _expression_span(tokens)
                assert IEEE754_START_TOKEN not in body and "<float>" not in body, \
                    "mask_all leaves no serialized values in the body"
                assert MASKED_CONSTANT_TOKEN in body
                assert batch["n_placeholders"][row] == body.count(MASKED_CONSTANT_TOKEN)
                mask = batch["task_mask"][row].tolist()
                assert mask[1], "the flag itself is NEVER supervised (harness-only)"
                start = tokens.index("<expression>")
                end = tokens.index("</expression>")
                assert not any(mask[start:end + 1]), \
                    "flagged placeholders are policy-determined: the whole body carries loss"

    def test_the_flag_is_absent_from_unmasked_instances(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        for batch in _iterate(tokenizer, mask_block=_mask_cfg()):
            assert all(mode is None for mode in batch["mask_mode"])
            for _, tokens in _rows(batch, tokenizer):
                assert MASK_ALL_TOKEN not in tokens and MASK_FITTABLE_TOKEN not in tokens
                assert MASKED_CONSTANT_TOKEN not in tokens

    def test_t0_no_config_no_key(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        for batch in _iterate(tokenizer):
            assert "mask_mode" not in batch
            assert "n_placeholders" not in batch
            assert "predict_constants" not in batch


class TestPerSlotPolicies:
    def test_fittable_slots_match_the_simplipy_policy(self, engine) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        from flash_ansr.utils.skeleton import fittable_slots

        # The worker's target dialect (symbolic-data's tagged_canonical) spells the
        # exact rational 2.5 STRUCTURALLY as 5 <div> 2 -- two coefficient sites --
        # while the exponent is one structural site. Slot granularity is the literal
        # site (the serialization's own premise), and the policy decides per site:
        # both coefficient halves fittable, the exponent kept.
        from symbolic_data.token_ops import tagged_canonical
        concrete = tagged_canonical(engine, engine.to_prefix("2.5*x1 + x2^3"))
        slots = fittable_slots(engine, list(concrete))
        assert sorted(slots) == [False, True, True]

    def test_serializer_keeps_none_entries_as_placeholders(self) -> None:
        tokens = ["+", "<constant>", "<constant>"]
        out_a, num_a = serialize_constant_tokens(tokens, [None, 2.5])
        out_b, num_b = serialize_constant_tokens(["+", "<constant>"], [2.5])
        assert out_a[1] == "<constant>" and math.isnan(num_a[1])
        assert out_a[2:] == out_b[1:]
        # The kept value is a span, never a <float>: it is part of the expression.
        assert out_a[2] == "<ieee754>" and "<float>" not in out_a


class TestPredictConstantsBlock:
    def test_block_spans_bind_positionally_and_decode(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        cfg = _mask_cfg(p_mask_all=1.0, p_predict_constants_flagged=1.0)
        for batch in _iterate(tokenizer, mask_block=cfg):
            saw_block = False
            for row, tokens in _rows(batch, tokenizer):
                draw = batch["predict_constants"][row]
                if draw is None:
                    # collection restructured this instance (merge or sign absorption):
                    # the gate withholds the block rather than supervise engine-internal
                    # values. The collected expression is still the target.
                    assert PREDICT_CONSTANTS_START_TOKEN not in tokens
                    continue
                saw_block = True
                k = batch["n_placeholders"][row]
                assert len(draw["values"]) == k >= 1
                start = tokens.index(PREDICT_CONSTANTS_START_TOKEN)
                end = tokens.index(PREDICT_CONSTANTS_END_TOKEN)
                assert end > tokens.index("</expression>"), "the block is a suffix"
                block = tokens[start:end + 1]
                assert block.count(IEEE754_START_TOKEN) == k
                # first span decodes to the first placeheld value, positional binding
                first = block.index(IEEE754_START_TOKEN)
                nibbles = block[first + 1:first + 1 + IEEE754_N_NIBBLES]
                assert nibble_tokens_to_float32(nibbles) == float(np.float32(draw["values"][0]))
                # loss discipline: openers force-fed, nibbles + closers supervised
                mask = batch["task_mask"][row].tolist()
                assert mask[start], "the block opener is harness-owned"
                assert mask[start + 1], "each span opener is harness-owned"
                assert not any(mask[start + 2:start + 2 + IEEE754_N_NIBBLES]), \
                    "the nibbles are the model's"
                assert not mask[end], "the closing tag is the model's"
                seg = batch["task_segments"][row].tolist()
                assert set(seg[start:end + 1]) == {3}, "the block is segment 3"
            assert saw_block, "some instances must be collection-stable"

    def test_block_probability_zero_means_no_block(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        cfg = _mask_cfg(p_mask_all=1.0)
        for batch in _iterate(tokenizer, mask_block=cfg):
            assert all(draw is None for draw in batch["predict_constants"])
            for _, tokens in _rows(batch, tokenizer):
                assert PREDICT_CONSTANTS_START_TOKEN not in tokens


class TestPartialInstances:
    def test_partial_placeholders_are_context_only(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        cfg = _mask_cfg(p_partial=1.0, p_placeheld=0.5, p_predict_constants_partial=1.0)
        saw_partial = False
        for batch in _iterate(tokenizer, steps=2, batch_size=16, mask_block=cfg):
            assert all(mode is None for mode in batch["mask_mode"]), "partial is unflagged"
            for row, tokens in _rows(batch, tokenizer):
                assert MASK_ALL_TOKEN not in tokens and MASK_FITTABLE_TOKEN not in tokens
                k = batch["n_placeholders"][row]
                mask = batch["task_mask"][row].tolist()
                placeholder_positions = [i for i, t in enumerate(tokens)
                                         if t == MASKED_CONSTANT_TOKEN]
                assert len(placeholder_positions) == k
                if k == 0:
                    continue
                saw_partial = True
                assert all(mask[i] for i in placeholder_positions), \
                    "random placeholders are unlearnable: context-only, loss-masked"
                draw = batch["predict_constants"][row]
                assert draw is not None and len(draw["values"]) == k
        assert saw_partial, "p_placeheld=0.5 over 32 instances must place at least once"


class TestOrderRandomization:
    def test_prefix_elements_permute(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        orders = set()
        for batch in _iterate(tokenizer, steps=2, batch_size=16,
                              mask_block=_mask_cfg(p_mask_all=1.0),
                              complexity_block={"p_present": 1.0, "p_hypothesize": 0.0},
                              predict_y_block={"p_present": 1.0, "p_conditional": 0.0,
                                               "min_n_support": 1}):
            for order in batch["block_order"]:
                orders.add(tuple(order["prefix"]))
        assert len(orders) >= 2, f"three commutative elements never permuted: {orders}"

    def test_hypothesis_element_is_pinned_last(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        for batch in _iterate(tokenizer, steps=1, batch_size=16,
                              mask_block=_mask_cfg(p_mask_all=1.0),
                              complexity_block={"p_present": 0.0, "p_hypothesize": 1.0}):
            for order in batch["block_order"]:
                # The boundary is its OWN element and it is uttered exactly once, after
                # every given element (owner ruling 2026-08-27). From it on, the pen is
                # the model's until </expression>.
                assert order["prefix"][-2:] == ["hypothesize", "complexity"], order["prefix"]
                assert order["prefix"].count("hypothesize") == 1

    def test_suffix_blocks_swap(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        orders = set()
        for batch in _iterate(tokenizer, steps=3, batch_size=16,
                              mask_block=_mask_cfg(p_mask_all=1.0,
                                                   p_predict_constants_flagged=1.0),
                              predict_y_block={"p_present": 1.0, "p_conditional": 1.0,
                                               "min_n_support": 1}):
            for order in batch["block_order"]:
                if len(order["suffix"]) == 2:
                    orders.add(tuple(order["suffix"]))
        assert len(orders) == 2, f"the two suffix blocks never swapped: {orders}"


class TestConfigValidation:
    @pytest.mark.parametrize("bad", [
        {"p_mask_all": 0.05},                                      # missing keys
        {**{"p_mask_all": 0.05, "p_mask_fittable": 0.05, "p_partial": 0.1, "p_placeheld": 0.5,
            "p_predict_constants_flagged": 0.5, "p_predict_constants_partial": 0.9}, "x": 1},
        {"p_mask_all": 0.7, "p_mask_fittable": 0.7, "p_partial": 0.1, "p_placeheld": 0.5,
         "p_predict_constants_flagged": 0.5, "p_predict_constants_partial": 0.9},  # mass > 1
    ])
    def test_malformed_blocks_refuse(self, tokenizer, bad) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        with pytest.raises(ValueError):
            FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                             target_dialect="tagged", mask_block=bad)


class TestSplitsAndAnchor:
    def test_mask_splits_partition_and_anchor_excludes_masked(self, tokenizer) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        from flash_ansr.train.train import _ce_split_metrics

        cfg = _mask_cfg(p_mask_all=1.0, p_predict_constants_flagged=1.0)
        for batch in _iterate(tokenizer, steps=1, batch_size=16, mask_block=cfg):
            batch["labels"] = batch["input_ids"].clone()[..., 1:]
            torch.manual_seed(0)
            logits = torch.randn(batch["input_ids"].shape[0], batch["input_ids"].shape[1],
                                 len(tokenizer))
            parts = _ce_split_metrics(batch, logits, ignore_index=tokenizer["<pad>"])
            assert "expression/mask_all" in parts
            assert "constants/after_flagged" in parts
            assert "expression/anchor" not in parts, \
                "every instance is masked, so nothing is base-shaped"
            assert "constants/partial" not in parts, "no partial rows in a flagged-only run"


class TestPromptSurface:
    def test_mask_prefix_composes_and_stands_alone(self, tokenizer, engine) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
        from flash_ansr.model.flash_ansr_model import FlashANSRModel
        cfg = load_config(get_path("configs", "test", "model.yaml"))
        kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
        model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)

        tokens, numeric = model.complexity_prefix(mask="all")
        assert [tokenizer.vocab[i] for i in tokens] == ["<bos>", MASK_ALL_TOKEN, "<expression>"]
        assert all(np.isnan(v) for v in numeric)

        tokens, numeric = model.complexity_prefix(76000, mask="fittable")
        names = [tokenizer.vocab[i] for i in tokens]
        assert names == ["<bos>", MASK_FITTABLE_TOKEN, "<complexity>", "<float>",
                         "</complexity>", "<expression>"]
        assert numeric[3] == 76000.0

        # The flag stays first and the boundary stays last, whatever is given between them.
        tokens, _ = model.complexity_prefix(76000, mask="all", hypothesize=True)
        assert [tokenizer.vocab[i] for i in tokens] == [
            "<bos>", MASK_ALL_TOKEN, "<complexity>", "<float>", "</complexity>", "<hypothesize>"]

        with pytest.raises(ValueError, match="mask"):
            model.complexity_prefix(mask="everything")
        with pytest.raises(ValueError):
            model.complexity_prefix()


class TestCollectionStability:
    def test_the_check_detects_restructuring(self, engine) -> None:  # noqa: F811  # type: ignore[no-untyped-def]
        # The measured unstable class: two long exact additive constants (one in a
        # <sub> section) merge under collection into a single placeholder whose
        # value is engine-internal (summed, sign-absorbed). Such instances carry no
        # <predict_constants> block until simplipy's value-carrying mask exists.
        from flash_ansr.utils.skeleton import mask_selected_sites, nonspecial_site_positions

        tokens = ["<add>", "<mul>", "3.0636630579799418", "x1", "</mul>",
                  "<sub>", "4.44534694254616499745643103212266", "1.8272203218203917", "</add>"]
        placeheld = [True, True, True]
        collected = mask_selected_sites(engine, tokens, placeheld, collect=True)
        positions = nonspecial_site_positions(engine, tokens)
        expected = list(tokens)
        for pos, ph in zip(positions, placeheld):
            if ph:
                expected[pos] = "<constant>"
        assert collected != expected, "the merge must be detected"
        assert collected.count("<constant>") < sum(placeheld), "placeholders merged"

    def test_plain_substitution_is_stable(self, engine) -> None:  # noqa: F811  # type: ignore[no-untyped-def]
        from flash_ansr.utils.skeleton import mask_selected_sites

        tokens = ["<mul>", "2.71875", "x1", "</mul>"]
        assert mask_selected_sites(engine, tokens, [True], collect=True) == \
            ["<mul>", "<constant>", "x1", "</mul>"]

    def test_subset_masking_counts_sites_exactly(self, engine) -> None:  # noqa: F811  # type: ignore[no-untyped-def]
        from flash_ansr.utils.skeleton import mask_selected_sites

        tokens = ["<add>", "<mul>", "2.71875", "x1", "</mul>", "0.15625", "</add>"]
        out = mask_selected_sites(engine, tokens, [False, True], collect=False)
        assert out == ["<add>", "<mul>", "2.71875", "x1", "</mul>", "<constant>", "</add>"]
        with pytest.raises(ValueError, match="site count"):
            mask_selected_sites(engine, tokens, [True], collect=False)


class TestAuditRegressions:
    def test_the_numeric_channel_never_carries_placeholder_values(self, tokenizer) -> None:  # noqa: F811  # type: ignore[no-untyped-def]
        # THE audit-critical leak: iterate()'s numeric-channel recompute wrote each
        # constant's value at its <constant> position, handing the model the ground
        # truth at exactly the masked placeholder positions. The worker channel is
        # authoritative now; placeholders must be NaN on the model input.
        import math as _math
        cfg = _mask_cfg(p_mask_all=1.0, p_predict_constants_flagged=1.0)
        with FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                              target_dialect="tagged",
                              mask_block=cfg) as ds:
            for batch in ds.iterate(steps=2, batch_size=16):
                vocab = list(tokenizer.vocab)
                for row in range(len(batch["input_ids"])):
                    ids = [int(i) for i in batch["input_ids"][row]]
                    numeric = [float(v) for v in batch["input_num"][row]]
                    for pos, token_id in enumerate(ids):
                        if vocab[token_id] == "<constant>":
                            assert _math.isnan(numeric[pos]), \
                                f"ground truth leaked at position {pos}"

    def test_truncation_keeps_task_channels_aligned(self, tokenizer) -> None:  # noqa: F811  # type: ignore[no-untyped-def]
        # Flag-bearing rows near the budget edge are DROPPED, never emitted with a
        # cut </expression>; surviving rows carry task channels at input length.
        cfg = _mask_cfg(p_mask_all=1.0)
        with FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                              target_dialect="tagged",
                              mask_block=cfg) as ds:
            pad_id = int(tokenizer["<pad>"])
            for batch in ds.iterate(steps=3, batch_size=16, max_seq_len=12):
                for row, tokens in _rows(batch, tokenizer):
                    ids = [int(i) for i in batch["input_ids"][row]]
                    true_len = len(ids)
                    while true_len > 0 and ids[true_len - 1] == pad_id:
                        true_len -= 1
                    # the audit defect: task channels at PRE-truncation length,
                    # silently clipped downstream. They must match the true length.
                    assert len(batch["task_mask"][row]) == true_len
                    assert len(batch["task_segments"][row]) == true_len
                    body = tokens[:true_len]
                    if "<expression>" in body:
                        assert "</expression>" in body, "no cut wrappers"

    def test_the_block_is_cfg_gated(self, tokenizer) -> None:  # noqa: F811  # type: ignore[no-untyped-def]
        # An unconditioned (nulled-memory) instance must not carry a supervised
        # infilling block: the values are unknowable from the prompt.
        cfg = _mask_cfg(p_partial=1.0, p_placeheld=1.0, p_predict_constants_partial=1.0)
        with FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                              target_dialect="tagged",
                              condition_dropout=1.0, mask_block=cfg) as ds:
            for batch in ds.iterate(steps=1, batch_size=16):
                assert all(draw is None for draw in batch["predict_constants"])
                for _, tokens in _rows(batch, tokenizer):
                    assert PREDICT_CONSTANTS_START_TOKEN not in tokens

    def test_explicit_dialect_refuses_mask_block(self, tokenizer) -> None:  # noqa: F811  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="tagged"):
            FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                             target_dialect="explicit", mask_block=_mask_cfg(p_mask_all=1.0))
