"""v24 task blocks end-to-end: the <complexity> block (nibbles / <float> summary), the
<predict_y> block (unconditional / conditional placement, clean-holdout semantics), and
the task_mask loss discipline (harness owns the grammar, model owns nibbles + closers)."""
import numpy as np
import pytest
import torch

from flash_ansr import FlashANSRDataset, get_path
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.train.train import Trainer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.ieee754 import IEEE754_N_NIBBLES, nibble_tokens_to_float32

COMPLEXITY_NIBBLES = {"p_present": 1.0, "p_nibbles": 1.0, "p_hypothesize": 0.0}
COMPLEXITY_FLOAT = {"p_present": 1.0, "p_nibbles": 0.0, "p_hypothesize": 0.0}
PREDICT_A = {"p_present": 1.0, "p_conditional": 0.0, "min_n_support": 1}
PREDICT_B = {"p_present": 1.0, "p_conditional": 1.0, "min_n_support": 1}


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(load_config(get_path("configs", "v24-template", "tokenizer.yaml")))


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("base", install=True)


def _source():  # type: ignore[no-untyped-def]
    from symbolic_data import ProblemSource
    from symbolic_data.generative import LampleChartonCatalog

    catalog = LampleChartonCatalog.from_config({
        "type": "lample_charton",
        "simplipy_engine": "base",
        "holdout_pools": [],
        "sample_strategy": {
            "n_operator_distribution": "length_proportional",
            "min_operators": 1, "max_operators": 6, "power": 1,
            "max_length": 21, "max_tries": 1, "independent_dimensions": True,
        },
        "allow_nan": False,
        "simplify": True,
        "literal_prior": {"name": "normal", "kwargs": {"loc": 0, "scale": 5}},
        "support_sampler": {
            "support_prior": {"name": "uniform", "kwargs": {"low": -5, "high": 5, "min_value": -5, "max_value": 5}},
            "n_support_prior": {"name": "uniform", "kwargs": {"low": 12, "high": 16, "min_value": 12, "max_value": 16}},
        },
        "variables": ["x1", "x2", "x3"],
        "operator_weights": {"+": 10, "-": 10, "*": 10, "sin": 2},
    })
    catalog.skeletons = {
        ("*", "<constant>", "x1"),
        ("+", "<constant>", "*", "<constant>", "x2"),
        ("+", "*", "<constant>", "x1", "*", "<constant>", "sin", "x2"),
    }
    catalog.skeleton_codes = catalog.compile_codes()
    return ProblemSource({"catalog": catalog,
                          "sampling": {"n_support": "prior", "n_validation": 0, "noise": 0.0}})


def _iterate(tokenizer, steps=2, batch_size=8, **blocks):  # type: ignore[no-untyped-def]
    with FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                          constant_representation="ieee754_mixed",
                          target_dialect="tagged", **blocks) as dataset:
        for batch in dataset.iterate(steps=steps, batch_size=batch_size):
            yield dataset.collate(batch, device=torch.device("cpu"))


def _rows(batch, tokenizer):  # type: ignore[no-untyped-def]
    vocab = list(tokenizer.vocab)
    for row in range(batch["input_ids"].shape[0]):
        ids = batch["input_ids"][row].tolist()
        tokens = [vocab[i] for i in ids]
        yield row, tokens


def test_complexity_nibbles_block(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    for batch in _iterate(tokenizer, complexity_block=COMPLEXITY_NIBBLES):
        assert "task_mask" in batch and batch["task_mask"].dtype == torch.bool
        for row, tokens in _rows(batch, tokenizer):
            assert tokens[:3] == ["<bos>", "<complexity>", "<ieee754>"]
            nibbles = tokens[3:3 + IEEE754_N_NIBBLES]
            assert tokens[3 + IEEE754_N_NIBBLES:3 + IEEE754_N_NIBBLES + 3] == \
                ["</ieee754>", "</complexity>", "<expression>"]
            mu = batch["complexity_mu"][row]
            assert nibble_tokens_to_float32(nibbles) == float(np.float32(mu))
            assert batch["complexity_variant"][row] == "nibbles"
            # mu is the masked target's complexity, recomputable from the streamed skeleton
            assert mu == engine.complexity(list(batch["skeleton"][row]))
            # loss discipline: openers/selectors masked, nibbles + closers supervised
            mask = batch["task_mask"][row].tolist()
            assert mask[1] and mask[2], "opener and selector must be loss-masked"
            assert not any(mask[3:3 + IEEE754_N_NIBBLES]), "nibbles are supervised"
            assert not mask[3 + IEEE754_N_NIBBLES] and not mask[4 + IEEE754_N_NIBBLES], \
                "closing tags are supervised"


def test_complexity_float_block_rides_the_numeric_channel(tokenizer) -> None:  # type: ignore[no-untyped-def]
    for batch in _iterate(tokenizer, complexity_block=COMPLEXITY_FLOAT):
        for row, tokens in _rows(batch, tokenizer):
            assert tokens[1:4] == ["<complexity>", "<float>", "</complexity>"]
            assert batch["complexity_variant"][row] == "float"
            value = float(batch["input_num"][row, 2, 0])
            assert value == float(np.float32(batch["complexity_mu"][row]))
            assert all(batch["task_mask"][row][1:4].tolist()), \
                "the summary variant is pure context: no loss anywhere in the block"


def test_predict_y_unconditional_before_expression(tokenizer) -> None:  # type: ignore[no-untyped-def]
    for batch in _iterate(tokenizer, predict_y_block=PREDICT_A):
        for row, tokens in _rows(batch, tokenizer):
            draw = batch["predict_y"][row]
            assert draw is not None and draw["conditional"] is False
            assert tokens[1:3] == ["<predict_y>", "<point>"]
            n_dims = len(draw["x"])
            assert tokens[3:3 + n_dims] == ["<float>"] * n_dims
            for k in range(n_dims):
                assert float(batch["input_num"][row, 3 + k, 0]) == pytest.approx(
                    float(np.float32(draw["x"][k])), rel=0, abs=0)
            tail = tokens[3 + n_dims:3 + n_dims + 2]
            assert tail == ["</point>", "<ieee754>"]
            nibbles = tokens[5 + n_dims:5 + n_dims + IEEE754_N_NIBBLES]
            assert nibble_tokens_to_float32(nibbles) == draw["y"]
            assert tokens[5 + n_dims + IEEE754_N_NIBBLES:8 + n_dims + IEEE754_N_NIBBLES] == \
                ["</ieee754>", "</predict_y>", "<expression>"]
            # the held-out point is REMOVED from the encoder set (prior-exact holdout)
            n = int(batch["data_attn_mask"][row].sum())
            assert batch["n_support"][row] == n
            x_rows = batch["x_tensors"][row, :n, :n_dims].numpy()
            x_star = np.asarray(draw["x"], dtype=np.float32)
            assert not np.any(np.all(x_rows == x_star, axis=1)), "holdout row must not be in the input set"


def test_predict_y_conditional_after_expression(tokenizer) -> None:  # type: ignore[no-untyped-def]
    for batch in _iterate(tokenizer, predict_y_block=PREDICT_B):
        for row, tokens in _rows(batch, tokenizer):
            draw = batch["predict_y"][row]
            assert draw is not None and draw["conditional"] is True
            assert tokens[1] == "<expression>"
            end = tokens.index("</expression>")
            assert tokens[end + 1] == "<predict_y>"
            eos = tokens.index("<eos>")
            assert tokens[eos - 1] == "</predict_y>"


def test_task_mask_shifted_label_application() -> None:
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.metrics_ignore_index = 0
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 2]])
    labels = input_ids.clone()[..., 1:]
    task_mask = torch.tensor([[False, True, False, True, False, False]])
    batch = {"task_mask": task_mask, "labels": labels}
    trainer._apply_task_mask(batch)
    assert batch["labels"].tolist() == [[0, 6, 0, 8, 2]]


def test_without_block_configs_the_batch_surface_is_unchanged(tokenizer) -> None:  # type: ignore[no-untyped-def]
    for batch in _iterate(tokenizer):
        for key in ("task_mask", "complexity_mu", "complexity_variant", "predict_y"):
            assert key not in batch


def test_complexity_prefix_for_generation(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.model.flash_ansr_model import FlashANSRModel
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)

    tokens, numeric = model.complexity_prefix(76000)
    names = [tokenizer.vocab[i] for i in tokens]
    assert names == ["<bos>", "<complexity>", "<float>", "</complexity>"]
    assert numeric[2] == 76000.0 and all(np.isnan(v) for i, v in enumerate(numeric) if i != 2)

    tokens, numeric = model.complexity_prefix(predict=True)
    assert [tokenizer.vocab[i] for i in tokens] == ["<bos>", "<complexity>", "<ieee754>"]
    assert all(np.isnan(v) for v in numeric)

    with pytest.raises(ValueError, match="exactly one"):
        model.complexity_prefix()
    with pytest.raises(ValueError, match="exactly one"):
        model.complexity_prefix(42.0, predict=True)


def test_task_segments_label_the_blocks(tokenizer) -> None:  # type: ignore[no-untyped-def]
    for batch in _iterate(tokenizer, complexity_block=COMPLEXITY_NIBBLES, predict_y_block=PREDICT_A):
        assert batch["task_segments"].dtype == torch.long
        for row, tokens in _rows(batch, tokenizer):
            segments = batch["task_segments"][row].tolist()
            for position, token in enumerate(tokens):
                if token == "<pad>":
                    break
                inside_complexity = "<complexity>" in tokens[:position + 1] and "</complexity>" not in tokens[:position]
                inside_predict = "<predict_y>" in tokens[:position + 1] and "</predict_y>" not in tokens[:position]
                expected = 1 if inside_complexity else 2 if inside_predict else 0
                assert segments[position] == expected, (position, token)


def test_per_task_ce_splits_by_segment() -> None:
    from flash_ansr.train.train import _per_task_ce
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 10)
    #                 labels for positions 1..5 of the input; 9 = ignore_index
    labels = torch.tensor([[4, 9, 5, 6, 7]])
    segments = torch.tensor([[0, 0, 1, 1, 2, 0]])   # aligned with input positions 0..5
    parts = _per_task_ce(logits, labels, segments, ignore_index=9)
    assert set(parts) == {"expression", "complexity", "predict_y"}
    # labels sit at input positions 1..5; position 2 is ignore_index. Valid: pos 1 (seg 0),
    # pos 3 (seg 1), pos 4 (seg 2), pos 5 (seg 0).
    assert parts["expression"][1] == 2
    assert parts["complexity"][1] == 1
    assert parts["predict_y"][1] == 1
    assert sum(count for _, count in parts.values()) == 4


def test_ce_split_metrics_cross_tasks_with_conditioning(tokenizer) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.train.train import _ce_split_metrics

    with FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                          constant_representation="ieee754_mixed", target_dialect="tagged",
                          condition_dropout=0.5,
                          complexity_block={"p_present": 0.5, "p_nibbles": 1.0, "p_hypothesize": 0.0},
                          predict_y_block={"p_present": 1.0, "p_conditional": 0.5, "min_n_support": 1}) as ds:
        for batch in ds.iterate(steps=1, batch_size=32):
            batch = ds.collate(batch, device=torch.device("cpu"))
            batch["labels"] = batch["input_ids"].clone()[..., 1:]
            vocab_size = len(tokenizer)
            torch.manual_seed(0)
            logits = torch.randn(batch["input_ids"].shape[0], batch["input_ids"].shape[1], vocab_size)
            parts = _ce_split_metrics(batch, logits, ignore_index=tokenizer["<pad>"])

            # with 32 instances at these priors every split should occur
            expected = {"expression/data_cond", "expression/data_uncond",
                        "expression/complexity_present", "expression/complexity_absent",
                        "complexity/data_cond", "predict_y/conditional", "predict_y/unconditional"}
            assert expected <= set(parts), sorted(parts)
            # the two data-conditioning expression splits partition the expression tokens
            cond, uncond = parts["expression/data_cond"], parts["expression/data_uncond"]
            n_expression = int(((batch["task_segments"][..., 1:] == 0)
                                & (batch["labels"] != tokenizer["<pad>"])).sum())
            assert cond[1] + uncond[1] == n_expression
            # predict_y splits partition the predict_y tokens
            n_predict = int(((batch["task_segments"][..., 1:] == 2)
                             & (batch["labels"] != tokenizer["<pad>"])).sum())
            assert parts["predict_y/conditional"][1] + parts["predict_y/unconditional"][1] == n_predict
            # An unconditioned instance DOES carry predict_y, but only in the SUFFIX placement:
            # with the expression in scope the task is function evaluation, which is well posed.
            # The prefix placement there would be nulled memory AND no expression -- nothing to
            # condition on -- so the gate pins rather than drops (owner ruling 2026-08-26).
            n_uncond_blocks = 0
            for row, draw in enumerate(batch["predict_y"]):
                if draw is not None and not bool(batch["condition_mask"][row]):
                    n_uncond_blocks += 1
                    assert draw["conditional"] is True, (
                        "an unconditioned instance must take the suffix placement")
            # p_present=1.0 with condition_dropout=0.5 over 32 rows: vacuous pass is not credible.
            assert n_uncond_blocks > 0, "no unconditioned instance carried a block to check"


HYPOTHESIS_ONLY = {"p_present": 0.0, "p_nibbles": 1.0, "p_hypothesize": 1.0}


def test_hypothesis_mode_supervises_the_whole_block_after_the_flag(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    for batch in _iterate(tokenizer, complexity_block=HYPOTHESIS_ONLY):
        for row, tokens in _rows(batch, tokenizer):
            assert tokens[:4] == ["<bos>", "<hypothesize>", "<complexity>", "<ieee754>"]
            assert batch["complexity_variant"][row] == "hypothesis"
            nibbles = tokens[4:4 + IEEE754_N_NIBBLES]
            mu = batch["complexity_mu"][row]
            assert nibble_tokens_to_float32(nibbles) == float(np.float32(mu))
            assert mu == engine.complexity(list(batch["skeleton"][row]))
            mask = batch["task_mask"][row].tolist()
            assert mask[1], "the flag itself is NEVER supervised (harness-only)"
            block_end = 4 + IEEE754_N_NIBBLES + 2   # through </ieee754> </complexity>
            assert not any(mask[2:block_end]), \
                "opener, selector, nibbles and closers are the model's own hypothesis: all supervised"


def test_hypothesis_config_validation(tokenizer) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="exceed"):
        FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                         constant_representation="ieee754_mixed", target_dialect="tagged",
                         complexity_block={"p_present": 0.6, "p_nibbles": 0.5, "p_hypothesize": 0.6})
    with pytest.raises(ValueError, match="p_hypothesize"):
        FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                         constant_representation="ieee754_mixed", target_dialect="tagged",
                         complexity_block={"p_present": 1.0, "p_nibbles": 1.0})  # missing key
    old_tokenizer = Tokenizer.from_config(load_config(get_path("configs", "v24.0-T13", "tokenizer.yaml")))
    with pytest.raises(ValueError, match="hypothesize"):
        FlashANSRDataset(source=_source(), tokenizer=old_tokenizer, padding="zero",
                         constant_representation="ieee754_mixed", target_dialect="tagged",
                         complexity_block=HYPOTHESIS_ONLY)


def test_complexity_prefix_hypothesize_mode(tokenizer, engine) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.model.flash_ansr_model import FlashANSRModel
    cfg = load_config(get_path("configs", "test", "model.yaml"))
    kwargs = {k: v for k, v in cfg.items() if k not in ("simplipy_engine", "tokenizer")}
    model = FlashANSRModel(simplipy_engine=engine, tokenizer=tokenizer, **kwargs)
    tokens, numeric = model.complexity_prefix(hypothesize=True)
    assert [tokenizer.vocab[i] for i in tokens] == ["<bos>", "<hypothesize>"]
    assert all(np.isnan(v) for v in numeric)
    with pytest.raises(ValueError, match="exactly one"):
        model.complexity_prefix(42.0, hypothesize=True)


def test_ce_split_anchor_is_the_cross_arm_common_ground(tokenizer) -> None:  # type: ignore[no-untyped-def]
    from flash_ansr.train.train import _ce_split_metrics

    with FlashANSRDataset(source=_source(), tokenizer=tokenizer, padding="zero",
                          constant_representation="ieee754_mixed", target_dialect="tagged",
                          condition_dropout=0.5,
                          complexity_block={"p_present": 0.5, "p_nibbles": 1.0, "p_hypothesize": 0.0},
                          predict_y_block={"p_present": 1.0, "p_conditional": 0.5, "min_n_support": 1}) as ds:
        for batch in ds.iterate(steps=1, batch_size=32):
            batch = ds.collate(batch, device=torch.device("cpu"))
            batch["labels"] = batch["input_ids"].clone()[..., 1:]
            torch.manual_seed(0)
            logits = torch.randn(batch["input_ids"].shape[0], batch["input_ids"].shape[1], len(tokenizer))
            parts = _ce_split_metrics(batch, logits, ignore_index=tokenizer["<pad>"])
            assert "expression/anchor" in parts
            # the definition, recomputed independently: complexity absent AND predict_y
            # absent-or-conditional (a conditional block follows the expression and
            # cannot reach its logits under causal masking)
            qualifies = [
                batch["complexity_variant"][i] is None
                and (batch["predict_y"][i] is None or bool(batch["predict_y"][i]["conditional"]))
                for i in range(len(batch["complexity_variant"]))
            ]
            seg = batch["task_segments"][..., 1:]
            valid = batch["labels"] != tokenizer["<pad>"]
            expected_n = int((valid & (seg == 0) & torch.tensor(qualifies)[:, None]).sum())
            assert parts["expression/anchor"][1] == expected_n
            assert expected_n <= int((valid & (seg == 0)).sum())


def test_ce_split_anchor_exists_in_base_shaped_batches() -> None:
    # No task blocks at all -- no task_segments key, no block metadata. The anchor must
    # still be emitted so base arms carry the common-ground curve, and it must cover the
    # whole batch: every position is expression, every instance qualifies.
    from flash_ansr.train.train import _ce_split_metrics

    labels = torch.tensor([[4, 9, 5], [6, 7, 9]])
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)
    parts = _ce_split_metrics({"labels": labels}, logits, ignore_index=9)
    assert "expression/anchor" in parts
    assert parts["expression/anchor"][1] == int((labels != 9).sum())
