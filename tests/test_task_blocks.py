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

COMPLEXITY_NIBBLES = {"p_present": 1.0, "p_nibbles": 1.0}
COMPLEXITY_FLOAT = {"p_present": 1.0, "p_nibbles": 0.0}
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
