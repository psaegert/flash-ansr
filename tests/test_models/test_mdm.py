"""Experiment exp/mdm-decoder: the masked-diffusion decoder, its training objective and its sampler.

The autoregressive path must be untouched by the switch; the masked-diffusion path must attend in both
directions, corrupt exactly the tokens the autoregressive loss supervises (plus the pad tail), weight its
loss 1/r, and denoise every free position of the canvas.
"""
import math

import pytest
import torch

from flash_ansr import FlashANSRModel, get_path
from flash_ansr.train.mdm import RATIO_FLOOR, corrupt, masked_diffusion_loss, targets_and_supervision
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.generation import MaskedDiffusionConfig, create_generation_config
from flash_ansr.utils.numeric import NUMERIC_DTYPE


def _model(objective: str | None) -> FlashANSRModel:
    config = load_config(get_path("configs", "test", "model.yaml"))
    if objective is not None:
        config["decoder_objective"] = objective
    return FlashANSRModel.from_config(config).eval()


@pytest.fixture(scope="module")
def mdm() -> FlashANSRModel:
    torch.manual_seed(0)
    return _model("mdm")


@pytest.fixture(scope="module")
def ar() -> FlashANSRModel:
    torch.manual_seed(0)
    return _model(None)


def _first_position_shift(model: FlashANSRModel) -> float:
    """How much the logits at position 1 move when only the LAST token changes."""
    memory = model._create_memory(torch.rand(1, 13, model.encoder_max_n_variables, dtype=NUMERIC_DTYPE))
    tokens = torch.randint(20, 40, (1, 8))
    changed = tokens.clone()
    changed[0, -1] = 41
    with torch.no_grad():
        a = model.forward(tokens, None, memory=memory)
        b = model.forward(changed, None, memory=memory)
    return float((a[0, 1] - b[0, 1]).abs().max())


def test_the_default_objective_is_autoregressive_and_causal(ar: FlashANSRModel) -> None:
    assert ar.decoder_objective == "ar"
    assert not ar.decoder.bidirectional
    assert all(layer.causal_self_attn for layer in ar.decoder.layers)
    assert _first_position_shift(ar) == 0.0, "a causal decoder must not read a later token"


def test_the_mdm_decoder_reads_both_directions_with_the_same_weights(mdm: FlashANSRModel, ar: FlashANSRModel) -> None:
    assert mdm.decoder.bidirectional and not any(layer.causal_self_attn for layer in mdm.decoder.layers)
    assert _first_position_shift(mdm) > 0.0
    assert {k: v.shape for k, v in mdm.state_dict().items()} == {k: v.shape for k, v in ar.state_dict().items()}


def test_the_mdm_decoder_refuses_the_token_by_token_paths(mdm: FlashANSRModel) -> None:
    assert mdm.supports_static_decode()[0] is False
    with pytest.raises(ValueError, match="masked-diffusion"):
        mdm.sample_top_kp(torch.rand(1, 13, mdm.encoder_max_n_variables, dtype=NUMERIC_DTYPE), draws=2)
    with pytest.raises(ValueError):
        _model("diffusion")


def _toy_batch(model: FlashANSRModel) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Two collated rows: <bos> <harness> <expression> body </expression> <eos> <pad>..., with the
    autoregressive labels masked at the harness token, as the trainer's prompt/task masks leave them."""
    t = model.tokenizer
    ids = {name: int(t[name]) for name in ("<pad>", "<bos>", "<eos>", "<expression>", "</expression>", "<mask>", "<float>")}
    body = [int(t["x1"]), int(t["x2"])]
    row_a = [ids["<bos>"], ids["<float>"], ids["<expression>"], *body, ids["</expression>"], ids["<eos>"], ids["<pad>"]]
    row_b = [ids["<bos>"], ids["<expression>"], body[0], ids["</expression>"], ids["<eos>"], ids["<pad>"], ids["<pad>"], ids["<pad>"]]
    input_ids = torch.tensor([row_a, row_b])
    labels = input_ids[:, 1:].clone()
    labels[0, 0] = ids["<pad>"]          # the harness token at position 1 is not supervised
    return input_ids, labels, ids


def test_targets_follow_the_autoregressive_supervision_plus_the_pad_tail(mdm: FlashANSRModel) -> None:
    input_ids, labels, ids = _toy_batch(mdm)
    maskable, supervised = targets_and_supervision(input_ids, labels, ids["<pad>"], ids["<eos>"], ids["<expression>"])
    # row a: <bos> and the harness token are given, <expression> too; body, </expression>, <eos> supervised; the pad is a target.
    assert supervised[0].tolist() == [False, False, False, True, True, True, True, False]
    assert maskable[0].tolist() == [False, False, False, True, True, True, True, True]
    assert supervised[1].tolist() == [False, False, True, True, True, False, False, False]
    assert maskable[1].tolist() == [False, False, True, True, True, True, True, True]


def test_corruption_masks_only_targets_and_pads_to_the_canvas(mdm: FlashANSRModel) -> None:
    input_ids, labels, ids = _toy_batch(mdm)
    input_num = torch.full((*input_ids.shape, 1), float("nan"), dtype=NUMERIC_DTYPE)
    input_num[0, 1, 0] = 2.5
    generator = torch.Generator().manual_seed(1)
    batch = corrupt(input_ids, labels, ignore_index=ids["<pad>"], eos_id=ids["<eos>"], expression_id=ids["<expression>"],
                    mask_id=ids["<mask>"], canvas=16, input_num=input_num, generator=generator)
    assert batch.noisy_ids.shape == (2, 16) and batch.targets.shape == (2, 16)
    maskable, supervised = targets_and_supervision(batch.targets, labels, ids["<pad>"], ids["<eos>"], ids["<expression>"])
    assert bool((batch.masked <= maskable).all())
    assert bool((batch.noisy_ids[batch.masked] == ids["<mask>"]).all())
    assert bool((batch.noisy_ids[~batch.masked] == batch.targets[~batch.masked]).all())
    assert int(batch.n_supervised) == int(supervised.sum()) == 7
    assert float(batch.input_num[0, 1, 0]) == 2.5, "the given numeric payload survives"
    assert bool(((batch.ratio >= RATIO_FLOOR) & (batch.ratio <= 1.0)).all())
    # every target of the canvas tail (positions 8..15) is a pad target
    assert bool(maskable[:, 8:].all())


def test_the_ratios_are_stratified_over_the_batch() -> None:
    input_ids = torch.ones(64, 8, dtype=torch.long)
    labels = torch.ones(64, 7, dtype=torch.long)
    batch = corrupt(input_ids, labels, ignore_index=0, eos_id=2, expression_id=3, mask_id=4, canvas=8,
                    generator=torch.Generator().manual_seed(0))
    gaps = torch.sort(batch.ratio).values.diff()
    assert torch.allclose(gaps, torch.full_like(gaps, 1 / 64), atol=1e-5), "one ratio per 1/B stratum"


def test_the_loss_is_the_inverse_ratio_weighted_masked_cross_entropy() -> None:
    input_ids = torch.tensor([[1, 5, 6, 7, 2, 0]])
    labels = input_ids[:, 1:].clone()
    batch = corrupt(input_ids, labels, ignore_index=0, eos_id=2, expression_id=3, mask_id=4, canvas=6,
                    generator=torch.Generator().manual_seed(3))
    batch.masked[:] = False
    batch.masked[0, 2] = batch.masked[0, 5] = True
    batch.ratio[:] = 0.25
    logits = torch.randn(1, 6, 9)
    loss, masked_logits = masked_diffusion_loss(logits, batch)
    expected = 4.0 * (torch.nn.functional.cross_entropy(logits[0, [2, 5]], input_ids[0, [2, 5]], reduction="sum")) / 4
    assert masked_logits.shape == (2, 9)
    assert math.isclose(float(loss), float(expected), rel_tol=1e-6)


@pytest.mark.parametrize("order", ["random", "confidence"])
def test_the_sampler_fills_every_free_position(mdm: FlashANSRModel, order: str) -> None:
    data = torch.rand(1, 13, mdm.encoder_max_n_variables, dtype=NUMERIC_DTYPE)
    torch.manual_seed(0)
    sequences, scores = mdm.sample_masked_diffusion(data, draws=6, order=order, batch_size=4, canvas=24, return_raw=True)
    assert len(sequences) == len(scores) == 6
    mask_id = int(mdm.tokenizer["<mask>"])
    for seq, score in zip(sequences, scores):
        assert mask_id not in seq
        assert math.isfinite(score)
        assert seq[0] == int(mdm.tokenizer["<bos>"])
    # fewer steps than positions reveal several at once and still finish
    sequences, _ = mdm.sample_masked_diffusion(data, draws=3, steps=4, order=order, canvas=24, return_raw=True)
    assert all(mask_id not in seq for seq in sequences)


def test_the_generation_config_names_the_diffusion_sampler() -> None:
    config = create_generation_config(method="masked_diffusion", draws=8, steps=16, order="confidence")
    assert isinstance(config, MaskedDiffusionConfig)
    assert config.to_kwargs()["steps"] == 16 and config.method == "masked_diffusion"
    with pytest.raises(ValueError):
        create_generation_config(method="masked_diffusion", order="left_to_right")
