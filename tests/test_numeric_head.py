import math
from typing import Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.model.flash_ansr_model import FlashANSRModel, NumericHeadPrediction
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.model.mdn import MixtureDensityNetwork
from flash_ansr.utils import GenerationConfig, get_path, load_config


MODEL_CONFIGS = {
    'none': 'model.yaml',
    'deterministic': 'model_numeric_head_deterministic.yaml',
    'mdn': 'model_numeric_head.yaml',
}


def _build_model(kind: str) -> FlashANSRModel:
    config_name = MODEL_CONFIGS[kind]
    config = load_config(get_path('configs', 'test', config_name))
    if isinstance(config, dict) and 'use_numeric_head' not in config:
        config = dict(config)
    model = FlashANSRModel.from_config(config)
    model.eval()
    return model


def _configure_mdn_head(
    model: FlashANSRModel,
    *,
    mean: float,
    sigma: float,
    favored_component: int = 0,
) -> None:
    assert model.numeric_head_kind == 'mdn'
    mdn = model.constant_value_head
    assert isinstance(mdn, MixtureDensityNetwork)

    eps = 1e-6

    with torch.no_grad():
        for module in mdn.pi_network:
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        for module in mdn.normal_network:
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()

        final_pi = mdn.pi_network[-1]
        final_norm = mdn.normal_network[-1]
        assert isinstance(final_pi, nn.Linear)
        assert isinstance(final_norm, nn.Linear)

        logits = torch.full_like(final_pi.bias, fill_value=-20.0)
        logits[favored_component] = 20.0
        final_pi.bias.copy_(logits)

        dim_out = mdn.dim_out
        n_components = mdn.n_components
        mu_bias = torch.full((dim_out * n_components,), mean, device=final_norm.bias.device, dtype=final_norm.bias.dtype)
        raw_sigma = math.log(sigma) - eps
        sigma_bias = torch.full((dim_out * n_components,), raw_sigma, device=final_norm.bias.device, dtype=final_norm.bias.dtype)

        final_norm.bias[:dim_out * n_components].copy_(mu_bias)
        final_norm.bias[dim_out * n_components:].copy_(sigma_bias)


def test_forward_without_numeric_head_returns_logits_only() -> None:
    model = _build_model('none')

    batch_size = 2
    set_size = 3
    seq_len = 4

    input_tokens = torch.full((batch_size, seq_len), model.tokenizer['<bos>'], dtype=torch.long)
    data_tensor = torch.zeros((batch_size, set_size, model.encoder_max_n_variables), dtype=torch.float32)

    logits, numeric_pred = model.forward(input_tokens, data_tensor, input_num=None)
    assert logits.shape == (batch_size, seq_len, len(model.tokenizer))
    assert numeric_pred is None


def test_forward_with_deterministic_numeric_head_returns_predictions() -> None:
    model = _build_model('deterministic')

    batch_size = 2
    set_size = 3
    seq_len = 4

    input_tokens = torch.full((batch_size, seq_len), model.tokenizer['<bos>'], dtype=torch.long)
    data_tensor = torch.zeros((batch_size, set_size, model.encoder_max_n_variables), dtype=torch.float32)
    numeric_tokens = torch.zeros((batch_size, seq_len, 1), dtype=torch.float32)

    output = model.forward(input_tokens, data_tensor, input_num=numeric_tokens)
    assert isinstance(output, tuple)
    logits, numeric_pred = output
    assert logits.shape == (batch_size, seq_len, len(model.tokenizer))
    assert isinstance(numeric_pred, NumericHeadPrediction)
    assert numeric_pred.kind == 'deterministic'
    assert numeric_pred.values is not None
    assert numeric_pred.values.shape == (batch_size, seq_len)
    assert torch.isfinite(numeric_pred.values).all()


def test_forward_with_mdn_numeric_head_returns_predictions() -> None:
    model = _build_model('mdn')

    batch_size = 2
    set_size = 3
    seq_len = 4

    input_tokens = torch.full((batch_size, seq_len), model.tokenizer['<bos>'], dtype=torch.long)
    data_tensor = torch.zeros((batch_size, set_size, model.encoder_max_n_variables), dtype=torch.float32)
    numeric_tokens = torch.zeros((batch_size, seq_len, 1), dtype=torch.float32)

    output = model.forward(input_tokens, data_tensor, input_num=numeric_tokens)
    assert isinstance(output, tuple), "forward should return (logits, numeric) when numeric head is active"
    logits, numeric_pred = output

    assert logits.shape == (batch_size, seq_len, len(model.tokenizer))
    assert isinstance(numeric_pred, NumericHeadPrediction)
    assert numeric_pred.kind == 'mdn'
    assert numeric_pred.log_pi is not None
    assert numeric_pred.log_pi.shape[:2] == (batch_size, seq_len)
    torch.manual_seed(0)
    sample = numeric_pred.sample()
    assert sample.shape[:2] == (batch_size, seq_len)


def _deterministic_sequence(tokenizer: Tokenizer) -> tuple[list[int], list[float]]:
    tokens = [
        tokenizer['<prompt>'],
        tokenizer['<complexity>'],
        tokenizer['<float>'],
        tokenizer['</complexity>'],
        tokenizer['</prompt>'],
        tokenizer['<expression>'],
        tokenizer['*'],
        tokenizer['x1'],
        tokenizer['<constant>'],
        tokenizer['</expression>'],
        tokenizer['<eos>'],
    ]

    numeric = [float('nan')] * len(tokens)
    numeric[8] = 2.5
    return tokens, numeric


def _install_deterministic_forward(
    model: FlashANSRModel,
    target_tokens: Sequence[int],
    numeric_values: Sequence[float],
) -> callable:
    tokenizer = model.tokenizer
    vocab_size = len(tokenizer)

    def fake_forward(self, input_tokens, data, input_num=None, memory=None, data_attn_mask=None):  # noqa: ANN001, ANN201
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        logits = torch.full((batch_size, seq_len, vocab_size), -50.0, device=device)
        for step in range(seq_len):
            token_idx = target_tokens[min(step, len(target_tokens) - 1)]
            logits[:, step, token_idx] = 0.0

        numeric_pred = torch.full((batch_size, seq_len), float('nan'), device=device)
        finite_defaults = [value for value in numeric_values if not math.isnan(value)]
        constant_token_id = tokenizer['<constant>']
        constant_value = finite_defaults[0] if finite_defaults else 0.0

        for step in range(seq_len):
            if step < len(numeric_values):
                value = numeric_values[step]
                if not math.isnan(value):
                    numeric_pred[:, step] = value

            constant_positions = (input_tokens[:, step] == constant_token_id)
            if constant_positions.any():
                numeric_pred[constant_positions, step] = constant_value

        numeric_prediction = NumericHeadPrediction(kind='deterministic', values=numeric_pred)
        return logits, numeric_prediction

    original_forward = model.forward
    model.forward = fake_forward.__get__(model, FlashANSRModel)
    return original_forward


def _install_symbolic_forward(
    model: FlashANSRModel,
    target_tokens: Sequence[int],
) -> callable:
    tokenizer = model.tokenizer
    vocab_size = len(tokenizer)

    def fake_forward(self, input_tokens, data, input_num=None, memory=None, data_attn_mask=None):  # noqa: ANN001, ANN201
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        logits = torch.full((batch_size, seq_len, vocab_size), -50.0, device=device)
        for step in range(seq_len):
            token_idx = target_tokens[min(step, len(target_tokens) - 1)]
            logits[:, step, token_idx] = 0.0

        return logits, None

    original_forward = model.forward
    model.forward = fake_forward.__get__(model, FlashANSRModel)
    return original_forward


def _install_mdn_forward(
    model: FlashANSRModel,
    target_tokens: Sequence[int],
    numeric_value: float,
) -> callable:
    tokenizer = model.tokenizer
    vocab_size = len(tokenizer)

    def fake_forward(self, input_tokens, data, input_num=None, memory=None, data_attn_mask=None):  # noqa: ANN001, ANN201
        batch_size, seq_len = input_tokens.shape
        logits = torch.full((batch_size, seq_len, vocab_size), -50.0, device=input_tokens.device)
        for step in range(seq_len):
            token_idx = target_tokens[min(step, len(target_tokens) - 1)]
            logits[:, step, token_idx] = 0.0

        n_components = 3
        log_pi = torch.full((batch_size, seq_len, n_components), fill_value=-10.0, device=input_tokens.device)
        log_pi[..., 1] = 0.0
        mu = torch.full((batch_size, seq_len, n_components, 1), fill_value=numeric_value, device=input_tokens.device)
        sigma = torch.zeros_like(mu)

        prediction = NumericHeadPrediction(kind='mdn', log_pi=log_pi, mu=mu, sigma=sigma)
        return logits, prediction

    original_forward = model.forward
    model.forward = fake_forward.__get__(model, FlashANSRModel)
    return original_forward


def test_beam_search_records_latest_numeric_sequences() -> None:
    model = _build_model('deterministic')

    tokenizer = model.tokenizer
    target_tokens, numeric_values = _deterministic_sequence(tokenizer)

    restore_forward = _install_deterministic_forward(model, target_tokens, numeric_values)
    try:
        data_tensor = torch.zeros((1, 2, model.encoder_max_n_variables), dtype=torch.float32)
        beams, scores, completed = model.beam_search(
            data=data_tensor,
            beam_width=1,
            max_len=len(target_tokens),
            mini_batch_size=1,
            equivalence_pruning=False,
            limit_expansions=True,
        )
    finally:
        model.forward = restore_forward

    assert beams, "expected at least one beam"
    assert scores and completed

    numeric_sequences = model.get_latest_numeric_sequences()
    assert numeric_sequences is not None, "numeric sequences should be cached when numeric head is enabled"
    assert len(numeric_sequences) == len(beams)

    constant_token = tokenizer['<constant>']
    constant_index = beams[0].index(constant_token)
    assert numeric_sequences[0][constant_index] == pytest.approx(2.5)


def test_beam_search_without_numeric_head_skips_numeric_sequences() -> None:
    model = _build_model('none')

    tokenizer = model.tokenizer
    target_tokens, _numeric_values = _deterministic_sequence(tokenizer)

    restore_forward = _install_symbolic_forward(model, target_tokens)
    try:
        data_tensor = torch.zeros((1, 2, model.encoder_max_n_variables), dtype=torch.float32)
        beams, _, _ = model.beam_search(
            data=data_tensor,
            beam_width=1,
            max_len=len(target_tokens),
            mini_batch_size=1,
            equivalence_pruning=False,
            limit_expansions=True,
        )
    finally:
        model.forward = restore_forward

    assert beams, "expected at least one beam"
    assert model.get_latest_numeric_sequences() is None


def test_beam_search_with_mdn_head_samples_numeric_sequences() -> None:
    model = _build_model('mdn')

    tokenizer = model.tokenizer
    target_tokens, _numeric_values = _deterministic_sequence(tokenizer)
    numeric_value = 1.75

    restore_forward = _install_mdn_forward(model, target_tokens, numeric_value)
    try:
        data_tensor = torch.zeros((1, 2, model.encoder_max_n_variables), dtype=torch.float32)
        beams, _, _ = model.beam_search(
            data=data_tensor,
            beam_width=1,
            max_len=len(target_tokens),
            mini_batch_size=1,
            equivalence_pruning=False,
            limit_expansions=True,
        )
    finally:
        model.forward = restore_forward

    assert beams, "expected at least one beam"
    numeric_sequences = model.get_latest_numeric_sequences()
    assert numeric_sequences is not None
    constant_token = tokenizer['<constant>']
    constant_index = beams[0].index(constant_token)
    assert numeric_sequences[0][constant_index] == pytest.approx(numeric_value, rel=1e-3, abs=1e-3)


def test_flash_ansr_fit_uses_numeric_predictions() -> None:
    model = _build_model('deterministic')
    tokenizer = model.tokenizer

    target_tokens, numeric_values = _deterministic_sequence(tokenizer)

    restore_forward = _install_deterministic_forward(model, target_tokens, numeric_values)
    try:
        generation_config = GenerationConfig(
            method='beam_search',
            beam_width=1,
            max_len=len(target_tokens),
            mini_batch_size=1,
            equivalence_pruning=False,
            limit_expansions=True,
        )

        flash = FlashANSR(
            simplipy_engine=model.simplipy_engine,
            flash_ansr_transformer=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            numeric_head=True,
            n_restarts=1,
            refiner_method='curve_fit_lm',
            refiner_p0_noise=None,
            refiner_p0_noise_kwargs=None,
            numpy_errors='raise',
            parsimony=0.0,
        )

        x = torch.linspace(-1.0, 1.0, steps=8).unsqueeze(-1)
        y = 2.5 * x

        with torch.no_grad():
            flash.fit(x, y, complexity=5)
    finally:
        model.forward = restore_forward

    assert flash._generated_numeric_sequences, "fit should cache numeric sequences when decoding succeeds"

    beam_sequence = flash._results[0]['raw_beam']
    constant_idx = beam_sequence.index(tokenizer['<constant>'])
    generated_numeric = flash._generated_numeric_sequences[0][constant_idx]
    assert math.isfinite(generated_numeric)

    assert not flash.results.empty
    numeric_column = flash.results['numeric_prediction']
    finite_predictions = [vals for vals in numeric_column if isinstance(vals, np.ndarray) and np.isfinite(vals).any()]
    assert finite_predictions, "expected at least one finite numeric prediction in the results"


def test_mdn_numeric_head_samples_near_configured_mean() -> None:
    model = _build_model('mdn')
    _configure_mdn_head(model, mean=1.5, sigma=1e-3, favored_component=1)

    batch_size = 512
    seq_len = 1
    decoder_dim = model.next_token_head[0].in_features
    dtype = model.next_token_head[0].weight.dtype
    decoder_output = torch.zeros((batch_size, seq_len, decoder_dim), dtype=dtype)

    log_pi, mu, sigma = model.constant_value_head(decoder_output)
    prediction = NumericHeadPrediction(kind='mdn', log_pi=log_pi, mu=mu, sigma=sigma)

    torch.manual_seed(0)
    samples = prediction.sample()
    assert samples.shape == (batch_size, seq_len)

    sample_mean = float(samples.mean().item())
    sample_std = float(samples.std(unbiased=False).item())

    assert sample_mean == pytest.approx(1.5, abs=1e-3)
    assert sample_std == pytest.approx(1e-3, rel=0.1)


def test_mdn_numeric_loss_prefers_matching_targets() -> None:
    model = _build_model('mdn')
    target_mean = 0.75
    _configure_mdn_head(model, mean=target_mean, sigma=5e-3, favored_component=0)

    seq_len = 3
    decoder_dim = model.next_token_head[0].in_features
    dtype = model.next_token_head[0].weight.dtype
    decoder_output = torch.zeros((1, seq_len, decoder_dim), dtype=dtype)

    log_pi, mu, sigma = model.constant_value_head(decoder_output)
    prediction = NumericHeadPrediction(kind='mdn', log_pi=log_pi, mu=mu, sigma=sigma)

    numeric_targets = torch.tensor([[0.0, target_mean, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[False, True, False]], dtype=torch.bool)

    loss_match, _ = model.compute_numeric_loss(prediction, numeric_targets, mask)
    assert loss_match is not None

    mismatch_targets = numeric_targets.clone()
    mismatch_targets[0, 1] = target_mean + 1.0
    loss_mismatch, _ = model.compute_numeric_loss(prediction, mismatch_targets, mask)
    assert loss_mismatch is not None

    assert float(loss_match.item()) < float(loss_mismatch.item())


def test_mdn_numeric_loss_returns_none_when_mask_empty() -> None:
    model = _build_model('mdn')
    _configure_mdn_head(model, mean=0.0, sigma=1e-3, favored_component=0)

    decoder_dim = model.next_token_head[0].in_features
    dtype = model.next_token_head[0].weight.dtype
    decoder_output = torch.zeros((1, 1, decoder_dim), dtype=dtype)

    log_pi, mu, sigma = model.constant_value_head(decoder_output)
    prediction = NumericHeadPrediction(kind='mdn', log_pi=log_pi, mu=mu, sigma=sigma)

    numeric_targets = torch.zeros((1, 1), dtype=torch.float32)
    mask = torch.zeros((1, 1), dtype=torch.bool)

    loss, per_element = model.compute_numeric_loss(prediction, numeric_targets, mask)
    assert loss is None and per_element is None


def _build_numeric_loss_inputs(model: FlashANSRModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
    tokenizer = model.tokenizer
    bos = tokenizer['<bos>']
    constant = tokenizer['<constant>']
    eos = tokenizer['<eos>']

    input_tokens = torch.tensor([[bos, constant, eos]], dtype=torch.long)
    numeric_targets = torch.tensor([[0.0, 1.25, 0.0]], dtype=torch.float32)
    numeric_tokens = numeric_targets.unsqueeze(-1)
    mask = (input_tokens == constant)
    data_tensor = torch.zeros((1, 2, model.encoder_max_n_variables), dtype=torch.float32)
    return input_tokens, numeric_tokens, numeric_targets, mask, data_tensor


def test_compute_numeric_loss_returns_none_without_numeric_head() -> None:
    model = _build_model('none')
    _input_tokens, _numeric_tokens, numeric_targets, mask, _data_tensor = _build_numeric_loss_inputs(model)
    loss, per_element = model.compute_numeric_loss(None, numeric_targets, mask)
    assert loss is None and per_element is None


def test_compute_numeric_loss_runs_for_deterministic_head() -> None:
    model = _build_model('deterministic')
    input_tokens, numeric_tokens, numeric_targets, mask, data_tensor = _build_numeric_loss_inputs(model)
    logits, numeric_pred = model.forward(input_tokens, data_tensor, input_num=numeric_tokens)
    assert isinstance(numeric_pred, NumericHeadPrediction)
    loss, per_element = model.compute_numeric_loss(numeric_pred, numeric_targets, mask)
    assert loss is not None and torch.isfinite(loss)
    assert per_element is not None and per_element.numel() == 1


def test_compute_numeric_loss_runs_for_mdn_head() -> None:
    model = _build_model('mdn')
    input_tokens, numeric_tokens, numeric_targets, mask, data_tensor = _build_numeric_loss_inputs(model)
    logits, numeric_pred = model.forward(input_tokens, data_tensor, input_num=numeric_tokens)
    assert isinstance(numeric_pred, NumericHeadPrediction)
    loss, per_element = model.compute_numeric_loss(numeric_pred, numeric_targets, mask)
    assert loss is not None and torch.isfinite(loss)
    assert per_element is not None and per_element.numel() == 1
