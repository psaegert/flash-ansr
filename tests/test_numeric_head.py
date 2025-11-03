import math
from typing import Sequence

import numpy as np
import pytest
import torch

from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.utils import GenerationConfig, get_path, load_config


def _build_numeric_model() -> FlashANSRModel:
    config = load_config(get_path('configs', 'test', 'model.yaml'))
    config['use_numeric_head'] = True
    model = FlashANSRModel.from_config(config)
    model.eval()
    return model


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

        return logits, numeric_pred

    original_forward = model.forward
    model.forward = fake_forward.__get__(model, FlashANSRModel)
    return original_forward


def test_forward_emits_numeric_predictions() -> None:
    model = _build_numeric_model()

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
    assert isinstance(numeric_pred, torch.Tensor)
    assert numeric_pred.shape == (batch_size, seq_len)


def test_beam_search_records_latest_numeric_sequences() -> None:
    model = _build_numeric_model()

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


def test_flash_ansr_fit_uses_numeric_predictions() -> None:
    model = _build_numeric_model()
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
