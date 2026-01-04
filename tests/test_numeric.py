import math

import pytest
import torch

from flash_ansr.utils.numeric import build_numeric_sequence, merge_numeric_sequence


class StubTokenizer:
    def __init__(self, extra_tokens: list[str] | None = None) -> None:
        vocab = ['<pad>', '<bos>', '<eos>', '<constant>']
        if extra_tokens:
            vocab.extend(extra_tokens)
        self._token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}

    def __getitem__(self, key: str | int) -> int | str:
        if isinstance(key, int):
            return self._id_to_token[key]
        return self._token_to_id[key]


@pytest.fixture(scope="module")
def tokenizer() -> StubTokenizer:
    return StubTokenizer(extra_tokens=[f'C_{index}' for index in range(3)])


def test_build_numeric_sequence_with_constants(tokenizer: StubTokenizer) -> None:
    sequence = [
        tokenizer['<bos>'],
        tokenizer['<constant>'],
        tokenizer['<eos>'],
        tokenizer['<constant>'],
    ]
    constants = torch.tensor([1.5, 2.5], dtype=torch.float32)

    numeric = build_numeric_sequence(tokenizer, sequence, constants)

    assert math.isnan(numeric[0])
    assert numeric[1] == pytest.approx(1.5)
    assert math.isnan(numeric[2])
    assert numeric[3] == pytest.approx(2.5)


def test_build_numeric_sequence_with_indexed_constants(tokenizer: StubTokenizer) -> None:
    sequence = [
        tokenizer['<bos>'],
        tokenizer['C_0'],
        tokenizer['C_1'],
        tokenizer['<eos>'],
    ]
    constants = torch.tensor([7.0, 11.0], dtype=torch.float32)

    numeric = build_numeric_sequence(tokenizer, sequence, constants)

    assert math.isnan(numeric[0])
    assert numeric[1] == pytest.approx(7.0)
    assert numeric[2] == pytest.approx(11.0)
    assert math.isnan(numeric[3])


def test_merge_numeric_sequence_merges_nan_gaps() -> None:
    existing = [float('nan'), 3.0, 4.0]
    computed = [float('nan'), 2.0, float('nan')]

    merged = merge_numeric_sequence(existing, computed)

    assert math.isnan(merged[0])
    assert merged[1] == pytest.approx(2.0)
    assert merged[2] == pytest.approx(4.0)


def test_merge_numeric_sequence_handles_tensors() -> None:
    existing = torch.tensor([float('nan'), 5.0], dtype=torch.float32)
    computed = [float('nan'), float('nan')]

    merged = merge_numeric_sequence(existing, computed)

    assert math.isnan(merged[0])
    assert merged[1] == pytest.approx(5.0)
