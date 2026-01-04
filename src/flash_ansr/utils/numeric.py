"""Helpers for constructing and merging numeric token channels."""
import math
import re
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from flash_ansr.model.tokenizer import Tokenizer

_CONSTANT_TOKEN_PATTERN = re.compile(r"C_\d+")


def _to_list(values: Any) -> list:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return list(values)
    return [values]


def _normalize_numeric(values: Sequence[Any]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        if value is None:
            normalized.append(float("nan"))
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            normalized.append(float("nan"))
            continue
        if math.isnan(numeric_value):
            normalized.append(float("nan"))
        else:
            normalized.append(numeric_value)
    return normalized


def build_numeric_sequence(
    tokenizer: "Tokenizer",
    input_ids: Sequence[int] | torch.Tensor,
    constants: Sequence[float] | torch.Tensor,
) -> list[float]:
    token_ids = _to_list(input_ids)
    const_values = [float(value) for value in _to_list(constants)]

    tokens = [tokenizer[int(token)] for token in token_ids]

    numeric_sequence: list[float] = []
    constant_index = 0

    for token in tokens:
        if token == "<pad>":
            numeric_sequence.append(float("nan"))
            continue

        if token == "<constant>" or _CONSTANT_TOKEN_PATTERN.match(token):
            if constant_index < len(const_values):
                numeric_sequence.append(const_values[constant_index])
                constant_index += 1
            else:
                numeric_sequence.append(float("nan"))
        else:
            numeric_sequence.append(float("nan"))

    return numeric_sequence


def build_numeric_sequences(
    tokenizer: "Tokenizer",
    input_ids: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor,
    constants: Sequence[Sequence[float] | torch.Tensor],
) -> list[list[float]]:
    input_iterable = _to_list(input_ids)
    constant_iterable = _to_list(constants)

    return [
        build_numeric_sequence(tokenizer, seq_ids, const_values)
        for seq_ids, const_values in zip(input_iterable, constant_iterable)
    ]


def merge_numeric_sequence(
    existing: Sequence[Any] | torch.Tensor | None,
    computed: Sequence[float],
) -> list[float]:
    result = [float(value) for value in computed]
    if existing is None:
        return result

    existing_list = _normalize_numeric(_to_list(existing))
    if len(existing_list) < len(result):
        existing_list.extend([float("nan")] * (len(result) - len(existing_list)))

    for index, value in enumerate(result):
        if math.isnan(value) and index < len(existing_list):
            result[index] = existing_list[index]

    return result
