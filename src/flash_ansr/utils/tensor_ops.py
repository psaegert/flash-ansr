"""Tensor helpers used across the project."""
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn


def pad_input_set(X: "np.ndarray | torch.Tensor | Any", length: int) -> "np.ndarray | torch.Tensor":
    """Right-pad ``X`` to ``length`` along the last dimension.

    A pandas DataFrame is converted to its values first. It used to fall through both branches and
    be returned UNCHANGED, so a DataFrame reached ``Refiner.predict``, where ``self.expression_lambda
    (*X.T, ...)`` unpacked the transposed frame's ROW LABELS as positional arguments -- surfacing as
    "TypeError: <lambda>() takes 19 positional arguments but 66 were given" from a public verb whose
    own signature documents DataFrame input. Anything else that is neither array nor tensor now
    raises here rather than silently passing through.
    """
    if hasattr(X, "to_numpy") and not isinstance(X, (np.ndarray, torch.Tensor)):
        X = X.to_numpy()
    if not isinstance(X, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"pad_input_set expects an ndarray, a Tensor or a DataFrame, got {type(X).__name__}.")

    pad_length = length - X.shape[-1]
    if pad_length > 0:
        if isinstance(X, torch.Tensor):
            X = nn.functional.pad(X, (0, pad_length, 0, 0), value=0)
        else:
            X = np.pad(X, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)  # type: ignore[arg-type]
    return X


def mask_unused_variable_columns(
    arrays: Iterable[np.ndarray],
    *,
    variables: Sequence[str] | None,
    skeleton_tokens: Sequence[str] | None,
    padding: str | None,
) -> None:
    """Zero unused variable columns when padding is configured for zero masking."""

    if padding != "zero":
        return

    if not variables:
        return

    variable_to_index = {var: idx for idx, var in enumerate(variables)}
    if not variable_to_index:
        return

    used_variables: set[str] = set()
    if skeleton_tokens:
        for token in skeleton_tokens:
            if token in variable_to_index:
                used_variables.add(token)

    if len(used_variables) == len(variable_to_index):
        return

    unused_indices = [variable_to_index[var] for var in variables if var not in used_variables]
    if not unused_indices:
        return

    for array in arrays:
        if not isinstance(array, np.ndarray):
            continue
        if array.ndim < 2 or array.shape[1] == 0:
            continue
        for idx in unused_indices:
            if idx < array.shape[1]:
                array[:, idx] = 0
