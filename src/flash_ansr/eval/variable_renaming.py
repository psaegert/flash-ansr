"""Variable renaming helpers for baseline model outputs.

Baseline models (PySR, NeSymReS, E2E) use different variable naming
conventions than the ground truth.  These helpers normalise predicted
skeleton tokens back to the ``x1, x2, ...`` convention used in the
ground-truth expressions.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable


def rename_variables_nesymres(
    skeleton: list[str] | None,
    original_variable_names: list[str] | None,
) -> list[str] | None:
    """Rename NeSymReS variables (``x_1, x_2, ...``) back to original names.

    NeSymReS strips padding and re-indexes variables starting at ``x_1``.
    This restores the original variable names from the ground-truth skeleton.
    """
    if skeleton is None or original_variable_names is None:
        return None
    renamed: list[str] = []
    for token in skeleton:
        if token.startswith('x_'):
            index = int(token[2:]) - 1  # 'x_1' -> 0
            if index < len(original_variable_names):
                renamed.append(original_variable_names[index])
            else:
                renamed.append(token)
        else:
            renamed.append(token)
    return renamed


def rename_variables_pysr(
    skeleton: list[str] | None,
    *args: Any,
    **kwargs: Any,
) -> list[str] | None:
    """Rename PySR variables from 0-indexed (``x0, x1, ...``) to 1-indexed (``x1, x2, ...``)."""
    if skeleton is None:
        return None
    renamed: list[str] = []
    for token in skeleton:
        if token.startswith('x'):
            index = int(token[1:]) + 1  # 'x0' -> 'x1'
            renamed.append(f'x{index}')
        else:
            renamed.append(token)
    return renamed


def rename_variables_e2e(
    skeleton: list[str] | None,
    *args: Any,
    **kwargs: Any,
) -> list[str] | None:
    """Rename E2E variables from ``x_0, x_1, ...`` to ``x1, x2, ...``."""
    if skeleton is None:
        return None
    renamed: list[str] = []
    for token in skeleton:
        if token.startswith('x_'):
            index = int(token[2:]) + 1  # 'x_0' -> 'x1'
            renamed.append(f'x{index}')
        else:
            renamed.append(token)
    return renamed


# Map from model name to its renaming function.
RENAME_FUNCTIONS: dict[str, Callable[..., list[str] | None]] = {
    'nesymres': rename_variables_nesymres,
    'pysr': rename_variables_pysr,
    'e2e': rename_variables_e2e,
}


def apply_variable_renaming(
    results: dict[str, Any],
    model_rename_map: dict[str, Callable[..., list[str] | None]] | None = None,
    test_sets: Sequence[str] | None = None,
) -> None:
    """Apply variable renaming in-place for baseline model results.

    Parameters
    ----------
    results : dict
        Nested results dict: ``results[model]['results'][test_set][scaling_value]``.
    model_rename_map : dict, optional
        Map from model name to renaming function.  Defaults to
        :data:`RENAME_FUNCTIONS`.
    test_sets : Sequence[str], optional
        Test sets to process.  If ``None``, processes all available test sets.
    """
    if model_rename_map is None:
        model_rename_map = RENAME_FUNCTIONS

    for model_name, rename_fn in model_rename_map.items():
        if model_name not in results:
            continue
        model_test_sets = test_sets or list(results[model_name].get('results', {}).keys())
        for test_set in model_test_sets:
            if test_set not in results[model_name].get('results', {}):
                continue
            for scaling_value in results[model_name]['results'][test_set]:
                r = results[model_name]['results'][test_set][scaling_value]

                unique_variables_in_ground_truth = [
                    sorted(
                        list(set(token for token in skeleton if token.startswith('x'))),
                        key=lambda x: int(x[1:]),
                    )
                    if skeleton is not None else None
                    for skeleton in r['skeleton']
                ]

                r['predicted_skeleton_prefix_raw'] = r['predicted_skeleton_prefix'].copy()
                r['predicted_skeleton_prefix'] = [
                    rename_fn(pred, var_names)
                    for pred, var_names in zip(
                        r['predicted_skeleton_prefix_raw'],
                        unique_variables_in_ground_truth,
                    )
                ]
