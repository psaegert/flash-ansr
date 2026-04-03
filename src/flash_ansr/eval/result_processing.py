"""Helpers for loading, cleaning, and deriving metrics from evaluation results."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from editdistance import eval as edit_distance

from flash_ansr.eval.metrics.numeric import (
    fvu,
    is_perfect_fit,
    log10_fvu,
    safe_divide,
)
from flash_ansr.eval.metrics.symbolic import total_nestedness
from flash_ansr.eval.metrics.token_prediction import f1_score, precision, recall
from flash_ansr.eval.metrics.zss import zss_tree_edit_distance


# ── Default placeholder values for missing / failed predictions ───────────

DEFAULT_NEGATIVES: dict[str, Any] = {
    'constants': [],
    'error': None,
    'skeleton': [],
    'skeleton_simplified': [],
    'expression': [],
    'variables': [],
    'variable_names': [],
    'complexity': np.nan,
    'placeholder_reason': None,
    'benchmark_metadata': {},
    'ground_truth_infix': [],
    'ground_truth_prefix': [],
    'fit_time': np.inf,
    'input_ids': [],
    'labels': [],
    'labels_decoded': [],
    'n_support': np.nan,
    'parsimony': np.nan,
    'predicted_expression': [],
    'predicted_expression_prefix': [],
    'predicted_skeleton_prefix': [],
    'predicted_skeleton_prefix_raw': [],
    'predicted_constants': [],
    'predicted_score': -np.inf,
    'predicted_log_prob': -np.inf,
    'prediction_success': False,
    'skeleton_hash': [],
    'x': np.nan,
    'x_val': np.nan,
    'y': np.nan,
    'y_noisy': np.nan,
    'y_noisy_val': np.nan,
    'y_pred': np.nan,
    'y_pred_val': np.nan,
    'y_val': np.nan,

    'f1_score': 0.0,
    'skeleton_length': np.inf,
    'expression_length': np.inf,
    'predicted_skeleton_prefix_length': np.inf,
    'fvu_fit': np.inf,
    'fvu_val': np.inf,
    'log10_fvu_fit': np.inf,
    'log10_fvu_val': np.inf,
    'only_approx_fvu_fit': np.inf,
    'only_approx_fvu_val': np.inf,
    'only_approx_log10_fvu_fit': np.inf,
    'only_approx_log10_fvu_val': np.inf,
    'numeric_recovery_fit': 0.0,
    'numeric_recovery_val': 0.0,
    'n_variables': np.nan,
    'n_constants': np.inf,
    'predicted_n_constants': np.inf,
    'n_constants_delta': np.inf,
    'symbolic_recovery': 0.0,
    'skeleton_length_ratio': np.inf,
    'edit_distance': np.inf,
    'zss_edit_distance': np.inf,
    'unique_variables': np.nan,
    'predicted_unique_variables': 0,
    'f1_score_unique_variables': 0.0,
    'precision_unique_variables': 0.0,
    'recall_unique_variables': 0.0,
    'total_nestedness': np.nan,
    'predicted_total_nestedness': np.inf,
}


# ── Utility ───────────────────────────────────────────────────────────────

def parse_p_notation(value: Any) -> float:
    """Convert strings like ``'0p001'`` to floats (e.g. ``0.001``).

    Passes through numeric types unchanged.  Returns ``np.nan`` on failure.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported type for parse_p_notation: {type(value)}")

    cleaned = value.strip().replace('p', '.')
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _extract_var_index(variable_name: str) -> int:
    """Extract the numeric index from a variable name like ``x1`` or ``x_1``."""
    if variable_name.startswith('x_'):
        return int(variable_name[2:])
    if variable_name.startswith('x'):
        return int(variable_name[1:])
    raise ValueError(f"Unexpected variable name format: {variable_name}")


# ── None → default replacement ────────────────────────────────────────────

def fill_none_with_defaults(
    results: dict[str, Any],
    test_sets: Sequence[str] | None = None,
    defaults: dict[str, Any] | None = None,
) -> None:
    """Replace ``None`` entries in result arrays with default placeholder values.

    Operates **in-place** on ``results``.

    Parameters
    ----------
    results : dict
        Nested results dict: ``results[model]['results'][test_set][scaling_value][metric]``.
    test_sets : Sequence[str], optional
        Restrict processing to these test sets.  ``None`` means all.
    defaults : dict, optional
        Override :data:`DEFAULT_NEGATIVES` with a custom mapping.
    """
    if defaults is None:
        defaults = DEFAULT_NEGATIVES

    for model in results:
        for test_set in results[model].get('results', {}):
            if test_sets is not None and test_set not in test_sets:
                continue
            for scaling_value in results[model]['results'][test_set]:
                for metric in results[model]['results'][test_set][scaling_value]:
                    data = results[model]['results'][test_set][scaling_value][metric]
                    if not hasattr(data, '__len__') or len(data) == 0:
                        continue
                    try:
                        replaced = [
                            (defaults.get(metric, np.nan) if r is None else r)
                            for r in data
                        ]
                        if not isinstance(defaults.get(metric, np.nan), list):
                            replaced = np.array(replaced)
                        results[model]['results'][test_set][scaling_value][metric] = replaced
                    except ValueError:
                        pass  # setting an array element with a sequence


# ── Derived metric computation ────────────────────────────────────────────

def compute_derived_metrics(
    results: dict[str, Any],
    test_sets: Sequence[str],
    operator_arity: Mapping[str, int],
    simplify_fn: Callable[[list[str]], list[str] | None] | None = None,
) -> None:
    """Compute derived evaluation metrics in-place on *results*.

    This adds the following keys to each ``results[model]['results'][test_set][scaling_value]``
    dict:

    - ``fvu_fit``, ``fvu_val``, ``log10_fvu_fit``, ``log10_fvu_val``
    - ``numeric_recovery_fit``, ``numeric_recovery_val``
    - ``only_approx_fvu_*``, ``only_approx_log10_fvu_*``
    - ``skeleton_simplified``, ``f1_score``, ``skeleton_length``,
      ``predicted_skeleton_prefix_length``
    - ``n_variables``, ``n_constants``, ``predicted_n_constants``, ``n_constants_delta``
    - ``symbolic_recovery``, ``skeleton_length_ratio``
    - ``edit_distance``, ``zss_edit_distance``
    - ``unique_variables``, ``predicted_unique_variables``
    - ``f1_score_unique_variables``, ``precision_unique_variables``,
      ``recall_unique_variables``
    - ``total_nestedness``, ``predicted_total_nestedness``

    Parameters
    ----------
    results : dict
        Nested results dict.
    test_sets : Sequence[str]
        Test sets to process.
    operator_arity : Mapping[str, int]
        Map from operator name to arity (needed for nestedness &
        tree-edit-distance).
    simplify_fn : callable, optional
        Function to simplify a skeleton token list (e.g.
        ``engine.simplify``).  If ``None``, simplified skeletons are
        set to the raw skeletons.
    """
    for model in results:
        for test_set in test_sets:
            if test_set not in results[model].get('results', {}):
                continue
            for scaling_value in results[model]['results'][test_set]:
                r = results[model]['results'][test_set][scaling_value]

                # ── FVU / NRR for fit and val splits ──────────────
                for split, saved_split_name in [('fit', ''), ('val', '_val')]:
                    y_key = f'y{saved_split_name}'
                    yp_key = f'y_pred{saved_split_name}'
                    if y_key not in r or yp_key not in r:
                        continue
                    r[f'fvu_{split}'] = np.array([
                        fvu(yt, yp) for yt, yp in zip(r[y_key], r[yp_key])
                    ])
                    r[f'log10_fvu_{split}'] = np.array([
                        log10_fvu(yt, yp) for yt, yp in zip(r[y_key], r[yp_key])
                    ])
                    r[f'numeric_recovery_{split}'] = np.array([
                        is_perfect_fit(yt, yp) for yt, yp in zip(r[y_key], r[yp_key])
                    ])
                    r[f'only_approx_fvu_{split}'] = np.where(
                        r[f'numeric_recovery_{split}'], -np.inf, r[f'fvu_{split}'],
                    )
                    r[f'only_approx_log10_fvu_{split}'] = np.where(
                        r[f'numeric_recovery_{split}'], -np.inf, r[f'log10_fvu_{split}'],
                    )

                # ── Simplified skeletons ──────────────────────────
                if simplify_fn is not None:
                    r['skeleton_simplified'] = [
                        simplify_fn(sk) if sk is not None else None
                        for sk in r['skeleton']
                    ]
                else:
                    r['skeleton_simplified'] = list(r['skeleton'])

                skel_sim = r['skeleton_simplified']
                pred_skel = r['predicted_skeleton_prefix']

                # ── Token-level F1 ────────────────────────────────
                r['f1_score'] = np.array([
                    f1_score(np.array([ps]), np.array([sk])) if ps is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ])

                # ── Lengths ───────────────────────────────────────
                r['skeleton_length'] = np.array([
                    len(sk) if sk is not None else None for sk in skel_sim
                ])
                r['predicted_skeleton_prefix_length'] = np.array([
                    len(ps) if ps is not None else None for ps in pred_skel
                ])

                # ── Variable / constant counts ────────────────────
                r['n_variables'] = np.array([
                    len(set(t for t in sk if t.startswith('x'))) if sk is not None else None
                    for sk in skel_sim
                ])
                r['n_constants'] = np.array([
                    sk.count('<constant>') if sk is not None else None
                    for sk in skel_sim
                ])
                r['predicted_n_constants'] = np.array([
                    ps.count('<constant>') if ps is not None else None
                    for ps in pred_skel
                ])
                r['n_constants_delta'] = np.array([
                    pnc - nc if pnc is not None and nc is not None else None
                    for pnc, nc in zip(r['predicted_n_constants'], r['n_constants'])
                ])

                # ── Symbolic recovery ─────────────────────────────
                r['symbolic_recovery'] = np.array([
                    ps == sk if ps is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ])

                # ── Length ratio ──────────────────────────────────
                r['skeleton_length_ratio'] = np.array([
                    safe_divide(pl, tl) if pl is not None and tl is not None else None
                    for pl, tl in zip(
                        r['predicted_skeleton_prefix_length'], r['skeleton_length'],
                    )
                ])

                # ── Edit distances ────────────────────────────────
                r['edit_distance'] = np.array([
                    edit_distance(ps, sk)
                    if ps is not None and sk is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ])
                r['zss_edit_distance'] = np.array([
                    zss_tree_edit_distance(ps, sk, operator_arity)
                    if ps is not None and sk is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ])

                # ── Unique variables ──────────────────────────────
                r['unique_variables'] = [
                    sorted(
                        list(set(t for t in sk if t.startswith('x'))),
                        key=_extract_var_index,
                    ) if sk is not None else None
                    for sk in skel_sim
                ]
                r['predicted_unique_variables'] = [
                    sorted(
                        list(set(t for t in ps if t.startswith('x'))),
                        key=_extract_var_index,
                    ) if ps is not None else None
                    for ps in pred_skel
                ]

                # ── Variable-level F1 / precision / recall ────────
                r['f1_score_unique_variables'] = np.array([
                    f1_score([puv], [uv])
                    if puv is not None and uv is not None else None
                    for puv, uv in zip(
                        r['predicted_unique_variables'], r['unique_variables'],
                    )
                ])
                r['precision_unique_variables'] = np.array([
                    precision([puv], [uv])
                    if puv is not None and uv is not None else None
                    for puv, uv in zip(
                        r['predicted_unique_variables'], r['unique_variables'],
                    )
                ])
                r['recall_unique_variables'] = np.array([
                    recall([puv], [uv])
                    if puv is not None and uv is not None else None
                    for puv, uv in zip(
                        r['predicted_unique_variables'], r['unique_variables'],
                    )
                ])

                # ── Nestedness ────────────────────────────────────
                r['total_nestedness'] = np.array([
                    total_nestedness(sk, operator_arity) if sk is not None else None
                    for sk in skel_sim
                ])
                r['predicted_total_nestedness'] = np.array([
                    total_nestedness(ps, operator_arity) if ps is not None else None
                    for ps in pred_skel
                ])
