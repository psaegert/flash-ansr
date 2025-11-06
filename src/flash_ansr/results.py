"""Utilities for aggregating FlashANSR inference results."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from flash_ansr.refine import ConvergenceError


def compile_results_table(
    results: Iterable[dict[str, Any]],
    *,
    parsimony: float,
    score_from_fvu: Callable[[float, int, float], float],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Recompute scores and return a sorted result list plus dataframe."""
    result_list = list(results)
    if not result_list:
        raise ConvergenceError("The optimization did not converge for any beam")

    for result in result_list:
        if "score" not in result:
            continue
        fvu = result.get("fvu", np.nan)
        if np.isfinite(fvu):
            result["score"] = score_from_fvu(float(fvu), len(result.get("expression", [])), parsimony)
        else:
            result["score"] = np.nan

    sorted_results = sorted(
        result_list,
        key=lambda item: (
            item["score"] if not np.isnan(item["score"]) else float("inf"),
            np.isnan(item["score"]),
        ),
    )

    results_df = pd.DataFrame(sorted_results)
    results_df = results_df.explode("fits")
    results_df["beam_id"] = results_df.index
    results_df.reset_index(drop=True, inplace=True)

    fits_columns = pd.DataFrame(results_df["fits"].tolist(), columns=["fit_constants", "fit_covariances", "fit_loss"])
    results_df = pd.concat([results_df.drop(columns=["fits"]), fits_columns], axis=1)

    return sorted_results, results_df
