"""Utilities for aggregating and persisting FlashANSR inference results."""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from flash_ansr.refine import ConvergenceError, Refiner
from flash_ansr.utils.paths import substitute_root_path

RESULTS_FORMAT_VERSION = 1


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


def _serialize_fit(fit: tuple[Any, Any, float]) -> dict[str, Any]:
    constants, covariances, fit_loss = fit
    return {
        "constants": np.asarray(constants),
        "covariances": None if covariances is None else np.asarray(covariances),
        "loss": float(fit_loss),
    }


def _deserialize_fit(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray | None, float]:
    return (
        np.asarray(payload["constants"]),
        None if payload.get("covariances") is None else np.asarray(payload["covariances"]),
        float(payload["loss"]),
    )


def serialize_results_payload(
    results: Iterable[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strip unserializable objects (refiner/lambda) and package results for disk."""

    def _strip(entry: dict[str, Any]) -> dict[str, Any]:
        cleaned = {k: copy.deepcopy(v) for k, v in entry.items() if k not in {"refiner", "function"}}
        fits = entry.get("fits") or []
        cleaned["fits"] = [_serialize_fit(fit) for fit in fits]
        return cleaned

    payload = {
        "version": RESULTS_FORMAT_VERSION,
        "metadata": copy.deepcopy(metadata) if metadata is not None else {},
        "results": [_strip(result) for result in results],
    }

    return payload


def save_results_payload(payload: dict[str, Any], path: str | Path) -> None:
    resolved = Path(substitute_root_path(str(path)))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as handle:
        pickle.dump(payload, handle)


def load_results_payload(path: str | Path) -> dict[str, Any]:
    resolved = Path(substitute_root_path(str(path)))
    with resolved.open("rb") as handle:
        return pickle.load(handle)


def deserialize_results_payload(
    payload: dict[str, Any],
    *,
    simplipy_engine: Any,
    n_variables: int,
    input_dim: int,
    rebuild_refiners: bool = True,
) -> list[dict[str, Any]]:
    """Recreate result entries from a serialized payload."""

    restored: list[dict[str, Any]] = []
    for raw in payload.get("results", []):
        entry = {k: copy.deepcopy(v) for k, v in raw.items() if k != "fits"}
        fits = [_deserialize_fit(fit) for fit in raw.get("fits", [])]

        if rebuild_refiners:
            refiner = Refiner.from_serialized(
                simplipy_engine=simplipy_engine,
                n_variables=n_variables,
                expression=entry["expression"],
                n_inputs=input_dim,
                fits=fits,
            )
            entry["refiner"] = refiner
            entry["function"] = refiner.expression_lambda
            entry["fits"] = copy.deepcopy(refiner._all_constants_values)
        else:
            entry["refiner"] = None
            entry["function"] = None
            entry["fits"] = fits

        restored.append(entry)

    return restored
