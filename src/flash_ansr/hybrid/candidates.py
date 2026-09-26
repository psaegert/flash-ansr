"""PySR adds candidates; Flash-ANSR's ranking picks.

PySR's whole hall of fame joins the Flash-ANSR candidate pool, priced the way Flash-ANSR prices its
own candidates -- its fittable literals re-fitted with flash-ansr's Refiner (warm at PySR's values),
re-spelled by the constant ladder, the realized expression priced (flash-ansr's FVU on the fitted
target, the certified f64 default-canon MDL) and scored with the ranking's own ``score_row`` -- and
Flash-ANSR's sorting (score, then the expression tokens) picks rank 0 of the extended pool.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from flash_ansr.flash_ansr import canonicalize_fitted, price_realized, respell_result
from flash_ansr.refine import DEFAULT_REFINE_SCOPE, Refiner, refinement_slots
from flash_ansr.scoring import compute_fvu, count_constants, score_row
from flash_ansr.spelling import ConstantLadderConfig
from simplipy.engine import Mode
from symbolic_data.token_ops import normalize_expression, normalize_skeleton

from flash_ansr.hybrid.bridge import evaluate_prefix, literal_value

__all__ = ["REFINE_DEFAULTS", "refine_settings", "score_key", "flash_candidate", "pysr_candidates",
           "rank_candidates", "pick_prediction", "prediction_fvu"]

#: the refinement settings a PySR candidate is fitted and re-spelled with when the caller gives none:
#: the doctrine arm's (8 restarts, LM, normal p0 noise of scale 5, the fittable scope, the ladder on)
REFINE_DEFAULTS: dict[str, Any] = {
    "n_restarts": 8, "method": "curve_fit_lm", "p0_noise": "normal", "p0_noise_kwargs": {"loc": 0.0, "scale": 5},
    "refine_scope": DEFAULT_REFINE_SCOPE, "constant_ladder": True,
}


def refine_settings(model: Any) -> dict[str, Any]:
    """The refinement settings of a loaded ``FlashANSR`` (what its own candidates are fitted and
    re-spelled with), for the PySR candidates that join its pool."""
    return {
        "n_restarts": int(getattr(model, "n_restarts", REFINE_DEFAULTS["n_restarts"])),
        "method": getattr(model, "refiner_method", REFINE_DEFAULTS["method"]),
        "p0_noise": getattr(model, "refiner_p0_noise", REFINE_DEFAULTS["p0_noise"]),
        "p0_noise_kwargs": getattr(model, "refiner_p0_noise_kwargs", REFINE_DEFAULTS["p0_noise_kwargs"]),
        "refine_scope": getattr(model, "refine_scope", REFINE_DEFAULTS["refine_scope"]),
        "constant_ladder": getattr(model, "constant_ladder", REFINE_DEFAULTS["constant_ladder"]),
    }


def score_key(candidate: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
    """Flash-ANSR's ranking order: the score, ties broken on the expression tokens; a missing or
    non-finite score sorts last."""
    score = candidate.get("score")
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = float("inf")
    if not math.isfinite(value):
        value = float("inf")
    expression = candidate.get("expression_prefix")
    if expression is None:
        expression = candidate.get("expression") or ()
    return (value, tuple(map(str, expression)))


def prediction_fvu(y: np.ndarray, y_pred: np.ndarray) -> float:
    """flash-ansr's FVU of a prediction curve on the fitted target: the mean squared residual over
    ddof-0 variance of the finite targets (what its own candidates are priced with)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    n = int(finite.sum())
    if n == 0 or y_pred.shape[0] != y.shape[0]:
        return float("inf")
    with np.errstate(all="ignore"):
        loss = float(np.mean((y[finite] - y_pred[finite]) ** 2))
    variance = float(np.var(y[finite])) if n > 1 else float("nan")
    return compute_fvu(loss, n, variance)


def flash_candidate(cand: Mapping[str, Any]) -> dict[str, Any]:
    """A Flash-ANSR candidate as a snapshot stores it (``best`` or a ``candidates`` row), as a pool
    entry: its stored score is the ranking's own, computed by the refine worker on the same criterion."""
    return {
        "source": "flash-ansr", "hof_index": -1,
        "expression_infix": str(cand.get("expression_infix")),
        "expression_prefix": list(cand["expression_prefix"]), "skeleton_prefix": list(cand["skeleton_prefix"]),
        "constants": cand.get("constants"), "fvu": cand.get("fvu"), "mdl": cand.get("mdl"), "score": cand.get("score"),
        "n_nodes": cand.get("n_nodes"), "log_prob": cand.get("log_prob"), "pareto_rank": cand.get("pareto_rank", -1),
        "y_pred": cand.get("y_pred"), "y_pred_val": cand.get("y_pred_val"),
    }


def _fit_pysr_candidate(prefix: list[str], X: np.ndarray, y: np.ndarray, *, engine: Any,
                        fit: Mapping[str, Any]) -> tuple[Any, list[str]] | None:
    """PySR's expression through flash-ansr's Refiner: its fittable literals become the slots
    (the same scope policy Flash-ANSR's own candidates get), warm-started at PySR's values with
    one restart, a cold fit at the doctrine's restarts when the warm start does not converge.
    ``None`` when the expression has no fittable literal or no fit converged."""
    try:
        slots = refinement_slots(list(prefix), engine, fit["refine_scope"])
    except Exception:  # noqa: BLE001 - a prefix the scope policy cannot walk (malformed): priced as spelled, or dropped
        return None
    if not slots:
        return None
    p0_values = [literal_value(prefix[i]) for i in slots]
    p0 = np.asarray(p0_values, dtype=float) if all(v is not None for v in p0_values) else None
    refiner = Refiner(simplipy_engine=engine, n_variables=int(X.shape[1]))
    try:
        if p0 is not None:
            refiner.fit(list(prefix), X, y, p0=p0, p0_noise=None, p0_noise_kwargs=None, n_restarts=1,
                        method=fit["method"], converge_error="ignore", refine_scope=fit["refine_scope"])
        if p0 is None or not refiner.valid_fit:
            refiner = Refiner(simplipy_engine=engine, n_variables=int(X.shape[1]))
            refiner.fit(list(prefix), X, y, p0=None, p0_noise=fit["p0_noise"], p0_noise_kwargs=fit["p0_noise_kwargs"],
                        n_restarts=int(fit["n_restarts"]), method=fit["method"], converge_error="ignore",
                        refine_scope=fit["refine_scope"])
    except Exception:  # noqa: BLE001 - a candidate the refiner cannot fit is priced as spelled
        return None
    if not refiner.valid_fit or not refiner.all_constants_values:
        return None
    abstracted = ["<constant>" if i in set(slots) else tok for i, tok in enumerate(prefix)]
    # the canonical member of the fitted candidate, as flash-ansr emits its own (a factor the fit made
    # cancel, a constant the fit made fold): the price is the class price either way
    refiner, abstracted, _changed, _same = canonicalize_fitted(
        engine, refiner, abstracted, X, n_variables=int(X.shape[1]), refine_scope=fit["refine_scope"])
    return refiner, list(abstracted)


def _row_from_prefix(prefix: list[str], *, engine: Any, names: Sequence[str], X: np.ndarray, Xv: np.ndarray,
                     fvu: float, mdl: float | None, score: float, spelling: str | None = None) -> dict[str, Any] | None:
    """A pool entry for a realized prefix (its curves from the engine's realizations); a strictly
    shorter canonical form of the prefix is the entry's spelling (the class price does not change)."""
    try:
        canonical = list(engine.simplify(list(prefix)))
        if len(canonical) < len(prefix) and engine.is_valid(canonical):
            prefix = canonical
    except Exception:  # noqa: BLE001 - priced and evaluated as spelled
        pass
    try:
        cols = evaluate_prefix(engine, list(prefix), list(names), X, Xv)
    except Exception:  # noqa: BLE001 - unevaluable: not a candidate
        return None
    y_pred = np.asarray(cols[0], dtype=float).reshape(-1)
    y_pred_val = np.asarray(cols[1], dtype=float).reshape(-1) if Xv.shape[0] else np.empty(0)
    constants = [v for v in (literal_value(tok) for tok in prefix) if v is not None]
    return {
        "expression_prefix": list(prefix), "skeleton_prefix": list(normalize_skeleton(prefix) or []),
        "constants": constants, "fvu": float(fvu), "mdl": mdl, "score": float(score), "n_nodes": len(prefix),
        "log_prob": None, "pareto_rank": -1, "y_pred": y_pred, "y_pred_val": y_pred_val, "spelling": spelling,
    }


def _fitted_rows(refiner: Any, abstracted: list[str], *, engine: Any, weights: Mapping[str, float], names: Sequence[str],
                 X: np.ndarray, Xv: np.ndarray, y: np.ndarray, y_variance: float, fit: Mapping[str, Any],
                 ladder: ConstantLadderConfig | None) -> list[dict[str, Any]]:
    """The fitted PySR candidate as a pool entry and, when the constant ladder finds a better
    spelling, its re-spelled variant -- through flash-ansr's own ladder pass (``respell_result``):
    a tie replaces the parent, a strict improvement stands beside it, as in Flash-ANSR's pool."""
    n = int(y.shape[0])
    fvu = compute_fvu(float(refiner.loss), n, y_variance)
    mdl = price_realized(engine, refiner, abstracted)
    constant_count = len(refiner.slot_indices)
    score = score_row({"fvu": fvu, "expression": abstracted, "constant_count": constant_count, "log_prob": None, "mdl": mdl}, weights)
    parent = {"fvu": float(fvu), "mdl": mdl, "score": float(score), "expression": list(abstracted), "constant_count": constant_count,
              "fits": list(refiner.all_constants_values), "valid_fit": True, "log_prob": None, "spelling": None, "respelled": None}
    realized = list(refiner.transform(list(abstracted), return_prefix=True))
    parent_row = _row_from_prefix(list(normalize_expression(realized) or realized), engine=engine, names=names, X=X, Xv=Xv,
                                  fvu=fvu, mdl=mdl, score=float(score))
    rows = [parent_row] if parent_row is not None else []
    if ladder is None or mdl is None:
        return rows
    payload = {"constant_ladder": ladder, "ranking_weights": dict(weights), "expression": list(abstracted), "log_prob": None,
               "y_variance": y_variance, "n_variables": int(X.shape[1]), "method": fit["method"],
               "n_restarts": int(fit["n_restarts"]), "p0_noise": fit["p0_noise"], "p0_noise_kwargs": fit["p0_noise_kwargs"]}
    try:
        child = respell_result(payload, engine, refiner, X, y, parent)
    except Exception:  # noqa: BLE001 - the ladder is best-effort, the fitted candidate stands
        child = None
    if child is None:
        return rows
    try:
        child_refiner = Refiner.from_serialized(simplipy_engine=engine, n_variables=int(X.shape[1]), expression=list(child["expression"]),
                                                n_inputs=int(X.shape[1]), fits=list(child["fits"]), refine_scope="placeholders")
        child_realized = list(child_refiner.transform(list(child["expression"]), return_prefix=True))
    except Exception:  # noqa: BLE001
        return rows
    child_row = _row_from_prefix(list(normalize_expression(child_realized) or child_realized), engine=engine, names=names, X=X, Xv=Xv,
                                 fvu=float(child["fvu"]), mdl=child["mdl"], score=float(child["score"]), spelling=child.get("spelling"))
    if child_row is None:
        return rows
    return [child_row] if child.get("replaces_parent") else rows + [child_row]


def pysr_candidates(
    equations: Sequence[Mapping[str, Any] | str] | None,
    *,
    engine: Any,
    weights: Mapping[str, float],
    x_support: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray | None,
    variables: Sequence[str],
    refine: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """PySR's hall of fame as Flash-ANSR candidates.

    Each equation (PySR's infix in the fit's variable names) goes through the engine's reader into the
    engine grammar with ``x1..xn`` variables (by column position) and then through what Flash-ANSR's
    own candidates go through: its fittable literals are re-fitted with flash-ansr's Refiner (warm at
    PySR's values), the constant ladder re-spells them, the realized expression is priced (fit as
    flash-ansr's FVU on the fitted target, MDL as the certified f64 default-canon price) and scored
    with the ranking's own ``score_row`` under ``weights`` (``RankingConfig.effective_weights``).
    ``refine`` carries the model's refinement settings (:func:`refine_settings`; the doctrine's when
    None). An expression without a fittable literal, or one the refiner cannot fit, is priced as
    spelled; one the engine cannot read or evaluate is dropped; an unpriceable one scores +inf under
    a live MDL weight and sorts last."""
    X = np.asarray(x_support, dtype=float)
    Xv = np.asarray(x_val, dtype=float) if x_val is not None and np.size(x_val) else np.empty((0, X.shape[1]))
    y = np.asarray(y_fit, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    # ddof=0 over the finite targets: flash-ansr's own definition, so the selection FVU equals the evaluation FVU
    y_variance = float(np.var(y[finite])) if int(finite.sum()) > 1 else float("nan")
    names = [f"x{i + 1}" for i in range(X.shape[1])]
    # `variables` names the columns of X in order. A list of another length cannot be placed and the
    # reader's own canonical names (`v3` -> `x3`) are trusted instead.
    rename = {str(v): names[i] for i, v in enumerate(variables)} if len(variables) == X.shape[1] else {}
    fit = dict(REFINE_DEFAULTS, **dict(refine or {}))
    ladder_setting = fit.get("constant_ladder")
    ladder = ladder_setting if isinstance(ladder_setting, ConstantLadderConfig) else ConstantLadderConfig.from_mapping(ladder_setting)
    out: list[dict[str, Any]] = []
    for k, entry in enumerate(equations or []):
        text = entry.get("equation") if isinstance(entry, Mapping) else entry
        if not text:
            continue
        try:
            raw = [rename.get(str(t), str(t)) for t in engine.read_infix(str(text))]
            prefix = list(normalize_expression(raw) or [])
        except Exception:  # noqa: BLE001 - an unreadable equation is not a candidate
            continue
        if not prefix:
            continue
        fitted = _fit_pysr_candidate(prefix, X, y, engine=engine, fit=fit)
        if fitted is not None:
            rows = _fitted_rows(fitted[0], fitted[1], engine=engine, weights=weights, names=names, X=X, Xv=Xv, y=y,
                                y_variance=y_variance, fit=fit, ladder=ladder)
        else:
            # no fittable literal (or no converged fit): the expression as PySR spelled it
            try:
                mdl: float | None = float(engine.complexity(prefix, certified=True, mode=Mode.f64, canon="default"))
            except Exception:  # noqa: BLE001 - unpriceable: the scorer decides (+inf under a live mdl weight)
                mdl = None
            if mdl is not None and not np.isfinite(mdl):
                mdl = None
            try:
                cols = evaluate_prefix(engine, prefix, names, X, Xv)
            except Exception:  # noqa: BLE001
                continue
            y_pred = np.asarray(cols[0], dtype=float).reshape(-1)
            if y_pred.shape[0] != y.shape[0]:
                continue
            fvu = prediction_fvu(y, y_pred)
            score = score_row({"fvu": fvu, "expression": prefix, "constant_count": count_constants(prefix), "log_prob": None, "mdl": mdl}, weights)
            row = _row_from_prefix(prefix, engine=engine, names=names, X=X, Xv=Xv, fvu=fvu, mdl=mdl, score=float(score))
            rows = [row] if row is not None else []
        for row in rows:
            row.update({"source": "pysr", "hof_index": k, "expression_infix": str(text),
                        "pysr_complexity": entry.get("complexity") if isinstance(entry, Mapping) else None,
                        "pysr_loss": entry.get("loss") if isinstance(entry, Mapping) else None})
            out.append(row)
    return out


def rank_candidates(flash: Sequence[Mapping[str, Any]], pysr: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The extended candidate pool -- Flash-ANSR's candidates plus PySR's -- in Flash-ANSR's ranking
    order: by score, ties broken on the expression tokens (flash-ansr's own deterministic tie-break).
    Rank 0 is the prediction."""
    entries: list[dict[str, Any]] = [flash_candidate(c) for c in flash]
    entries.extend(dict(p) for p in pysr)
    return sorted(entries, key=score_key)


def pick_prediction(
    values: Any,
    *,
    flash: Sequence[Mapping[str, Any]],
    equations: Sequence[Mapping[str, Any] | str] | None,
    engine: Any,
    weights: Mapping[str, float],
    x_support: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray | None,
    variables: Sequence[str],
    refine: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Flash-ANSR's sorting picks the prediction from the extended pool (in place); returns rank 0.

    ``values`` is a result record whose ``predicted_*`` fields hold PySR's own choice on entry; that
    choice stays under ``pysr_expression`` / ``pysr_expression_prefix`` for reference. The ranked
    pool (source, index, score, fvu, mdl, n_nodes) is stored as ``hybrid_candidates`` and
    ``predicted_source`` names the origin of rank 0. When nothing could be priced the record keeps
    PySR's answer and ``hybrid_ranking_error`` says so; a failed GP stage (no hall of fame) leaves the
    Flash-ANSR candidates to rank alone, PySR's error kept under ``pysr_error``."""
    values["pysr_expression"] = values.get("predicted_expression")
    values["pysr_expression_prefix"] = values.get("predicted_expression_prefix")
    added = pysr_candidates(equations, engine=engine, weights=weights, x_support=x_support, y_fit=y_fit,
                            x_val=x_val, variables=variables, refine=refine)
    ranked = rank_candidates(flash, added)
    values["hybrid_candidates"] = [{k: e.get(k) for k in ("source", "hof_index", "score", "fvu", "mdl", "n_nodes")} for e in ranked]
    if not ranked:
        values["predicted_source"] = "pysr"
        values["hybrid_ranking_error"] = "no candidate could be priced"
        return None
    best = ranked[0]
    if best.get("y_pred") is None or not np.size(best.get("y_pred")):
        # a Flash-ANSR candidate stored without its predictions (a snapshot's `candidates` row)
        names = [f"x{i + 1}" for i in range(np.asarray(x_support).shape[1])]
        try:
            cols = evaluate_prefix(engine, list(best["expression_prefix"]), names, np.asarray(x_support, dtype=float),
                                   np.asarray(x_val, dtype=float) if x_val is not None and np.size(x_val) else np.empty((0, len(names))))
            best["y_pred"] = np.asarray(cols[0], dtype=float).reshape(-1)
            best["y_pred_val"] = np.asarray(cols[1], dtype=float).reshape(-1)
        except Exception:  # noqa: BLE001 - the prediction stays, its curve is missing
            pass
    values["predicted_source"] = best["source"]
    values["predicted_hof_index"] = best["hof_index"]
    values["predicted_expression"] = best["expression_infix"]
    values["predicted_expression_prefix"] = list(best["expression_prefix"])
    values["predicted_skeleton_prefix"] = list(best["skeleton_prefix"])
    values["predicted_constants"] = best["constants"]
    values["predicted_score"] = best["score"]
    values["predicted_mdl"] = best["mdl"]
    values["predicted_n_nodes"] = best["n_nodes"]
    values["predicted_log_prob"] = best["log_prob"]
    values["predicted_pareto_rank"] = best["pareto_rank"]
    n_fit = int(np.asarray(y_fit).reshape(-1).shape[0])
    y_pred = best.get("y_pred")
    values["y_pred"] = (np.asarray(y_pred, dtype=float).reshape(-1, 1) if y_pred is not None and np.size(y_pred)
                        else np.full((n_fit, 1), np.nan))
    y_pred_val = best.get("y_pred_val")
    values["y_pred_val"] = (np.asarray(y_pred_val, dtype=float).reshape(-1, 1)
                            if y_pred_val is not None and np.size(y_pred_val) else np.empty((0, 1)))
    if values.get("error") and not values.get("prediction_success"):
        values["pysr_error"] = values.get("error")
        values["error"] = None
    values["prediction_success"] = True
    return best
