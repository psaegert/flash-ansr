"""Model adapter implementations for the evaluation engine."""
from __future__ import annotations

import time
import warnings
import functools
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING

import numpy as np
import simplipy
from flash_ansr.expressions.normalization import normalize_skeleton, normalize_expression
from sympy import lambdify

try:  # pragma: no cover - optional dependency
    from pysr import PySRRegressor  # type: ignore
    _HAVE_PYSR = True
except Exception:  # pragma: no cover - optional dependency
    PySRRegressor = Any  # type: ignore
    _HAVE_PYSR = False

try:  # pragma: no cover - optional dependency
    from nesymres.architectures.model import Model as _RuntimeNeSymResModel  # type: ignore
    _HAVE_NESYMRES = True
except Exception:  # pragma: no cover - optional dependency missing
    _RuntimeNeSymResModel = Any  # type: ignore
    _HAVE_NESYMRES = False

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from nesymres.architectures.model import Model as NesymresModel  # type: ignore
else:
    NesymresModel = Any

from flash_ansr.eval.core import EvaluationModelAdapter, EvaluationResult, EvaluationSample
from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.refine import ConvergenceError


class FlashANSRAdapter(EvaluationModelAdapter):
    """Wrap the :class:`FlashANSR` model with the evaluation adapter protocol."""

    def __init__(
        self,
        model: FlashANSR,
        *,
        device: str = "cpu",
        complexity: str | list[int | float] | int | float = "none",
        refiner_workers: int | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.complexity = complexity
        self.refiner_workers = refiner_workers

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        return self.model.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.model.to(self.device).eval()
        if self.refiner_workers is not None:
            self.model.refiner_workers = self.refiner_workers

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        record["parsimony"] = getattr(self.model, "parsimony", None)

        y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
        complexity_value = self._resolve_complexity(record)
        variable_names = record.get("variable_names")

        fit_time_start = time.time()
        try:
            fit_args: list[Any] = [sample.x_support, y_fit]
            if variable_names is not None:
                fit_args.append(variable_names)
            self.model.fit(*fit_args, complexity=complexity_value)
            fit_time = time.time() - fit_time_start
            record["fit_time"] = fit_time
            record["prediction_success"] = True
        except (ConvergenceError, OverflowError, TypeError, ValueError) as exc:
            warnings.warn(f"Error while fitting model: {exc}. Filling nan.")
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        if not getattr(self.model, "_results", None):
            warnings.warn("Model produced no results. Filling nan.")
            record["error"] = "Model produced no results."
            record["prediction_success"] = False
            return EvaluationResult(record)

        best_result = self.model._results[0]

        try:
            y_pred = self.model.predict(sample.x_support, nth_best_beam=0, nth_best_constants=0)
            if sample.x_validation.shape[0] > 0:
                y_pred_val = self.model.predict(sample.x_validation, nth_best_beam=0, nth_best_constants=0)
            else:
                y_pred_val = np.empty_like(sample.y_validation)
            record["y_pred"] = np.asarray(y_pred).copy()
            record["y_pred_val"] = np.asarray(y_pred_val).copy()
        except (ConvergenceError, ValueError) as exc:
            warnings.warn(f"Error while predicting: {exc}. Filling nan.")
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        predicted_expression = self.model.get_expression(
            nth_best_beam=0,
            nth_best_constants=0,
            map_variables=True,
        )
        predicted_prefix = self.model.get_expression(
            nth_best_beam=0,
            nth_best_constants=0,
            return_prefix=True,
            map_variables=False,
        )

        record["predicted_expression"] = predicted_expression
        # normalize prefix for expression (keep numeric literals) and skeleton (constants -> <constant>)
        record["predicted_expression_prefix"] = normalize_expression(predicted_prefix).copy()
        record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix).copy()
        record["predicted_constants"] = (
            best_result["fits"][0][0].tolist() if best_result.get("fits") else None
        )
        record["predicted_score"] = best_result.get("score")
        record["predicted_log_prob"] = best_result.get("log_prob")

        return EvaluationResult(record)

    # ------------------------------------------------------------------
    def _resolve_complexity(self, metadata: dict[str, Any]) -> int | float | None:
        mode = self.complexity
        if isinstance(mode, (int, float)):
            return mode
        if isinstance(mode, list):
            return mode[0] if mode else None
        if mode == "none":
            return None
        if mode == "ground_truth":
            return metadata.get("complexity")
        raise NotImplementedError(f"Unsupported complexity configuration: {mode}")


__all__ = ["FlashANSRAdapter", "PySRAdapter", "NeSymReSAdapter"]


class PySRAdapter(EvaluationModelAdapter):
    """Adapter that wraps a PySRRegressor for evaluation."""

    def __init__(
        self,
        *,
        timeout_in_seconds: int,
        niterations: int,
        use_mult_div_operators: bool,
        padding: bool,
        parsimony: float,
        simplipy_engine: Any,
    ) -> None:
        if not _HAVE_PYSR:  # pragma: no cover - import guard
            raise ImportError("PySR is not installed; please install pysr to use PySRAdapter")

        self.timeout_in_seconds = timeout_in_seconds
        self.niterations = niterations
        self.use_mult_div_operators = use_mult_div_operators
        self.padding = padding
        self.parsimony = parsimony
        self.simplipy_engine = simplipy_engine

        self._model: Optional[Any] = None

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self._model = _create_pysr_model(
            timeout_in_seconds=self.timeout_in_seconds,
            niterations=self.niterations,
            use_mult_div_operators=self.use_mult_div_operators,
        )

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        if self._model is None:
            raise RuntimeError("PySRAdapter.prepare must be called before evaluation")

        record = sample.clone_metadata()
        record["parsimony"] = self.parsimony

        X_support = sample.x_support.copy()
        X_val = sample.x_validation.copy()
        y_support = (sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support).copy()
        y_val = sample.y_validation.copy()

        used_variables: list[str] | None = None
        if not self.padding:
            mask, used_variables = _compute_variable_mask(record.get("variables"), record.get("skeleton"))
            if mask is not None:
                X_support = X_support[:, mask]
                X_val = X_val[:, mask] if X_val.size else X_val

        fit_time_start = time.time()
        try:
            self._model.fit(X_support, y_support.ravel(), variable_names=used_variables)
            record["fit_time"] = time.time() - fit_time_start
            record["prediction_success"] = True
        except Exception as exc:  # pragma: no cover - PySR exceptions vary
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        try:
            y_pred = self._model.predict(X_support).reshape(-1, 1)
            y_pred_val = self._model.predict(X_val).reshape(-1, 1) if X_val.size else np.empty_like(y_val)
        except Exception as exc:  # pragma: no cover - PySR exceptions vary
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        record["y_pred"] = y_pred.copy()
        record["y_pred_val"] = y_pred_val.copy()

        try:
            best = self._model.get_best()
            predicted_expression = str(best["equation"])
            record["predicted_expression"] = predicted_expression
            predicted_prefix = self.simplipy_engine.infix_to_prefix(predicted_expression)
            record["predicted_expression_prefix"] = normalize_expression(predicted_prefix).copy()
            record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix).copy()
        except Exception as exc:  # pragma: no cover - defensive
            record["error"] = f"Failed to parse PySR expression: {exc}"
            record["prediction_success"] = False

        return EvaluationResult(record)


class NeSymReSAdapter(EvaluationModelAdapter):
    """Adapter for NeSymReS models using the generic evaluation engine."""

    def __init__(
        self,
        model: NesymresModel,
        fitfunc: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
        simplipy_engine: Any,
        *,
        device: str = "cpu",
        beam_width: int | None = None,
    ) -> None:
        if not _HAVE_NESYMRES:  # pragma: no cover - defensive guard
            raise ImportError("The 'nesymres' package is required for NeSymReSAdapter")
        self.model = model
        self.fitfunc = fitfunc
        self.simplipy_engine = simplipy_engine
        self.device = device
        self.beam_width = beam_width

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        return self.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.model.to(self.device).eval()
        if self.beam_width is not None:
            cfg_params = _extract_cfg_params(self.fitfunc)
            if cfg_params is not None and hasattr(cfg_params, "beam_size"):
                cfg_params.beam_size = self.beam_width

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        record["parsimony"] = getattr(self.model, "parsimony", None)

        X_support = sample.x_support
        y_fit = (sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support).reshape(-1)

        fit_time_start = time.time()
        try:
            nesymres_output = self.fitfunc(X_support, y_fit)
            record["fit_time"] = time.time() - fit_time_start
            record["prediction_success"] = True
        except Exception as exc:  # pragma: no cover - upstream exceptions vary
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        predicted_expr = nesymres_output.get("best_bfgs_preds", [None])[0]
        if predicted_expr is None:
            record["error"] = "NeSymReS returned no expression"
            record["prediction_success"] = False
            return EvaluationResult(record)

        try:
            predicted_expression = str(predicted_expr)
            record["predicted_expression"] = predicted_expression
            predicted_prefix = self.simplipy_engine.infix_to_prefix(predicted_expression)
            record["predicted_expression_prefix"] = normalize_expression(predicted_prefix)
            record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix)
        except Exception as exc:  # pragma: no cover - parse errors
            record["error"] = f"Failed to parse NeSymReS expression: {exc}"
            record["prediction_success"] = False
            return EvaluationResult(record)

        predicted_constants = nesymres_output.get("best_bfgs_consts")
        if predicted_constants is not None:
            record["predicted_constants"] = _convert_constants(predicted_constants)

        try:
            y_pred, y_pred_val = _evaluate_symbolic_expression(
                predicted_expr,
                sample.x_support,
                sample.x_validation,
            )
            record["y_pred"] = y_pred
            record["y_pred_val"] = y_pred_val
        except Exception as exc:  # pragma: no cover - evaluation errors
            record["error"] = f"Failed to evaluate NeSymReS expression: {exc}"
            record["prediction_success"] = False

        return EvaluationResult(record)


# ---------------------------------------------------------------------------
# Helper utilities

def _create_pysr_model(
    *,
    timeout_in_seconds: int,
    niterations: int,
    use_mult_div_operators: bool,
) -> Any:
    additional_unary_operators: list[str]
    additional_extra_sympy_mappings: dict[str, Any]
    if use_mult_div_operators:
        additional_unary_operators = [
            "mult2(x) = 2*x",
            "mult3(x) = 3*x",
            "mult4(x) = 4*x",
            "mult5(x) = 5*x",
            "div2(x) = x/2",
            "div3(x) = x/3",
            "div4(x) = x/4",
            "div5(x) = x/5",
        ]
        additional_extra_sympy_mappings = {
            "mult2": simplipy.operators.mult2,
            "mult3": simplipy.operators.mult3,
            "mult4": simplipy.operators.mult4,
            "mult5": simplipy.operators.mult5,
            "div2": simplipy.operators.div2,
            "div3": simplipy.operators.div3,
            "div4": simplipy.operators.div4,
            "div5": simplipy.operators.div5,
        }
    else:
        additional_unary_operators = []
        additional_extra_sympy_mappings = {}

    return PySRRegressor(
        temp_equation_file=True,
        delete_tempfiles=True,
        timeout_in_seconds=timeout_in_seconds,
        niterations=niterations,
        unary_operators=[
            "neg",
            "abs",
            "inv",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "exp",
            "log",
            "pow2(x) = x^2",
            "pow3(x) = x^3",
            "pow4(x) = x^4",
            "pow5(x) = x^5",
            r"pow1_2(x::T) where {T} = x >= 0 ? T(x^(1/2)) : T(NaN)",
            r"pow1_3(x::T) where {T} = x >= 0 ? T(x^(1/3)) : T(-((-x)^(1/3)))",
            r"pow1_4(x::T) where {T} = x >= 0 ? T(x^(1/4)) : T(NaN)",
            r"pow1_5(x::T) where {T} = x >= 0 ? T(x^(1/5)) : T(-((-x)^(1/5)))",
        ]
        + additional_unary_operators,
        binary_operators=["+", "-", "*", "/", "^"],
        extra_sympy_mappings={
            "pow2": simplipy.operators.pow2,
            "pow3": simplipy.operators.pow3,
            "pow4": simplipy.operators.pow4,
            "pow5": simplipy.operators.pow5,
            "pow1_2": simplipy.operators.pow1_2,
            "pow1_3": lambda x: x ** (1 / 3),
            "pow1_4": simplipy.operators.pow1_4,
            "pow1_5": lambda x: x ** (1 / 5),
        }
        | additional_extra_sympy_mappings,
        constraints={
            "^": (-1, 3),
        },
    )


def _compute_variable_mask(
    variables: Iterable[str] | None,
    skeleton_tokens: Iterable[str] | None,
) -> tuple[np.ndarray | None, list[str] | None]:
    if not variables or not skeleton_tokens:
        return None, None
    skeleton_set = set(skeleton_tokens)
    mask = []
    kept = []
    for var in variables:
        keep = var in skeleton_set
        mask.append(keep)
        if keep:
            kept.append(var)
    if not any(mask):
        return None, None
    return np.array(mask, dtype=bool), kept


def _extract_cfg_params(fitfunc: Any) -> Any:
    if hasattr(fitfunc, "cfg_params"):
        return fitfunc.cfg_params
    if isinstance(fitfunc, functools.partial):  # type: ignore[name-defined]
        keywords = fitfunc.keywords or {}
        return keywords.get("cfg_params")
    return None


def _convert_constants(constants: Any) -> list[float] | Any:
    if isinstance(constants, np.ndarray):
        return constants.tolist()
    if isinstance(constants, (list, tuple)):
        return list(constants)
    return constants


def _evaluate_symbolic_expression(predicted_expr: Any, X_support: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    var_symbols = [f"x_{idx + 1}" for idx in range(X_support.shape[1])]
    evaluate_expression = lambdify(var_symbols, predicted_expr, "numpy")
    y_pred = np.asarray(evaluate_expression(*X_support.T), dtype=float).reshape(-1, 1)
    if X_val.size > 0:
        y_pred_val = np.asarray(evaluate_expression(*X_val.T), dtype=float).reshape(-1, 1)
    else:
        y_pred_val = np.empty((0, 1), dtype=float)
    return y_pred, y_pred_val
