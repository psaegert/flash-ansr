"""Model adapter implementations for the evaluation engine."""
from __future__ import annotations

import time
import warnings
import functools
import re
from contextlib import nullcontext
from typing import Any, Callable, Iterable, Mapping, Optional, TYPE_CHECKING

import numpy as np
import simplipy
from flash_ansr.baselines import BruteForceModel, SkeletonPoolModel
from flash_ansr.expressions.normalization import normalize_skeleton, normalize_expression
# sympy is imported lazily inside the two baseline adapters that use it (E2E, NeSymReS);
# it is an optional `[baselines]` extra, not a core runtime dependency.

from flash_ansr.eval.core import EvaluationModelAdapter, EvaluationResult, EvaluationSample
from flash_ansr.eval.candidate_store import CandidateStoreWriter, build_candidate_ledger
from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.refine import ConvergenceError

PySRRegressor: type[Any] | None  # pragma: no cover - assigned lazily
PySRRegressor = None

E2ERegressor: type[Any] | None  # pragma: no cover - assigned lazily
E2ERegressor = None

_torch_module: Any | None  # pragma: no cover - assigned lazily
_torch_module = None

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


class FlashANSRAdapter(EvaluationModelAdapter):
    """Wrap the `FlashANSR` model with the evaluation adapter protocol."""

    def __init__(
        self,
        model: FlashANSR,
        *,
        device: str = "cpu",
        complexity: str | list[int | float] | int | float = "none",
        refiner_workers: int | None = None,
        candidate_store_dir: str | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.complexity = complexity
        self.refiner_workers = refiner_workers
        # Save-all-candidates (thorough-tier quality shards only): when set, every problem's FULL
        # candidate ledger is streamed to a compact columnar store. Off (None) -> zero overhead, so
        # timing runs are untouched. The writer is created lazily on first capture. See STANDARD_EVAL.md
        # Section 7 + item 5.
        self.candidate_store_dir = candidate_store_dir
        self._candidate_store: Any | None = None

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        return self.model.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.model.to(self.device).eval()
        if self.refiner_workers is not None:
            self.model.refiner_workers = self.refiner_workers

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Serial fit + evaluate. Composes the two phases below; behaviour matches the pre-split path
        (the overlapped engine drives the phases separately across two threads).

        The ``np.errstate`` here restores the ``np.seterr(all=numpy_errors)`` policy that the old
        ``fit()`` applied around the whole generate+refine span. It is applied ONLY on this serial
        composition (single-threaded); the phases are left un-wrapped so the overlapped engine's two
        threads never clobber a process-global error policy across each other (refinement workers
        already set their own policy per job, and the deployed ``numpy_errors='ignore'`` is value-neutral
        vs the numpy default -- only warning emission differs)."""
        numpy_errors = getattr(self.model, "numpy_errors", None)
        with np.errstate(all=numpy_errors) if numpy_errors is not None else nullcontext():
            record, gen_state, fit_t0 = self.generate_phase(sample)
            return self.refine_extract_phase(sample, record, gen_state, fit_t0, wall_clock=True)

    def generate_phase(
        self,
        sample: EvaluationSample,
    ) -> tuple[dict[str, Any], Any | None, float]:
        """GPU generation phase: build the record and run ``model._fit_generate`` -> ``GenState``.

        Returns ``(record, gen_state, fit_t0)``. On a generation-phase error ``gen_state`` is ``None``
        and the error is recorded. Writes NOTHING to model instance state (the generate phase only
        touches ``self.model._prompt_prefix`` / ``_mcts_cache`` / ``_n_params``, none of which the
        commit path reads), so the overlapped engine can run this for problem N+1 on the GPU-owner
        thread while problem N is committed on the main thread.
        """
        record = sample.clone_metadata()
        record["length_penalty"] = getattr(self.model, "length_penalty", None)
        record["constants_penalty"] = getattr(self.model, "constants_penalty", None)
        record["likelihood_penalty"] = getattr(self.model, "likelihood_penalty", None)

        y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
        complexity_value = self._resolve_complexity(record)
        variable_names = record.get("variable_names")

        fit_t0 = time.time()
        try:
            if variable_names is not None:
                gen_state = self.model._fit_generate(sample.x_support, y_fit, variable_names, complexity=complexity_value)
            else:
                gen_state = self.model._fit_generate(sample.x_support, y_fit, complexity=complexity_value)
        except (ConvergenceError, OverflowError, TypeError, ValueError) as exc:
            record["error"] = str(exc)
            record["prediction_success"] = False
            return record, None, fit_t0
        return record, gen_state, fit_t0

    def refine_extract_phase(
        self,
        sample: EvaluationSample,
        record: dict[str, Any],
        gen_state: Any | None,
        fit_t0: float,
        *,
        wall_clock: bool = False,
        refine_seed: int | None = None,
    ) -> EvaluationResult:
        """CPU refinement + extraction phase: refine the GenState, commit it, predict and read out.

        MUST run on the commit thread in sample order: it commits the ``FitResult`` to model instance
        state via ``_apply_fit_result`` and then reads it back through ``predict`` / ``get_expression``.
        With ``prune_constant_budget == 0`` (every deployed eval config, and the overlap engine's hard
        gate) it touches no GPU. ``wall_clock`` selects the ``fit_time`` semantics: ``True`` (serial)
        records the wall-clock span since ``fit_t0`` (byte-identical to the pre-split ``fit()`` timing);
        ``False`` (overlapped, where wall-clock spans overlap and are meaningless per problem) records
        the well-defined component sum ``generation_time + refinement_time``. ``refine_seed`` (default
        ``None`` = fresh OS entropy, the deployed behaviour) pins the per-candidate p0-noise seeds so a
        quality-equivalence gate can drive serial and overlapped runs from identical candidates.
        """
        if gen_state is None:
            # Generation-phase error already recorded on `record`.
            return EvaluationResult(record)

        try:
            fit_result = self.model._fit_refine(gen_state, refine_seed=refine_seed)
        except (ConvergenceError, OverflowError, TypeError, ValueError) as exc:
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        self.model._apply_fit_result(fit_result)

        if self.candidate_store_dir is not None:
            self._capture_candidate_ledger(record, gen_state, fit_result)

        record["fit_time"] = (time.time() - fit_t0) if wall_clock else (fit_result.generation_time + fit_result.refinement_time)
        record["generation_time"] = fit_result.generation_time
        record["refinement_time"] = fit_result.refinement_time
        record["prediction_success"] = True

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
    def _capture_candidate_ledger(self, record: dict[str, Any], gen_state: Any, fit_result: Any) -> None:
        """Stream this problem's FULL candidate ledger to the compact columnar store (save-all tier).

        Joins the generation pool (``gen_state.raw_beams``/``log_probs``) with the refined results
        (``fit_result.results``) and writes one compressed .npz per problem, keyed by the resume-stable
        ``eval_row_index`` the data source stamped on the sample. Best-effort: any failure here warns and
        is swallowed -- candidate capture must never abort an eval row. Runs only on the consumer thread
        (single writer), so it is safe under the overlap engine too."""
        try:
            problem_id = record.get("eval_row_index")
            if problem_id is None:
                warnings.warn(
                    "candidate_store_dir is set but the sample carries no 'eval_row_index'; skipping "
                    "candidate capture for this problem (the data source must stamp it).",
                    RuntimeWarning,
                )
                return
            if self._candidate_store is None:
                self._candidate_store = CandidateStoreWriter(
                    self.candidate_store_dir, vocab_size=len(self.model.tokenizer)
                )
            if self._candidate_store.has_problem(int(problem_id)):
                return  # already written (resume)

            tokenizer = self.model.tokenizer

            def _decode_expr(raw_beam: list[int]) -> list[str]:
                expr_ids = tokenizer.extract_expression_from_beam(list(raw_beam))[0]
                return tokenizer.decode(expr_ids, special_tokens="<constant>")

            ledger = build_candidate_ledger(
                gen_state.raw_beams,
                gen_state.log_probs,
                fit_result.results,
                decode_expr=_decode_expr,
                is_valid=self.model.simplipy_engine.is_valid,
            )
            self._candidate_store.write_problem(int(problem_id), **ledger)
        except Exception as exc:  # noqa: BLE001 - capture is auxiliary; never break the eval row
            warnings.warn(f"Candidate-ledger capture failed for this problem: {exc}", RuntimeWarning)

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


def _evaluate_refiner_baseline(model: Any, sample: EvaluationSample) -> EvaluationResult:
    """Shared evaluation logic for refiner-backed baseline models."""

    record = sample.clone_metadata()
    record["length_penalty"] = getattr(model, "length_penalty", None)
    record["constants_penalty"] = getattr(model, "constants_penalty", None)
    record["likelihood_penalty"] = getattr(model, "likelihood_penalty", None)

    y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support

    fit_time_start = time.time()
    try:
        model.fit(sample.x_support, y_fit)
        record["fit_time"] = time.time() - fit_time_start
        record["prediction_success"] = True
    except Exception as exc:  # pragma: no cover - baseline errors vary
        record["error"] = str(exc)
        record["prediction_success"] = False
        return EvaluationResult(record)

    if not getattr(model, "_results", None):
        record["error"] = "Model produced no results."
        record["prediction_success"] = False
        return EvaluationResult(record)

    try:
        y_pred = model.predict(sample.x_support, nth_best=0)
        if sample.x_validation.size:
            y_pred_val = model.predict(sample.x_validation, nth_best=0)
        else:
            y_pred_val = np.empty_like(sample.y_validation)
        record["y_pred"] = np.asarray(y_pred).copy()
        record["y_pred_val"] = np.asarray(y_pred_val).copy()
    except Exception as exc:  # pragma: no cover - baseline errors vary
        record["error"] = str(exc)
        record["prediction_success"] = False
        return EvaluationResult(record)

    try:
        predicted_expression = model.get_expression(nth_best=0, return_prefix=False)
        predicted_prefix = model.get_expression(nth_best=0, return_prefix=True)
        record["predicted_expression"] = predicted_expression
        record["predicted_expression_prefix"] = normalize_expression(predicted_prefix).copy()
        record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix).copy()
    except Exception as exc:  # pragma: no cover - parse errors vary
        record["error"] = f"Failed to extract expression: {exc}"
        record["prediction_success"] = False
        return EvaluationResult(record)

    best_result = model._results[0]
    fits = best_result.get("fits")
    if fits:
        constants = np.asarray(fits[0][0]).tolist() if len(fits[0]) > 0 else None
        record["predicted_constants"] = constants
    record["predicted_score"] = best_result.get("score")
    record["predicted_log_prob"] = best_result.get("log_prob")

    return EvaluationResult(record)


class SkeletonPoolAdapter(EvaluationModelAdapter):
    """Adapter for the sampling-only `SkeletonPoolModel` baseline."""

    def __init__(self, model: SkeletonPoolModel) -> None:
        self.model = model

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        return self.model.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # noqa: ARG002
        return None

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        return _evaluate_refiner_baseline(self.model, sample)


class BruteForceAdapter(EvaluationModelAdapter):
    """Adapter for the exhaustive `BruteForceModel` baseline."""

    def __init__(self, model: BruteForceModel) -> None:
        self.model = model

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        return self.model.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # noqa: ARG002
        return None

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        return _evaluate_refiner_baseline(self.model, sample)


__all__ = [
    "FlashANSRAdapter",
    "E2EAdapter",
    "PySRAdapter",
    "NeSymReSAdapter",
    "SkeletonPoolAdapter",
    "BruteForceAdapter",
]


class PySRAdapter(EvaluationModelAdapter):
    """Adapter that wraps a PySRRegressor for evaluation."""

    def __init__(
        self,
        *,
        timeout_in_seconds: int,
        niterations: int,
        use_mult_div_operators: bool,
        padding: bool,
        simplipy_engine: Any,
    ) -> None:
        _require_pysr()  # import lazily to avoid initializing Julia unless needed

        self.timeout_in_seconds = timeout_in_seconds
        self.niterations = niterations
        self.use_mult_div_operators = use_mult_div_operators
        self.padding = padding
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


class E2EAdapter(EvaluationModelAdapter):
    """Adapter for the End-to-end symbolic regression (E2E) baseline."""

    def __init__(
        self,
        *,
        model_path: str,
        simplipy_engine: Any,
        device: str = "cpu",
        candidates_per_bag: int = 1,
        max_input_points: int = 200,
        max_number_bags: int = 10,
        n_trees_to_refine: int = 10,
        rescale: bool = True,
        max_generated_output_len: int = 200,
        debug: bool = False,
    ) -> None:
        self.model_path = model_path
        self.simplipy_engine = simplipy_engine
        self.device = device
        self.candidates_per_bag = candidates_per_bag
        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale
        self.max_generated_output_len = max_generated_output_len
        self.debug = debug

        self._estimator: Any | None = None

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        return self.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        torch_mod = _require_torch()
        Estimator = _require_e2e_regressor()

        # Allowlist E2E classes for safe deserialization on torch>=2.6.
        add_safe_globals = getattr(torch_mod.serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            try:  # pragma: no cover - depends on optional dependency
                from symbolicregression.model.model_wrapper import ModelWrapper  # type: ignore
                from symbolicregression.model.embedders import LinearPointEmbedder  # type: ignore

                add_safe_globals([ModelWrapper, LinearPointEmbedder])
            except Exception:
                pass

        try:
            model = torch_mod.load(self.model_path, map_location=torch_mod.device(self.device))
        except Exception as exc:  # pragma: no cover - defensive retry
            if "Weights only load failed" not in str(exc):
                raise
            model = torch_mod.load(
                self.model_path,
                map_location=torch_mod.device(self.device),
                weights_only=False,
            )
        try:
            model.to(self.device)
        except Exception:  # pragma: no cover - defensive guard
            pass

        if hasattr(model, "beam_size"):
            model.beam_size = self.candidates_per_bag
        elif hasattr(model, "module") and hasattr(model.module, "beam_size"):
            model.module.beam_size = self.candidates_per_bag

        # Allow overriding generation length to keep chunking stable for large beam sizes.
        if hasattr(model, "max_generated_output_len"):
            model.max_generated_output_len = self.max_generated_output_len
        elif hasattr(model, "module") and hasattr(model.module, "max_generated_output_len"):
            model.module.max_generated_output_len = self.max_generated_output_len

        self._estimator = Estimator(
            model=model,
            max_input_points=self.max_input_points,
            max_number_bags=self.max_number_bags,
            n_trees_to_refine=self.n_trees_to_refine,
            rescale=self.rescale,
        )

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        if self._estimator is None:
            raise RuntimeError("E2EAdapter.prepare must be called before evaluation")

        record = sample.clone_metadata()
        record["length_penalty"] = getattr(self._estimator, "length_penalty", None)
        record["constants_penalty"] = getattr(self._estimator, "constants_penalty", None)
        record["likelihood_penalty"] = getattr(self._estimator, "likelihood_penalty", None)

        X_support = sample.x_support.copy()
        X_val = sample.x_validation.copy()
        y_support = (sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support).copy()

        mask, used_variables = _compute_variable_mask(record.get("variables"), record.get("skeleton"))
        if mask is not None:
            X_support = X_support[:, mask]
            X_val = X_val[:, mask] if X_val.size else X_val
            if used_variables:
                record["variable_names"] = used_variables

        fit_time_start = time.time()
        try:
            self._estimator.fit(X_support, y_support, verbose=False)
            record["fit_time"] = time.time() - fit_time_start
            record["prediction_success"] = True
        except Exception as exc:  # pragma: no cover - upstream exceptions vary
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        try:
            y_pred = self._estimator.predict(X_support)
            y_pred_val = self._estimator.predict(X_val) if X_val.size else np.empty_like(sample.y_validation)
        except Exception as exc:  # pragma: no cover - defensive guard
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        if y_pred is None:
            record["error"] = "E2E returned no predictions"
            record["prediction_success"] = False
            return EvaluationResult(record)

        if y_pred_val is None:
            y_pred_val = np.empty_like(sample.y_validation)

        record["y_pred"] = np.asarray(y_pred).reshape(-1, 1)
        record["y_pred_val"] = np.asarray(y_pred_val).reshape(-1, 1)

        try:
            tree_info = self._estimator.retrieve_tree(with_infos=True)
            if isinstance(tree_info, list):
                tree_info = tree_info[0] if tree_info else None
            predicted_tree = None
            if isinstance(tree_info, Mapping):
                predicted_tree = tree_info.get("relabed_predicted_tree") or tree_info.get("predicted_tree")
            if predicted_tree is None:
                raise ValueError("E2E returned no tree")

            predicted_expression_raw = str(predicted_tree.infix())
            canonical_infix = _canonicalize_e2e_infix(predicted_expression_raw)
            try:
                import sympy as sp  # lazy: only the E2E baseline adapter needs sympy
                sympy_expr = sp.parse_expr(canonical_infix)
                predicted_expression = str(sympy_expr)
            except Exception:
                predicted_expression = canonical_infix

            # Normalize function casing for simplipy compatibility.
            predicted_expression = re.sub(r"\bAbs\b", "abs", predicted_expression)

            if self.debug:
                print("[E2EAdapter][debug] raw infix:", predicted_expression_raw, flush=True)
                print("[E2EAdapter][debug] canonical infix:", canonical_infix, flush=True)
                print("[E2EAdapter][debug] sympy infix:", predicted_expression, flush=True)

            record["predicted_expression"] = predicted_expression
            predicted_prefix = self.simplipy_engine.infix_to_prefix(predicted_expression)
            record["predicted_expression_prefix"] = normalize_expression(predicted_prefix).copy()
            record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix).copy()

            if self.debug:
                print("[E2EAdapter][debug] prefix:", predicted_prefix, flush=True)
                print("[E2EAdapter][debug] skeleton:", record["predicted_skeleton_prefix"], flush=True)
        except Exception as exc:  # pragma: no cover - parse errors vary
            record["error"] = f"Failed to parse E2E expression: {exc}"
            record["prediction_success"] = False
            return EvaluationResult(record)

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
        remove_padding: bool = True,
    ) -> None:
        if not _HAVE_NESYMRES:  # pragma: no cover - defensive guard
            raise ImportError("The 'nesymres' package is required for NeSymReSAdapter")
        self.model = model
        self.fitfunc = fitfunc
        self.simplipy_engine = simplipy_engine
        self.device = device
        self.beam_width = beam_width
        self.remove_padding = remove_padding
        self._fit_cfg_params: Any | None = None
        self._max_variables: int | None = None
        self._warned_feature_mismatch = False

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        return self.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.model.to(self.device).eval()
        cfg_params = _extract_cfg_params(self.fitfunc)
        self._fit_cfg_params = cfg_params
        if cfg_params is not None:
            total_vars = getattr(cfg_params, "total_variables", None)
            if isinstance(total_vars, Iterable):
                try:
                    self._max_variables = len(list(total_vars))
                except TypeError:  # pragma: no cover - defensive
                    self._max_variables = None
            if self.beam_width is not None and hasattr(cfg_params, "beam_size"):
                cfg_params.beam_size = self.beam_width

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        record["length_penalty"] = getattr(self.model, "length_penalty", None)
        record["constants_penalty"] = getattr(self.model, "constants_penalty", None)
        record["likelihood_penalty"] = getattr(self.model, "likelihood_penalty", None)

        X_support = sample.x_support.copy()
        X_validation = sample.x_validation.copy()

        if self.remove_padding:
            variables = record.get("variables") or record.get("variable_names")
            mask, used_variables = _compute_variable_mask(variables, record.get("skeleton"))
            if mask is not None:
                X_support = X_support[:, mask]
                X_validation = X_validation[:, mask]
                if used_variables:
                    record["variable_names"] = used_variables

        X_support = self._prepare_inputs(X_support)
        X_validation = self._prepare_inputs(X_validation)
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

        predicted_expr = _extract_first_prediction(
            nesymres_output,
            preferred_key="best_bfgs_preds",
            fallback_key="best_preds",
        )
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

        predicted_constants = _extract_first_prediction(
            nesymres_output,
            preferred_key="best_bfgs_consts",
            fallback_key="best_consts",
        )
        if predicted_constants is not None:
            record["predicted_constants"] = _convert_constants(predicted_constants)

        try:
            y_pred, y_pred_val = _evaluate_symbolic_expression(
                predicted_expr,
                X_support,
                X_validation,
            )
            record["y_pred"] = y_pred
            record["y_pred_val"] = y_pred_val

            support_targets = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
            support_fvu = _compute_fvu_from_predictions(support_targets, y_pred)
            record["support_fvu"] = support_fvu

            validation_targets = (
                sample.y_validation_noisy if sample.y_validation_noisy is not None else sample.y_validation
            )
            validation_fvu: float | None = None
            if validation_targets.size and y_pred_val.size:
                validation_fvu = _compute_fvu_from_predictions(validation_targets, y_pred_val)
                record["validation_fvu"] = validation_fvu

            _print_fvu_summary(support_fvu, validation_fvu)
        except Exception as exc:  # pragma: no cover - evaluation errors
            record["error"] = f"Failed to evaluate NeSymReS expression: {exc}"
            record["prediction_success"] = False

        return EvaluationResult(record)

    def _prepare_inputs(self, array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray) or self._max_variables is None:
            return array
        n_features = array.shape[1]
        if n_features == self._max_variables:
            return array
        if n_features > self._max_variables:
            if not self._warned_feature_mismatch:
                warnings.warn(
                    (
                        "NeSymReS checkpoint supports only %d variables; "
                        "truncating FastSRB inputs from %d features."
                    )
                    % (self._max_variables, n_features),
                    RuntimeWarning,
                )
                self._warned_feature_mismatch = True
            return array[:, : self._max_variables].copy()
        pad_width = self._max_variables - n_features
        pad = np.zeros((array.shape[0], pad_width), dtype=array.dtype)
        return np.concatenate([array, pad], axis=1)


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

    PySR = _require_pysr()

    return PySR(
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
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
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
    from sympy import lambdify  # lazy: only the NeSymReS baseline adapter needs sympy
    var_symbols = [f"x_{idx + 1}" for idx in range(X_support.shape[1])]
    evaluate_expression = lambdify(var_symbols, predicted_expr, "numpy")
    y_pred = np.asarray(evaluate_expression(*X_support.T), dtype=float).reshape(-1, 1)
    if X_val.size > 0:
        y_pred_val = np.asarray(evaluate_expression(*X_val.T), dtype=float).reshape(-1, 1)
    else:
        y_pred_val = np.empty((0, 1), dtype=float)
    return y_pred, y_pred_val


def _extract_first_prediction(
    output: Mapping[str, Any] | None,
    *,
    preferred_key: str,
    fallback_key: str | None = None,
) -> Any:
    if not isinstance(output, Mapping):
        return None
    candidate = _first_non_none(output.get(preferred_key))
    if candidate is not None:
        return candidate
    if fallback_key is not None:
        return _first_non_none(output.get(fallback_key))
    return None


def _first_non_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is not None:
                return item
        return None
    return value


def _require_pysr() -> type[Any]:
    global PySRRegressor
    if PySRRegressor is not None:
        return PySRRegressor
    try:  # pragma: no cover - optional dependency
        from pysr import PySRRegressor as _PySRRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("PySR is not installed; please install pysr to use PySRAdapter") from exc
    PySRRegressor = _PySRRegressor
    return PySRRegressor


def _require_torch() -> Any:
    global _torch_module
    if _torch_module is not None:
        return _torch_module
    try:  # pragma: no cover - optional dependency
        import torch as _torch  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("PyTorch is required for the E2E adapter") from exc
    _torch_module = _torch
    return _torch_module


def _require_e2e_regressor() -> type[Any]:
    global E2ERegressor
    if E2ERegressor is not None:
        return E2ERegressor
    try:  # pragma: no cover - optional dependency
        from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor as _E2ERegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "symbolicregression is not installed; install the E2E dependencies to use the E2E adapter",
        ) from exc
    E2ERegressor = _E2ERegressor
    return E2ERegressor


def _compute_fvu_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        return float("nan")
    loss = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    variance = float(np.var(y_true_arr))
    return FlashANSR._compute_fvu(loss, y_true_arr.size, variance)


def _print_fvu_summary(support_fvu: float, validation_fvu: float | None) -> None:
    support_str = _format_fvu_value(support_fvu)
    message = f"[NeSymReSAdapter] support FVU={support_str}"
    if validation_fvu is not None:
        message = f"{message} | validation FVU={_format_fvu_value(validation_fvu)}"
    print(message, flush=True)


def _format_fvu_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):  # pragma: no cover - defensive
        return "+inf"
    if np.isneginf(value):  # pragma: no cover - defensive
        return "-inf"
    return f"{value:.6g}"


def _canonicalize_e2e_infix(expr: str) -> str:
    """Map E2E operator tokens to standard infix symbols before SymPy parsing."""
    replacements = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "pow": "**",
        "inv": "1/",
    }
    # Tokenize on whitespace and parentheses to avoid touching identifiers like "mulx_0".
    spaced = re.sub(r"([()])", r" \1 ", expr)
    tokens = spaced.split()
    mapped = [replacements.get(tok, tok) for tok in tokens]
    return " ".join(mapped)
