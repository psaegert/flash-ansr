import copy
import math
import os
import random
import warnings
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from simplipy import SimpliPyEngine

from flash_ansr.expressions import SkeletonPool, NoValidSampleFoundError
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.results import (
    RESULTS_FORMAT_VERSION,
    deserialize_results_payload,
    load_results_payload,
    save_results_payload,
    serialize_results_payload,
)
from flash_ansr.utils.paths import substitute_root_path


class SkeletonPoolModel(BaseEstimator):
    """Baseline model that samples skeletons from a pool and fits constants.

    This model is intended for research/ablation baselines that do **not** use
    the Flash-ANSR transformer. It samples expression skeletons from a provided
    ``SkeletonPool`` and refines their constants against user-provided data
    using the same `Refiner` used by `flash_ansr.flash_ansr.FlashANSR`.
    """

    FLOAT64_EPS: float = float(np.finfo(np.float64).eps)

    def __init__(
        self,
        *,
        simplipy_engine: SimpliPyEngine,
        skeleton_pool: str | dict[str, Any] | SkeletonPool,
        samples: int = 32,
        unique: bool = True,
        ignore_holdouts: bool = True,
        seed: int | None = None,
        n_restarts: int = 8,
        refiner_method: Literal[
            'curve_fit_lm',
            'minimize_bfgs',
            'minimize_lbfgsb',
            'minimize_neldermead',
            'minimize_powell',
            'least_squares_trf',
            'least_squares_dogbox',
        ] = 'curve_fit_lm',
        refiner_p0_noise: Literal['uniform', 'normal'] | None = 'normal',
        refiner_p0_noise_kwargs: dict | Literal['default'] | None = 'default',
        numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
        parsimony: float = 0.05,
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.samples = samples
        self.unique = unique
        self.ignore_holdouts = ignore_holdouts
        self.seed = seed
        self.n_restarts = n_restarts
        self.refiner_method = refiner_method
        self.refiner_p0_noise = refiner_p0_noise
        if refiner_p0_noise_kwargs == 'default':
            refiner_p0_noise_kwargs = {'low': -5, 'high': 5}
        self.refiner_p0_noise_kwargs = copy.deepcopy(refiner_p0_noise_kwargs) if refiner_p0_noise_kwargs is not None else None
        self.numpy_errors = numpy_errors
        self.parsimony = parsimony

        self._pool = self._ensure_pool(skeleton_pool)
        self._results: list[dict[str, Any]] = []
        self.results: pd.DataFrame = pd.DataFrame()
        self._input_dim: int | None = None

    @property
    def n_variables(self) -> int:
        return self._pool.n_variables

    @classmethod
    def _normalize_variance(cls, variance: float) -> float:
        if not np.isfinite(variance):
            return cls.FLOAT64_EPS
        return max(float(variance), cls.FLOAT64_EPS)

    @classmethod
    def _compute_fvu(cls, loss: float, sample_count: int, variance: float) -> float:
        if sample_count <= 1:
            return float(loss)
        return float(loss) / cls._normalize_variance(variance)

    @classmethod
    def _score_from_fvu(cls, fvu: float, complexity: int, parsimony: float) -> float:
        if not np.isfinite(fvu) or fvu <= 0:
            safe_fvu = cls.FLOAT64_EPS
        else:
            safe_fvu = max(float(fvu), cls.FLOAT64_EPS)
        return float(math.log10(safe_fvu) + parsimony * complexity)

    def _ensure_pool(self, skeleton_pool_ref: str | dict[str, Any] | SkeletonPool) -> SkeletonPool:
        if isinstance(skeleton_pool_ref, SkeletonPool):
            pool = skeleton_pool_ref
        elif isinstance(skeleton_pool_ref, str):
            resolved = substitute_root_path(skeleton_pool_ref)
            if os.path.isdir(resolved):
                _, pool = SkeletonPool.load(resolved)
            else:
                pool = SkeletonPool.from_config(resolved)
        elif isinstance(skeleton_pool_ref, dict):
            pool = SkeletonPool.from_config(copy.deepcopy(skeleton_pool_ref))
        else:
            raise TypeError("`skeleton_pool` must be a SkeletonPool, path string, or configuration dictionary.")

        if self.ignore_holdouts:
            pool.clear_holdouts()

        return pool

    def _truncate_input(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[-1]
        if n_features == self.n_variables:
            return X
        if n_features < self.n_variables:
            pad_width = self.n_variables - n_features
            pad = np.zeros((*X.shape[:-1], pad_width), dtype=X.dtype)
            return np.concatenate([X, pad], axis=-1)

        return X[..., : self.n_variables]

    def _sample_skeletons(self) -> list[tuple[str, ...]]:
        if self.samples <= 0:
            return []

        rng = random.Random()
        if self.seed is not None:
            rng.seed(int(self.seed))

        cached_skeletons = list(self._pool.skeletons)
        if cached_skeletons:
            unique_sampling = self.unique and self.samples <= len(cached_skeletons)
            if self.unique and not unique_sampling:
                warnings.warn(
                    "Requested unique samples exceeds pool size; sampling with replacement instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            if unique_sampling:
                return rng.sample(cached_skeletons, k=self.samples)
            return [rng.choice(cached_skeletons) for _ in range(self.samples)]

        selected: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        while len(selected) < self.samples:
            try:
                skeleton, _code, _constants = self._pool.sample_skeleton(new=True, decontaminate=not self.ignore_holdouts)
            except NoValidSampleFoundError as exc:
                if not selected:
                    raise RuntimeError("Unable to sample skeletons from the pool configuration.") from exc
                continue

            skeleton_tuple = tuple(skeleton)
            if self.unique and skeleton_tuple in seen:
                continue
            seen.add(skeleton_tuple)
            selected.append(skeleton_tuple)

        return selected

    def fit(self, X: np.ndarray | torch.Tensor | pd.DataFrame, y: np.ndarray | torch.Tensor | pd.DataFrame | Sequence[float], *, verbose: bool = False) -> "SkeletonPoolModel":
        if len(np.shape(y)) == 1:
            y = np.reshape(y, (-1, 1))

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        elif isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        elif isinstance(y, (pd.DataFrame, pd.Series)):
            y_np = y.values
        else:
            y_np = np.asarray(y)

        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        elif y_np.shape[-1] != 1:
            raise ValueError("The target data must have a single output dimension.")

        X_np = self._truncate_input(np.asarray(X_np))
        self._input_dim = X_np.shape[1]

        sample_count = y_np.shape[0]
        if sample_count <= 1:
            y_variance = float('nan')
        else:
            y_variance = float(np.var(y_np, axis=0, ddof=1).item())

        skeletons = self._sample_skeletons()
        if not skeletons:
            self._results = []
            self.results = pd.DataFrame()
            return self

        numpy_state = np.geterr()
        np.seterr(all=self.numpy_errors)

        results: list[dict[str, Any]] = []
        for skeleton in skeletons:
            expression_tokens = list(skeleton)

            try:
                refiner = Refiner(self.simplipy_engine, n_variables=self.n_variables).fit(
                    expression=expression_tokens,
                    X=X_np,
                    y=y_np,
                    n_restarts=self.n_restarts,
                    method=self.refiner_method,
                    p0=None,
                    p0_noise=self.refiner_p0_noise,
                    p0_noise_kwargs=copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                    converge_error='ignore',
                )
            except ConvergenceError:
                continue

            # Accept constant-free expressions even though Refiner.valid_fit stays False in that case.
            if len(refiner._all_constants_values) == 0:
                continue

            has_constants = len(refiner.constants_symbols) > 0
            valid_fit = refiner.valid_fit or not has_constants
            if not valid_fit:
                continue

            loss = float(refiner._all_constants_values[0][-1])
            if not np.isfinite(loss):
                continue

            fvu = self._compute_fvu(loss, sample_count, y_variance)
            if not np.isfinite(fvu):
                continue

            score = self._score_from_fvu(fvu, len(expression_tokens), self.parsimony)

            results.append({
                'log_prob': float('nan'),
                'fvu': fvu,
                'score': score,
                'expression': expression_tokens,
                'complexity': len(expression_tokens),
                'requested_complexity': None,
                'raw_beam': expression_tokens,
                'beam': expression_tokens,
                'raw_beam_decoded': ' '.join(expression_tokens),
                'function': refiner.expression_lambda,
                'refiner': refiner,
                'fits': copy.deepcopy(refiner._all_constants_values),
                'prompt_metadata': None,
            })

        np.seterr(**numpy_state)

        # Lower scores are better because they are log-scaled FVU plus parsimony.
        results.sort(key=lambda item: item['score'])

        self._results = results
        self.results = pd.DataFrame(results)
        return self

    def predict(self, X: np.ndarray | torch.Tensor | pd.DataFrame, nth_best: int = 0) -> np.ndarray:
        if not self._results:
            raise ValueError("The model has not been fitted yet. Please call `fit` first.")

        if nth_best >= len(self._results):
            raise IndexError(f"nth_best={nth_best} is out of range for {len(self._results)} results.")

        refiner = self._results[nth_best]['refiner']

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        elif isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        X_np = self._truncate_input(np.asarray(X_np))
        return refiner.predict(X_np)

    def get_expression(self, nth_best: int = 0, *, return_prefix: bool = False, precision: int = 2) -> list[str] | str:
        if not self._results:
            raise ValueError("The model has not been fitted yet. Please call `fit` first.")

        if nth_best >= len(self._results):
            raise IndexError(f"nth_best={nth_best} is out of range for {len(self._results)} results.")

        refiner = self._results[nth_best]['refiner']
        return refiner.transform(
            self._results[nth_best]['expression'],
            nth_best_constants=0,
            return_prefix=return_prefix,
            precision=precision,
        )

    def save_results(self, path: str) -> None:
        """Persist fitted results (without lambdas) for reuse."""

        if not self._results:
            raise ValueError("No results available to save. Run `fit` first.")

        input_dim = self._input_dim if self._input_dim is not None else self.n_variables
        metadata = {
            "format_version": RESULTS_FORMAT_VERSION,
            "parsimony": self.parsimony,
            "n_variables": self.n_variables,
            "input_dim": input_dim,
            "variable_mapping": None,
        }

        payload = serialize_results_payload(self._results, metadata=metadata)
        save_results_payload(payload, path)

    def load_results(self, path: str, *, rebuild_refiners: bool = True) -> None:
        """Load previously saved results and rebuild refiners if requested."""

        payload = load_results_payload(path)
        metadata = payload.get("metadata", {})

        version = int(payload.get("version", 0))
        if version != RESULTS_FORMAT_VERSION:
            warnings.warn(
                f"Results payload version {version} does not match expected {RESULTS_FORMAT_VERSION}; attempting to proceed anyway."
            )

        parsimony = float(metadata.get("parsimony", self.parsimony))
        self.parsimony = parsimony
        n_variables = int(metadata.get("n_variables", self.n_variables))
        input_dim = int(metadata.get("input_dim", n_variables))

        self._input_dim = input_dim

        restored = deserialize_results_payload(
            payload,
            simplipy_engine=self.simplipy_engine,
            n_variables=n_variables,
            input_dim=input_dim,
            rebuild_refiners=rebuild_refiners,
        )

        self._results = sorted(
            restored,
            key=lambda item: item.get("score", float("inf")) if not math.isnan(item.get("score", float("nan"))) else float("inf"),
        )
        self.results = pd.DataFrame(self._results)
