import copy
import math
import os
from collections import defaultdict
from typing import Any, Generator, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from simplipy import SimpliPyEngine
from simplipy.utils import construct_expressions

from flash_ansr.expressions import SkeletonPool
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.utils.paths import substitute_root_path


class BruteForceModel(BaseEstimator):
    """Exhaustive baseline that enumerates expressions in increasing length.

    Expressions are generated shortest-first using ``simplipy.utils.construct_expressions``
    over the operator and variable vocabulary defined by the provided
    ``SkeletonPool``. Each candidate is refined with the shared ``Refiner`` to
    fit constants against user-supplied data.
    """

    FLOAT64_EPS: float = float(np.finfo(np.float64).eps)

    def __init__(
        self,
        *,
        simplipy_engine: SimpliPyEngine,
        skeleton_pool: str | dict[str, Any] | SkeletonPool,
        max_expressions: int = 10_000,
        max_length: int | None = None,
        include_constant_token: bool = True,
        ignore_holdouts: bool = True,
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
        self.max_expressions = int(max_expressions)
        self.max_length = max_length
        self.include_constant_token = include_constant_token
        self.ignore_holdouts = ignore_holdouts
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

    @staticmethod
    def _normalize_variance(variance: float) -> float:
        if not np.isfinite(variance):
            return BruteForceModel.FLOAT64_EPS
        return max(float(variance), BruteForceModel.FLOAT64_EPS)

    @staticmethod
    def _compute_fvu(loss: float, sample_count: int, variance: float) -> float:
        if sample_count <= 1:
            return float(loss)
        return float(loss) / BruteForceModel._normalize_variance(variance)

    @staticmethod
    def _score_from_fvu(fvu: float, complexity: int, parsimony: float) -> float:
        if not np.isfinite(fvu) or fvu <= 0:
            safe_fvu = BruteForceModel.FLOAT64_EPS
        else:
            safe_fvu = max(float(fvu), BruteForceModel.FLOAT64_EPS)
        return float(math.log10(safe_fvu) + parsimony * complexity)

    def _leaf_nodes(self) -> list[str]:
        leaves = list(self._pool.variables)
        if self.include_constant_token:
            leaves.append('<constant>')
        return leaves

    def _non_leaf_nodes(self) -> dict[str, int]:
        operator_weights = self._pool.operator_weights or {}
        return {op: arity for op, arity in self.simplipy_engine.operator_arity.items() if operator_weights.get(op, 0) > 0}

    def _expression_generator(self) -> Generator[tuple[str, ...], None, None]:
        hashes_by_size: defaultdict[int, set[tuple[str, ...]]] = defaultdict(set)
        seen: set[tuple[str, ...]] = set()

        for leaf in self._leaf_nodes():
            expr = (leaf,)
            hashes_by_size[1].add(expr)
            seen.add(expr)
            yield expr
            if len(seen) >= self.max_expressions:
                return

        target_length = 2
        while len(seen) < self.max_expressions:
            new_expressions: list[tuple[str, ...]] = []
            for expr in construct_expressions(hashes_by_size, self._non_leaf_nodes(), must_have_sizes=None):
                expr_len = len(expr)
                if self.max_length is not None and expr_len > self.max_length:
                    continue
                if expr_len != target_length:
                    continue
                if expr in seen:
                    continue
                if not self.simplipy_engine.is_valid(list(expr)):
                    continue

                seen.add(expr)
                new_expressions.append(expr)
                yield expr
                if len(seen) >= self.max_expressions:
                    break

            if not new_expressions:
                break

            hashes_by_size[target_length].update(new_expressions)
            target_length += 1

    def fit(self, X: np.ndarray | torch.Tensor | pd.DataFrame, y: np.ndarray | torch.Tensor | pd.DataFrame | Sequence[float], *, verbose: bool = False) -> "BruteForceModel":
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

        numpy_state = np.geterr()
        np.seterr(all=self.numpy_errors)

        results: list[dict[str, Any]] = []
        for skeleton in self._expression_generator():
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

            if len(results) >= self.max_expressions:
                break

        np.seterr(**numpy_state)

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
