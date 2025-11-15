"""Utilities for sampling the FastSRB benchmark equations using SimpliPy."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from simplipy import SimpliPyEngine

from flash_ansr.expressions.compilation import codify


Number = Union[int, float]


class FastSRBBenchmark:
    """Sample datasets from the FastSRB benchmark YAML specification."""

    def __init__(
        self,
        yaml_path: Union[str, Path],
        *,
        simplipy_engine: SimpliPyEngine | str = "dev_7-3",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Load the YAML benchmark specification and prepare a SimpliPy engine."""

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as handle:
            entries = yaml.safe_load(handle)

        if not isinstance(entries, Mapping):
            raise ValueError("Benchmark specification must be a mapping from equation ids to entries.")

        self._entries: Dict[str, MutableMapping[str, Any]] = dict(entries)
        self._yaml_path = path
        self._rng = self._resolve_rng(random_state)

        self._simplipy_engine = simplipy_engine if isinstance(simplipy_engine, SimpliPyEngine) else SimpliPyEngine.load(simplipy_engine, install=True)
        self._compiled_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _resolve_rng(random_state: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
        if isinstance(random_state, np.random.Generator):
            return random_state
        return np.random.default_rng(random_state)

    def equation_ids(self) -> List[str]:
        """Return the identifiers of all benchmark equations."""
        return list(self._entries.keys())

    @staticmethod
    def _resolve_variable_order(vars_info: Mapping[str, Mapping[str, Any]]) -> List[str]:
        candidate_keys = [key for key in vars_info.keys() if key.startswith("v") and key != "v0"]
        if not candidate_keys:
            raise ValueError("Entry does not define any input variables")

        try:
            indices = sorted(int(key[1:]) for key in candidate_keys)
        except ValueError as exc:
            raise ValueError("Variable identifiers must follow the 'v<int>' pattern") from exc

        variable_order: List[str] = []
        for idx in range(1, indices[-1] + 1):
            key = f"v{idx}"
            if key not in vars_info:
                raise KeyError(f"Missing sampling specification for {key}")
            variable_order.append(key)
        return variable_order

    @staticmethod
    def _normalize_prepared_expression(expression: str) -> str:
        """Normalize prepared expressions so SimpliPy can parse them consistently."""

        return expression.replace("^", "**")

    def _compile_expression(self, eq_id: str, entry: Mapping[str, Any]) -> Dict[str, Any]:
        cache = self._compiled_cache.get(eq_id)
        if cache is not None:
            return cache

        prepared = entry.get("prepared")
        if not isinstance(prepared, str) or not prepared.strip():
            raise ValueError(f"Entry {eq_id} has no prepared expression")

        prepared_text = self._normalize_prepared_expression(prepared)

        vars_info = entry.get("vars")
        if not isinstance(vars_info, Mapping):
            raise ValueError(f"Entry {eq_id} has no variable definitions")

        variable_order = self._resolve_variable_order(vars_info)

        prefix_parsed = self._simplipy_engine.parse(prepared_text, mask_numbers=False)
        try:
            prefix_simplified = self._simplipy_engine.simplify(prefix_parsed, max_pattern_length=4)
        except Exception as exc:  # pragma: no cover - defensive against SimpliPy regressions
            warnings.warn(
                f"Failed to simplify FastSRB expression {eq_id}: {exc}. Falling back to unsimplified prefix.",
                RuntimeWarning,
            )
            prefix_simplified = prefix_parsed

        used_variables = {token for token in prefix_simplified if isinstance(token, str) and token.startswith("v")}
        unknown_variables = used_variables - set(variable_order) - {"v0"}
        if unknown_variables:
            unknown_str = ", ".join(sorted(unknown_variables))
            raise KeyError(f"Prepared expression for {eq_id} references undefined variables: {unknown_str}")

        prefix_realized = self._simplipy_engine.operators_to_realizations(prefix_parsed)
        code_string = self._simplipy_engine.prefix_to_infix(prefix_realized, realization=True)
        code = codify(code_string, variable_order)
        expression_callable = self._simplipy_engine.code_to_lambda(code)
        normalized_infix = self._simplipy_engine.prefix_to_infix(prefix_simplified, realization=False)

        cache = {
            "code": code,
            "callable": expression_callable,
            "variable_order": variable_order,
            "prefix": tuple(prefix_simplified),
            "normalized_infix": normalized_infix,
        }
        self._compiled_cache[eq_id] = cache
        return cache

    def _evaluate(self, compiled: Dict[str, Any], values: Mapping[str, Any]) -> Any:
        ordered_inputs = [values[name] for name in compiled["variable_order"]]
        with np.errstate(all="ignore"):
            return compiled["callable"](*ordered_inputs)

    def _sample_points(
        self,
        low: Number,
        high: Number,
        n_points: int,
        *,
        method: str,
        distribution: str,
        sign_mode: str,
        integer: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if method not in {"random", "range"}:
            raise ValueError("method must be 'random' or 'range'")
        if distribution not in {"uni", "log"}:
            raise ValueError("distribution must be 'uni' or 'log'")
        if sign_mode not in {"pos", "neg", "pos_neg"}:
            raise ValueError("sign_mode must be 'pos', 'neg', or 'pos_neg'")
        if n_points < 1:
            raise ValueError("n_points must be at least 1")
        if method == "range" and n_points == 1:
            warnings.warn("Sampling one point with method='range' is degenerate; consider method='random'", RuntimeWarning, stacklevel=2)
        low_f = float(low)
        high_f = float(high)
        if low_f > high_f:
            raise ValueError("sample_range lower bound must not exceed upper bound")
        if math.isclose(low_f, high_f):
            arr = np.full(n_points, high_f, dtype=float)
        else:
            if distribution == "log":
                if low_f <= 0 or high_f <= 0:
                    raise ValueError("log sampling requires strictly positive bounds")
                low_val = math.log10(low_f)
                high_val = math.log10(high_f)
            else:
                low_val = low_f
                high_val = high_f
            if method == "random":
                arr = rng.uniform(low_val, high_val, size=n_points)
            else:
                arr = np.linspace(low_val, high_val, n_points)
                rng.shuffle(arr)
            if distribution == "log":
                arr = 10.0 ** arr
        if sign_mode == "neg":
            arr = -np.abs(arr)
        elif sign_mode == "pos_neg":
            signs = rng.choice([-1.0, 1.0], size=arr.shape)
            arr = arr * signs
        if integer:
            arr = np.rint(arr)
        return arr.astype(float, copy=False)

    def _sample_matrix(
        self,
        vars_info: Mapping[str, Mapping[str, Any]],
        variable_order: Sequence[str],
        n_points: int,
        method: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        columns: List[np.ndarray] = []
        for key in variable_order:
            spec = vars_info.get(key)
            if spec is None:
                raise KeyError(f"Missing sampling spec for {key}")
            try:
                sample_type = spec["sample_type"]
                sample_range = spec["sample_range"]
            except KeyError as exc:
                raise KeyError(f"Variable {key} is missing required field {exc.args[0]}") from exc
            if not isinstance(sample_type, Sequence) or len(sample_type) < 2:
                raise ValueError(f"sample_type for {key} must have two entries")
            if not isinstance(sample_range, Sequence) or len(sample_range) < 2:
                raise ValueError(f"sample_range for {key} must provide at least lower and upper bounds")
            distribution = sample_type[0]
            sign_mode = sample_type[1]
            integer = False
            if distribution == "int":
                distribution = "uni"
                integer = True
            column = self._sample_points(
                sample_range[0],
                sample_range[1],
                n_points,
                method=method,
                distribution=distribution,
                sign_mode=sign_mode,
                integer=integer,
                rng=rng,
            )
            columns.append(column)
        matrix = np.column_stack(columns)
        return matrix.astype(float, copy=False)

    def _sample_single_point(
        self,
        vars_info: Mapping[str, Mapping[str, Any]],
        variable_order: Sequence[str],
        method: str,
        rng: np.random.Generator,
        compiled: Dict[str, Any],
        max_trials: int,
    ) -> np.ndarray:
        for _ in range(max_trials):
            values: List[float] = []
            value_map: Dict[str, float] = {}
            for key in variable_order:
                spec = vars_info.get(key)
                if spec is None:
                    raise KeyError(f"Missing sampling spec for {key}")
                sample_type = spec["sample_type"]
                sample_range = spec["sample_range"]
                if not isinstance(sample_type, Sequence) or len(sample_type) < 2:
                    raise ValueError(f"sample_type for {key} must have two entries")
                if not isinstance(sample_range, Sequence) or len(sample_range) < 2:
                    raise ValueError(f"sample_range for {key} must provide at least lower and upper bounds")
                distribution = sample_type[0]
                sign_mode = sample_type[1]
                integer = False
                if distribution == "int":
                    distribution = "uni"
                    integer = True
                sample_value = self._sample_points(
                    sample_range[0],
                    sample_range[1],
                    1,
                    method=method,
                    distribution=distribution,
                    sign_mode=sign_mode,
                    integer=integer,
                    rng=rng,
                )[0]
                values.append(float(sample_value))
                value_map[key] = float(sample_value)
            try:
                target = self._evaluate(compiled, value_map)
            except Exception:
                continue
            combined = np.array(values + [float(target)], dtype=float)
            if np.all(np.isfinite(combined)):
                return combined
        raise RuntimeError("Exceeded max_trials while sampling a single data point")

    def _sample_entry(
        self,
        eq_id: str,
        entry: Mapping[str, Any],
        n_points: int,
        method: str,
        max_trials: int,
        incremental: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        compiled = self._compile_expression(eq_id, entry)
        vars_info = entry.get("vars")
        if not isinstance(vars_info, Mapping):
            raise ValueError(f"Entry {eq_id} has no variable definitions")

        variable_order = compiled["variable_order"]
        matrix: Optional[np.ndarray] = None
        for _ in range(max_trials):
            try:
                if incremental:
                    rows = [
                        self._sample_single_point(vars_info, variable_order, method, rng, compiled, max_trials)
                        for _ in range(n_points)
                    ]
                    matrix = np.vstack(rows)
                else:
                    inputs = self._sample_matrix(vars_info, variable_order, n_points, method, rng)
                    value_map = {var: inputs[:, idx] for idx, var in enumerate(variable_order)}
                    try:
                        target = self._evaluate(compiled, value_map)
                    except Exception:
                        continue
                    target_arr = np.asarray(target, dtype=float)
                    if target_arr.shape != (n_points,):
                        if target_arr.size == 1:
                            target_arr = np.full(n_points, float(target_arr), dtype=float)
                        else:
                            squeezed = np.squeeze(target_arr)
                            if squeezed.shape == (n_points,):
                                target_arr = squeezed
                            else:
                                try:
                                    target_arr = np.broadcast_to(target_arr, (n_points,))
                                except ValueError as exc:
                                    raise ValueError(
                                        f"Could not broadcast target values to length {n_points} for {eq_id}"
                                    ) from exc
                    matrix = np.column_stack((inputs, target_arr))
                if np.all(np.isfinite(matrix)):
                    return matrix
            except Exception:
                continue
        raise RuntimeError(f"Failed to sample finite data for {eq_id} after {max_trials} attempts")

    def sample(
        self,
        eq_id: str,
        *,
        n_points: int = 100,
        method: str = "random",
        max_trials: int = 100,
        incremental: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> Dict[str, Any]:
        """Sample a dataset for the requested equation."""
        if eq_id not in self._entries:
            raise KeyError(f"Unknown equation id: {eq_id}")
        if n_points < 1:
            raise ValueError("n_points must be positive")
        rng = self._resolve_rng(random_state) if random_state is not None else self._rng
        entry = self._entries[eq_id]
        try:
            matrix = self._sample_entry(eq_id, entry, n_points, method, max_trials, incremental, rng)
        except RuntimeError as exc:
            if not incremental:
                warnings.warn(
                    f"Falling back to incremental sampling for {eq_id} after vectorized sampling failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                matrix = self._sample_entry(eq_id, entry, n_points, method, max_trials, True, rng)
            else:
                raise
        inputs = matrix[:, :-1]
        target = matrix[:, -1]
        vars_info = entry.get("vars", {})
        compiled = self._compile_expression(eq_id, entry)
        variable_order = compiled["variable_order"]
        feature_meta: List[Dict[str, Any]] = []
        for idx, key in enumerate(variable_order):
            spec = vars_info.get(key, {})
            feature_meta.append(
                {
                    "id": key,
                    "name": spec.get("name", key),
                    "metadata": spec,
                    "values": inputs[:, idx],
                }
            )
        target_meta = vars_info.get("v0", {})
        metadata = {k: v for k, v in entry.items() if k != "vars"}
        metadata["variable_order"] = list(variable_order)
        metadata["prepared_prefix"] = list(compiled["prefix"])
        metadata["prepared_normalized"] = compiled.get("normalized_infix")

        return {
            "eq_id": eq_id,
            "metadata": metadata,
            "n_points": n_points,
            "method": method,
            "incremental": incremental,
            "data": {
                "X": inputs,
                "y": target,
                "columns": list(variable_order),
                "target": target_meta.get("name", "v0"),
            },
            "variables": {
                "inputs": feature_meta,
                "target": {
                    "id": "v0",
                    "name": target_meta.get("name", "v0"),
                    "metadata": target_meta,
                    "values": target,
                },
            },
        }

    def sample_multiple(
        self,
        eq_ids: Optional[Union[str, Sequence[str]]] = None,
        *,
        count: int = 5,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        **sample_kwargs: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Sample multiple datasets per equation.

        Parameters
        ----------
        eq_ids:
            Optional identifier or iterable of identifiers to sample. If omitted, all
            equations in the benchmark are used.
        count:
            Number of datasets to draw for each equation (default: 5).
        random_state:
            Optional seed or generator to make the repeated sampling reproducible.
        sample_kwargs:
            Additional keyword arguments forwarded to :meth:`sample` (e.g. ``n_points``).

        Returns
        -------
        dict
            Mapping from equation id to a list with ``count`` sampled dataset dictionaries.
        """

        if count < 1:
            raise ValueError("count must be a positive integer")

        if eq_ids is None:
            eq_list = list(self._entries.keys())
        elif isinstance(eq_ids, str):
            eq_list = [eq_ids]
        else:
            eq_list = list(eq_ids)

        shared_rng: Optional[np.random.Generator] = None
        if random_state is not None:
            shared_rng = self._resolve_rng(random_state)

        results: Dict[str, List[Dict[str, Any]]] = {}
        for eq_id in eq_list:
            datasets: List[Dict[str, Any]] = []
            for _ in range(count):
                if shared_rng is not None:
                    sample = self.sample(eq_id, random_state=shared_rng, **sample_kwargs)
                else:
                    sample = self.sample(eq_id, **sample_kwargs)
                datasets.append(sample)
            results[eq_id] = datasets
        return results

    def iter_samples(
        self,
        eq_ids: Optional[Union[str, Sequence[str]]] = None,
        *,
        count: int = 5,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        **sample_kwargs: Any,
    ) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
        """Yield datasets lazily, one equation-instance at a time.

        Parameters
        ----------
        eq_ids:
            Optional identifier or iterable of identifiers to sample. If omitted, iterate over
            every equation in the benchmark.
        count:
            Number of datasets to draw for each equation (default: 5).
        random_state:
            Optional seed or generator to ensure reproducible iteration.
        sample_kwargs:
            Extra keyword arguments for :meth:`sample` (e.g. ``n_points``).

        Yields
        ------
        tuple
            ``(eq_id, index, sample_dict)`` where ``index`` runs from ``0`` to ``count-1`` for each
            equation.
        """

        if count < 1:
            raise ValueError("count must be a positive integer")

        if eq_ids is None:
            eq_list = list(self._entries.keys())
        elif isinstance(eq_ids, str):
            eq_list = [eq_ids]
        else:
            eq_list = list(eq_ids)

        rng = self._resolve_rng(random_state) if random_state is not None else None

        for eq_id in eq_list:
            for i in range(count):
                try:
                    if rng is not None:
                        sample = self.sample(eq_id, random_state=rng, **sample_kwargs)
                    else:
                        sample = self.sample(eq_id, **sample_kwargs)
                except Exception as exc:  # pragma: no cover - defensive against SimpliPy edge cases
                    warnings.warn(
                        f"Failed to sample FastSRB equation {eq_id}: {exc}. Skipping remaining repeats.",
                        RuntimeWarning,
                    )
                    break
                yield eq_id, i, sample
