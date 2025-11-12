"""
Evaluation helpers for the FastSRB benchmark.

Code translated and adapted from the Julia FastSRB benchmarking code by Viktor Martinek.

@misc{martinek2025fastsymbolicregressionbenchmarking,
      title={Fast Symbolic Regression Benchmarking},
      author={Viktor Martinek},
      year={2025},
      eprint={2508.14481},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.14481},
}

https://github.com/viktmar/FastSRB

MIT License

Copyright (c) 2025 Viktor Martinek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import pickle
import re
import time
import warnings
from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from simplipy.utils import numbers_to_constant
from tqdm import tqdm

from flash_ansr.benchmarks import FastSRBBenchmark
from flash_ansr.eval.evaluation import Evaluation
from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.refine import ConvergenceError
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.paths import substitute_root_path


class FastSRBEvaluation(Evaluation):
    """Evaluate a Flash-ANSR model on the FastSRB benchmark."""

    def __init__(
        self,
        n_support: int | None = None,
        noise_level: float = 0.0,
        complexity: str | list[int | float] = "none",
        preprocess: bool = False,
        device: str = "cpu",
        refiner_workers: int | None = None,
        *,
        benchmark_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            n_support=n_support,
            noise_level=noise_level,
            complexity=complexity,
            preprocess=preprocess,
            device=device,
            refiner_workers=refiner_workers,
        )

        default_cfg: dict[str, Any] = {
            "benchmark_path": "data/fastsrb/expressions.yaml",
            "count": 1,
            "n_points": 100,
            "method": "random",
            "max_trials": 100,
            "incremental": False,
            "random_state": None,
            "equations": None,
        }

        if benchmark_config is not None:
            for key, value in benchmark_config.items():
                if value is not None:
                    default_cfg[key] = value

        # Normalise and validate values
        try:
            default_cfg["count"] = int(default_cfg["count"])
        except (TypeError, ValueError) as exc:
            raise ValueError("fastsrb.count must be an integer") from exc

        try:
            default_cfg["n_points"] = int(default_cfg["n_points"])
        except (TypeError, ValueError) as exc:
            raise ValueError("fastsrb.n_points must be an integer") from exc

        try:
            default_cfg["max_trials"] = int(default_cfg["max_trials"])
        except (TypeError, ValueError) as exc:
            raise ValueError("fastsrb.max_trials must be an integer") from exc

        default_cfg["incremental"] = bool(default_cfg["incremental"])

        method = str(default_cfg["method"]).lower()
        if method not in {"random", "range"}:
            raise ValueError("fastsrb.method must be 'random' or 'range'")
        default_cfg["method"] = method

        random_state = default_cfg.get("random_state")
        if random_state is not None:
            try:
                random_state = int(random_state)
            except (TypeError, ValueError) as exc:
                raise ValueError("fastsrb.random_state must be an integer or null") from exc
        default_cfg["random_state"] = random_state

        equations = default_cfg.get("equations")
        if isinstance(equations, str):
            equations = [eq.strip() for eq in equations.split() if eq.strip()]
        elif equations is not None:
            equations = [str(eq) for eq in equations]
        default_cfg["equations"] = equations

        default_cfg["benchmark_path"] = str(default_cfg.get("benchmark_path") or "data/fastsrb/expressions.yaml")

        self.benchmark_config: dict[str, Any] = default_cfg
        self.benchmark_path: str = default_cfg["benchmark_path"]
        self.benchmark_random_state: Optional[int] = default_cfg["random_state"]
        self.benchmark_equations: Optional[list[str]] = default_cfg["equations"]

    @classmethod
    def from_config(cls, config: dict[str, Any] | str) -> "FastSRBEvaluation":
        config_dict = load_config(config)
        if "evaluation" in config_dict:
            config_section = config_dict["evaluation"]
        else:
            config_section = config_dict
        benchmark_cfg = dict(config_section.get("fastsrb", {}))
        points_value_raw = benchmark_cfg.get("n_points")
        if points_value_raw is None:
            raise ValueError("fastsrb.n_points must be specified in the evaluation config")
        try:
            points_value = int(points_value_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("fastsrb.n_points must be an integer") from exc
        config_section = dict(config_section)
        benchmark_cfg["n_points"] = points_value
        config_section["fastsrb"] = benchmark_cfg
        config_section["n_support"] = points_value
        base = Evaluation.from_config(config_section)
        return cls(
            n_support=base.n_support,
            noise_level=base.noise_level,
            complexity=base.complexity,
            preprocess=base.preprocess,
            device=base.device,
            refiner_workers=base.refiner_workers,
            benchmark_config=benchmark_cfg,
        )

    def evaluate(
        self,
        model: FlashANSR,
        benchmark: FastSRBBenchmark,
        *,
        count: Optional[int] = None,
        n_points: Optional[int] = None,
        method: Optional[str] = None,
        max_trials: Optional[int] = None,
        incremental: Optional[bool] = None,
        random_state: Optional[int] = None,
        eq_ids: Optional[Sequence[str]] = None,
        results_dict: Optional[dict[str, Any]] = None,
        size: Optional[int] = None,
        save_every: Optional[int] = None,
        output_file: Optional[str] = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the FastSRB benchmark and collect evaluation results."""
        cfg = self.benchmark_config
        count = cfg["count"] if count is None else count
        support_points_raw = cfg["n_points"] if n_points is None else n_points
        try:
            support_points = int(support_points_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_points must be an integer") from exc
        method = cfg["method"] if method is None else method
        max_trials = cfg["max_trials"] if max_trials is None else max_trials
        incremental = cfg["incremental"] if incremental is None else incremental
        resolved_random_state = cfg["random_state"] if random_state is None else random_state
        if eq_ids is None:
            eq_ids_cfg = cfg.get("equations")
            resolved_eq_ids: Optional[Sequence[str]]
            if eq_ids_cfg is None:
                resolved_eq_ids = None
            else:
                resolved_eq_ids = list(eq_ids_cfg)
        else:
            resolved_eq_ids = list(eq_ids)

        if count < 1:
            raise ValueError("count must be positive")
        if support_points < 1:
            raise ValueError("n_points must be positive")
        if method not in {"random", "range"}:
            raise ValueError("method must be 'random' or 'range'")
        if save_every is not None and output_file is None:
            raise ValueError("output_file must be provided when save_every is set")

        model.to(self.device).eval()

        if results_dict is None:
            results_store: defaultdict[str, list[Any]] = defaultdict(list)
            existing_results = 0
        else:
            results_store = defaultdict(list)
            for key, value in results_dict.items():
                results_store[key] = list(value)

            lengths = {key: len(value) for key, value in results_store.items()}
            if lengths and len(set(lengths.values())) != 1:
                raise ValueError("Existing results_dict has inconsistent lengths.")

            existing_results = next(iter(lengths.values())) if lengths else 0

        resolved_output_file: Optional[str] = None
        if save_every is not None:
            resolved_output_file = substitute_root_path(output_file)

        available_ids = set(benchmark.equation_ids())
        if resolved_eq_ids is None:
            eq_list = sorted(available_ids)
        else:
            missing = sorted(set(resolved_eq_ids) - available_ids)
            if missing:
                raise KeyError(f"Unknown FastSRB equation ids: {', '.join(missing)}")
            eq_list = list(resolved_eq_ids)

        total_samples = len(eq_list) * count
        if size is None:
            target_total = total_samples
        else:
            target_total = min(size, total_samples)

        if existing_results >= target_total:
            if verbose:
                print(
                    "Requested evaluation size is not larger than the existing results. Returning without new runs."
                )
            return dict(sorted(dict(results_store).items()))  # type: ignore

        if self.noise_level > 0:
            if resolved_random_state is not None:
                noise_rng = np.random.default_rng(resolved_random_state)
            else:
                noise_rng = np.random.default_rng()
        else:
            noise_rng = None

        remaining = target_total - existing_results
        collected = 0
        skipped = 0

        progress: Optional[tqdm] = None
        if verbose and remaining > 0:
            progress = tqdm(
                total=remaining,
                desc="FastSRB Evaluation",
                smoothing=0.0,
            )

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Sample twice as many points so we can hold out half for validation.
        total_sample_points = support_points * 2

        iterator = benchmark.iter_samples(
            eq_ids=eq_list,
            count=count,
            random_state=resolved_random_state,
            n_points=total_sample_points,
            method=method,
            max_trials=max_trials,
            incremental=incremental,
        )

        try:
            for global_index, (eq_id, sample_index, sample) in enumerate(iterator):
                if skipped < existing_results:
                    skipped += 1
                    continue

                if collected >= remaining:
                    break

                metadata: Mapping[str, Any] = sample.get("metadata", {})
                data_block: Mapping[str, Any] = sample["data"]
                inputs = np.asarray(data_block["X"], dtype=np.float32)
                targets = np.asarray(data_block["y"], dtype=np.float32)

                total_points = inputs.shape[0]
                if total_points == 0:
                    warnings.warn(f"Sample for {eq_id} has no data. Skipping.")
                    continue

                desired_support = int(support_points)
                if self.n_support is not None:
                    desired_support = min(int(self.n_support), desired_support)

                n_support = max(1, min(desired_support, total_points))

                X_support = inputs[:n_support]
                y_support = targets[:n_support]

                X_val = inputs[n_support:]
                y_val = targets[n_support:]

                if self.noise_level > 0 and noise_rng is not None:
                    y_std = float(np.std(y_support))
                    if np.isfinite(y_std) and y_std > 0:
                        noise = noise_rng.normal(size=y_support.shape)
                        y_noisy_support = y_support + self.noise_level * y_std * noise.astype(np.float32)
                    else:
                        y_noisy_support = y_support.copy()
                    if not np.all(np.isfinite(y_noisy_support)):
                        warnings.warn("Noisy targets contain non-finite values. Skipping sample.")
                        continue
                else:
                    y_noisy_support = y_support.copy()

                ground_truth_expr = metadata.get("prepared") or metadata.get("raw")
                ground_truth_prefix: Optional[list[str]] = None
                if isinstance(ground_truth_expr, str) and ground_truth_expr:
                    normalized = re.sub(r"\bv(\d+)\b", lambda match: f"x{match.group(1)}", ground_truth_expr)
                    try:
                        parsed = model.simplipy_engine.parse(normalized, mask_numbers=True)
                        ground_truth_prefix = list(parsed)
                    except Exception:
                        ground_truth_prefix = None
                        warnings.warn(
                            f"Failed to parse ground truth expression for {eq_id}.",
                            RuntimeWarning,
                        )

                if isinstance(self.complexity, (int, float)):
                    complexity_value: Optional[float | int] = self.complexity
                elif isinstance(self.complexity, list):
                    complexity_value = self.complexity[0] if self.complexity else None
                elif self.complexity == "ground_truth":
                    complexity_value = len(ground_truth_prefix) if ground_truth_prefix else None
                elif self.complexity == "none":
                    complexity_value = None
                else:
                    raise NotImplementedError(f"Unsupported complexity mode: {self.complexity}")

                variable_names: Optional[list[str]] = None
                variables_block = sample.get("variables")
                if isinstance(variables_block, Mapping):
                    inputs_meta = variables_block.get("inputs")
                    if isinstance(inputs_meta, Iterable):
                        variable_names = [str(meta.get("name", f"x{i + 1}")) for i, meta in enumerate(inputs_meta)]

                sample_results = {
                    "skeleton": None,
                    "skeleton_hash": tuple(ground_truth_prefix) if ground_truth_prefix else None,
                    "expression": ground_truth_prefix.copy() if ground_truth_prefix else None,
                    "input_ids": None,
                    "labels": None,
                    "constants": [],
                    "x": X_support.copy(),
                    "y": y_support.copy(),
                    "y_noisy": y_noisy_support.copy(),
                    "x_val": X_val.copy(),
                    "y_val": y_val.copy(),
                    "y_noisy_val": y_val.copy(),
                    "n_support": int(n_support),
                    "labels_decoded": ground_truth_prefix.copy() if ground_truth_prefix else None,
                    "parsimony": model.parsimony,
                    "noise_level": self.noise_level,
                    "fit_time": None,
                    "predicted_expression": None,
                    "predicted_expression_prefix": None,
                    "predicted_skeleton_prefix": None,
                    "predicted_constants": None,
                    "predicted_score": None,
                    "predicted_log_prob": None,
                    "y_pred": None,
                    "y_pred_val": None,
                    "prediction_success": False,
                    "error": None,
                    "benchmark_eq_id": eq_id,
                    "benchmark_sample_index": int(sample_index),
                    "benchmark_metadata": metadata,
                    "benchmark_n_points": int(total_points),
                    "benchmark_support_points": int(support_points),
                    "benchmark_method": method,
                    "ground_truth_infix": ground_truth_expr,
                    "ground_truth_prefix": ground_truth_prefix.copy() if ground_truth_prefix else None,
                }

                error_occured = False

                fit_time_start = time.time()
                try:
                    model.fit(
                        X_support,
                        y_noisy_support,
                        variable_names=variable_names,
                        complexity=complexity_value,
                    )
                    fit_time = time.time() - fit_time_start
                    sample_results["fit_time"] = fit_time
                    sample_results["prediction_success"] = True
                except (ConvergenceError, OverflowError, TypeError, ValueError) as exc:
                    warnings.warn(f"Error while fitting the model: {exc}. Filling nan.")
                    error_occured = True
                    sample_results["error"] = str(exc)

                if not error_occured:
                    if not model._results:
                        warnings.warn("Model produced no results. Filling nan.")
                        error_occured = True
                        sample_results["error"] = "Model produced no results."

                if not error_occured:
                    best_result = model._results[0]
                    try:
                        y_pred_support = model.predict(X_support, nth_best_beam=0, nth_best_constants=0)
                        if X_val.shape[0] > 0:
                            y_pred_val = model.predict(X_val, nth_best_beam=0, nth_best_constants=0)
                        else:
                            y_pred_val = np.empty(0, dtype=np.float32)
                        sample_results["y_pred"] = np.asarray(y_pred_support).copy()
                        sample_results["y_pred_val"] = np.asarray(y_pred_val).copy()
                    except (ConvergenceError, ValueError) as exc:
                        warnings.warn(f"Error while predicting: {exc}. Filling nan.")
                        error_occured = True
                        sample_results["error"] = str(exc)

                    if not error_occured:
                        predicted_expression_readable = model.get_expression(
                            nth_best_beam=0,
                            nth_best_constants=0,
                            map_variables=True,
                        )
                        predicted_skeleton_prefix = model.get_expression(
                            nth_best_beam=0,
                            nth_best_constants=0,
                            return_prefix=True,
                            map_variables=False,
                        )
                        sample_results["predicted_expression"] = predicted_expression_readable
                        sample_results["predicted_expression_prefix"] = predicted_skeleton_prefix.copy()
                        sample_results["predicted_skeleton_prefix"] = numbers_to_constant(predicted_skeleton_prefix).copy()

                        predicted_constants = None
                        if best_result.get("fits"):
                            predicted_constants = best_result["fits"][0][0].tolist()
                        sample_results["predicted_constants"] = predicted_constants
                        sample_results["predicted_score"] = best_result.get("score")
                        sample_results["predicted_log_prob"] = best_result.get("log_prob")

                for key, value in sample_results.items():
                    results_store[key].append(value)

                collected += 1
                if progress is not None:
                    progress.update(1)

                if save_every is not None and resolved_output_file is not None:
                    total_so_far = existing_results + collected
                    if total_so_far % save_every == 0:
                        try:
                            with open(resolved_output_file, "wb") as handle:
                                pickle.dump(dict(sorted(dict(results_store).items())), handle)
                        except KeyboardInterrupt:
                            warnings.warn("Interrupted during intermediate save. Results persisted before exiting.")
                            with open(resolved_output_file, "wb") as handle:
                                pickle.dump(dict(sorted(dict(results_store).items())), handle)
                            raise
        finally:
            if progress is not None:
                progress.close()

        if collected < remaining and verbose:
            warnings.warn(f"Collected {collected} samples but target was {remaining}.")

        results_sorted = dict(sorted(dict(results_store).items()))  # type: ignore

        if save_every is not None and resolved_output_file is not None:
            try:
                with open(resolved_output_file, "wb") as handle:
                    pickle.dump(results_sorted, handle)
            except KeyboardInterrupt:
                warnings.warn("Evaluation interrupted during saving. Trying to save results before exiting...")
                with open(resolved_output_file, "wb") as handle:
                    pickle.dump(results_sorted, handle)
                raise

        return results_sorted
