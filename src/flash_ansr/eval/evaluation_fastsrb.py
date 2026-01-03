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

import warnings
from typing import Any, Mapping, Optional, Sequence

from flash_ansr.benchmarks import FastSRBBenchmark
from flash_ansr.eval.evaluation import Evaluation
from flash_ansr.eval.data_sources import FastSRBSource
from flash_ansr.eval.engine import EvaluationEngine
from flash_ansr.eval.model_adapters import FlashANSRAdapter
from flash_ansr.eval.result_store import ResultStore
from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.utils.config_io import load_config


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
        if save_every is not None and output_file is None:
            raise ValueError("output_file must be provided when save_every is set")

        cfg = self.benchmark_config
        count = cfg["count"] if count is None else count
        support_points_raw = cfg["n_points"] if n_points is None else n_points
        try:
            support_points = int(support_points_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_points must be an integer") from exc
        method = (cfg["method"] if method is None else method).lower()
        if method not in {"random", "range"}:
            raise ValueError("method must be 'random' or 'range'")
        max_trials = cfg["max_trials"] if max_trials is None else max_trials
        incremental = cfg["incremental"] if incremental is None else incremental
        resolved_random_state = cfg["random_state"] if random_state is None else random_state

        if eq_ids is None:
            eq_spec = cfg.get("equations")
            resolved_eq_ids: Optional[Sequence[str]]
            if eq_spec is None:
                resolved_eq_ids = None
            else:
                resolved_eq_ids = list(eq_spec)
        else:
            resolved_eq_ids = list(eq_ids)

        available_ids = set(benchmark.equation_ids())
        if resolved_eq_ids is None:
            eq_list = sorted(available_ids)
        else:
            missing = sorted(set(resolved_eq_ids) - available_ids)
            if missing:
                raise KeyError(f"Unknown FastSRB equation ids: {', '.join(missing)}")
            eq_list = list(resolved_eq_ids)

        if not eq_list:
            raise ValueError("No FastSRB equations available for evaluation.")
        if count < 1:
            raise ValueError("count must be positive")
        if support_points < 1:
            raise ValueError("n_points must be positive")

        total_samples = len(eq_list) * count
        if total_samples <= 0:
            raise ValueError("FastSRB configuration produced zero samples.")

        store = ResultStore(results_dict)
        existing = store.size

        target_total = total_samples if size is None else min(size, total_samples)
        if existing >= target_total:
            if target_total < existing:
                warnings.warn(
                    "Requested evaluation size is smaller than the number of existing FastSRB results. "
                    "Returning existing results without additional evaluation."
                )
            return dict(sorted(store.snapshot().items()))

        remaining = target_total - existing

        data_source = FastSRBSource(
            benchmark=benchmark,
            target_size=remaining,
            skip=existing,
            eq_ids=eq_list,
            datasets_per_expression=count,
            support_points=support_points,
            sample_points=support_points * 2,
            n_support_override=self.n_support,
            method=method,
            max_trials=max_trials,
            incremental=incremental,
            random_state=resolved_random_state,
            noise_level=self.noise_level,
        )

        adapter = FlashANSRAdapter(
            model,
            device=self.device,
            complexity=self.complexity,
            refiner_workers=self.refiner_workers,
        )

        engine = EvaluationEngine(
            data_source=data_source,
            model_adapter=adapter,
            result_store=store,
        )

        results = engine.run(
            limit=remaining,
            save_every=save_every,
            output_path=output_file,
            verbose=verbose,
            progress=verbose,
        )

        return dict(sorted(results.items()))
