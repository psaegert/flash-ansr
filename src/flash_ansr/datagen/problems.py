"""Training-prior problems with clean validation rows, and the pool-size prior.

A problem is realized exactly as the training stream realizes it (the model bundle's ``catalog_train.yaml`` and the
``noise`` / ``n_support`` spec of its ``dataset_train.yaml``; unused input columns zeroed), plus ``n_validation``
clean rows from the same per-problem input distribution, for fresh-point scoring. Every problem carries the seed
that reproduces it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import yaml


@dataclass
class Problem:
    """One realized training-prior problem."""
    problem_id: str
    x: np.ndarray              # support inputs, unused columns zeroed
    y_noisy: np.ndarray        # support targets with the training noise (what the model reads and the refiner fits)
    y: np.ndarray              # clean support targets
    x_val: np.ndarray          # clean validation inputs (same input distribution)
    y_val: np.ndarray          # clean validation targets
    skeleton: list = field(default_factory=list)
    noise: dict | None = None
    k: int = 512               # pool size: number of raw draws to take for this problem


def log_uniform_k(rng: np.random.Generator, k_min: int = 1, k_max: int = 512) -> int:
    """Pool size with log K uniform on [log k_min, log k_max] (rounded to an integer in [k_min, k_max])."""
    return int(np.clip(np.round(np.exp(rng.uniform(np.log(k_min), np.log(k_max + 0.5)))), k_min, k_max))


def k_sampler(spec: str) -> Callable[[np.random.Generator], int]:
    """``'log2uniform:1:512'`` (K a power of two, log2 K uniform: the packed sampler decodes equal-K problems
    together), ``'loguniform:1:512'``, ``'uniform:1:512'`` or ``'fixed:512'`` -> callable(rng) -> int.

    A suffix ``@tag`` (e.g. ``uniform:1:512@e2``) leaves the distribution unchanged; it only changes the spec string
    that :func:`flash_ansr.datagen.targets.assign_k` hashes, so the same problems get independent K (a second epoch
    over the same draws with fresh K-pools and fronts)."""
    kind, *args = spec.split("@")[0].split(":")
    if kind == "fixed":
        k = int(args[0])
        return lambda rng: k
    lo, hi = int(args[0]), int(args[1])
    if kind == "log2uniform":
        exps = np.arange(int(np.log2(lo)), int(np.log2(hi)) + 1)
        if 2 ** exps[0] != lo or 2 ** exps[-1] != hi:
            raise ValueError(f"log2uniform bounds must be powers of two, got {lo}, {hi}")
        return lambda rng: int(2 ** rng.choice(exps))
    if kind == "loguniform":
        return lambda rng: log_uniform_k(rng, lo, hi)
    if kind == "uniform":
        return lambda rng: int(rng.integers(lo, hi + 1))
    raise ValueError(f"unknown K spec {spec!r}")


def problem_stream(model_dir: str, seed: int, *, n_validation: int = 512, k_spec: str = "fixed:512",
                   prefix: str = "p") -> Iterator[Problem]:
    """Yield training-prior problems of the model bundle in ``model_dir`` (deterministic given ``seed``).

    Problems that the prior realizes as placeholders, with non-finite values or a constant clean target are
    skipped (they would be dropped by any fitting stage anyway).
    """
    from symbolic_data import ProblemSource
    from flash_ansr.utils.tensor_ops import mask_unused_variable_columns

    ds = yaml.safe_load(open(os.path.join(model_dir, "dataset_train.yaml")))["source"]["sampling"]
    rng = np.random.default_rng([seed, 0])
    k_rng = np.random.default_rng([seed, 1])
    draw_k = k_sampler(k_spec)
    sampling = {"n_support": ds["n_support"], "n_validation": 0, "noise": ds.get("noise", 0.0),
                "problems_per_expression": 1}
    src = ProblemSource({"catalog": os.path.join(model_dir, "catalog_train.yaml"), "sampling": sampling}, rng=rng)
    # The prior's support-size mode has no validation split: draw the support size from the catalog's own prior
    # (as training does) and realize n_validation extra rows from the same per-problem input distribution.
    catalog = src._get_catalog()
    original_realize = src._realize_problem

    def realize_with_validation(catalog_: Any, entry: Any, n_support: Any, n_val: Any, engine: Any, rng_: Any) -> Any:
        if n_support is None:
            n_support, n_val = catalog.support_sampler.sample_n_support(rng_), n_validation
        return original_realize(catalog_, entry, n_support, n_val, engine, rng_)

    src._realize_problem = realize_with_validation
    i = 0
    for p in src:
        i += 1
        if p.is_placeholder or p.skeleton is None or len(p.constants):
            continue
        tokens = [str(t) for t in p.skeleton]
        x, x_val = np.array(p.x_support, float), np.array(p.x_validation, float)
        y, y_val = np.asarray(p.y_support, float).reshape(-1), np.asarray(p.y_validation, float).reshape(-1)
        y_noisy = np.asarray(p.y_support_noisy if p.y_support_noisy is not None else p.y_support, float).reshape(-1)
        if not all(np.isfinite(a).all() for a in (x, y, x_val, y_val, y_noisy)) or len(y) < 2 or np.var(y) == 0:
            continue
        names = [f"x{j + 1}" for j in range(x.shape[1])]
        mask_unused_variable_columns(arrays=(x, x_val), variables=names, skeleton_tokens=tokens, padding="zero")
        yield Problem(problem_id=f"{prefix}{seed}_{i:08d}", x=x, y_noisy=y_noisy, y=y, x_val=x_val, y_val=y_val,
                      skeleton=tokens, noise=p.noise if isinstance(p.noise, dict) else None, k=draw_k(k_rng))
