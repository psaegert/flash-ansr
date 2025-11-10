"""Distribution factories used to sample numerical constants."""
from functools import partial
from typing import Any, Callable

import numpy as np


def uniform_dist(
    low: float,
    high: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
) -> np.ndarray:
    """Sample uniformly from ``[low, high]`` with optional clipping."""
    low, high = min(low, high), max(low, high)
    samples = np.random.uniform(low, high, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def normal_dist(
    loc: float,
    scale: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
) -> np.ndarray:
    """Sample from a normal distribution with optional clipping."""
    scale = max(scale, 1e-9)
    samples = np.random.normal(loc, scale, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def log_uniform_dist(
    low: float,
    high: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
) -> np.ndarray:
    """Sample from a log-uniform distribution with optional clipping."""
    low, high = min(low, high), max(low, high)
    samples = np.exp(np.random.uniform(np.log(low), np.log(high), size=size))
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def log_normal_dist(
    mean: float,
    sigma: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
) -> np.ndarray:
    """Sample from a log-normal distribution with optional clipping."""
    sigma = max(sigma, 1e-9)
    samples = np.random.lognormal(mean, sigma, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def gamma_dist(
    shape: float,
    scale: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
) -> np.ndarray:
    """Sample from a gamma distribution with optional clipping."""
    samples = np.random.gamma(shape, scale, size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


def binomial_dist(
    n: int,
    p: float,
    min_value: float | None = None,
    max_value: float | None = None,
    size: Any = 1,
) -> np.ndarray:
    """Sample from a binomial distribution with optional clipping."""
    samples = np.random.binomial(int(n), float(p), size=size)
    if min_value is not None and max_value is not None:
        return np.clip(samples, min_value, max_value)
    return samples


BASE_DISTRIBUTIONS: dict[str, Callable[..., np.ndarray]] = {
    "uniform": uniform_dist,
    "normal": normal_dist,
    "log_uniform": log_uniform_dist,
    "log_normal": log_normal_dist,
    "gamma": gamma_dist,
    "binomial": binomial_dist,
}


def sampler_dist(
    base_dist_name: str,
    param_samplers: dict[str, Callable[..., np.ndarray]],
    base_kwargs: dict[str, Any] | None = None,
    size: Any = 1,
) -> np.ndarray:
    """Sample from ``base_dist_name`` after drawing its parameters from ``param_samplers``."""
    if base_dist_name not in BASE_DISTRIBUTIONS:
        raise ValueError(f"Unknown base_dist_name: {base_dist_name}")

    final_kwargs = base_kwargs.copy() if base_kwargs else {}
    for param_name, sampler_func in param_samplers.items():
        final_kwargs[param_name] = sampler_func(size=1)[0]  # type: ignore[index]

    base_dist_func = BASE_DISTRIBUTIONS[base_dist_name]
    return base_dist_func(**final_kwargs, size=size)


def get_distribution(config: dict[str, Any]) -> Callable[..., np.ndarray]:
    """Create a distribution callable from ``config`` (supports nested samplers)."""
    name = config["name"]
    kwargs = config.get("kwargs", {})

    if name == "constant":
        return lambda size=1: np.full(size, kwargs["value"])

    if name in BASE_DISTRIBUTIONS:
        return partial(BASE_DISTRIBUTIONS[name], **kwargs)

    if name == "sampler":
        resolved_samplers = {
            param_name: get_distribution(sampler_config)
            for param_name, sampler_config in kwargs["param_samplers"].items()
        }
        sampler_args = {
            "base_dist_name": kwargs["base_dist_name"],
            "param_samplers": resolved_samplers,
            "base_kwargs": kwargs.get("base_kwargs", {}),
        }
        return partial(sampler_dist, **sampler_args)

    raise ValueError(f"Unknown distribution name: {name}")
