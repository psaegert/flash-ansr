"""Metric utilities, including bootstrap confidence intervals."""
from typing import Callable

import numpy as np


def bootstrapped_metric_ci(
    data: np.ndarray,
    metric: Callable[[np.ndarray], float],
    n: int = 10_000,
    interval: float = 0.95,
) -> tuple[float, float, float]:
    """Estimate ``metric`` on ``data`` with a bootstrap confidence interval."""
    if interval > 1 and interval <= 100:
        interval /= 100

    n = int(n)

    indices = np.random.randint(0, len(data), size=(n, len(data)))
    samples = data[indices]

    if samples.ndim == 2:
        bootstrapped_metrics = np.apply_along_axis(metric, axis=1, arr=samples)
    else:
        bootstrapped_metrics = np.array([metric(sample) for sample in samples])

    median = np.nanmedian(bootstrapped_metrics)
    lower = np.nanpercentile(bootstrapped_metrics, (1 - interval) / 2 * 100)
    upper = np.nanpercentile(bootstrapped_metrics, (1 + interval) / 2 * 100)

    return median, lower, upper
