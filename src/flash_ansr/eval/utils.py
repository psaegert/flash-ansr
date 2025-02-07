from typing import Any, Callable

from nltk.stem.api import StemmerI
import numpy as np


class NoOpStemmer(StemmerI):
    def stem(self, token: Any) -> Any:
        return token


def bootstrapped_metric_ci(data: np.ndarray, metric: Callable, n: int = 10_000, interval: float = 0.95) -> tuple[float, float, float]:
    # Generate all bootstrap samples at once
    indices = np.random.randint(0, len(data), size=(n, len(data)))
    samples = data[indices]

    # Compute the metric across each bootstrap sample
    if samples.ndim == 2:
        bootstrapped_metrics = np.apply_along_axis(metric, axis=1, arr=samples)
        # bootstrapped_metrics = metric(samples, axis=1)
    else:
        bootstrapped_metrics = np.array([metric(sample) for sample in samples])

    # Calculate the median, lower, and upper bounds for the confidence interval
    median = np.median(bootstrapped_metrics)
    lower = np.percentile(bootstrapped_metrics, (1 - interval) / 2 * 100)
    upper = np.percentile(bootstrapped_metrics, (1 + interval) / 2 * 100)

    return median, lower, upper
