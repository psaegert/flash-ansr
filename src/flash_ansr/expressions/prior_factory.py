"""Factory helpers for constructing prior distribution callables."""
from typing import Any, Callable

import numpy as np

from flash_ansr.expressions.distributions import get_distribution


def build_prior_callable(config: dict[str, Any] | list[dict[str, Any]]) -> Callable:
    """Create a sampler function from a prior configuration."""
    if isinstance(config, list):
        distributions = [get_distribution(sub_config) for sub_config in config]
        weights = np.array([sub_config.get("weight", 1.0) for sub_config in config], dtype=np.float64)
        if weights.sum() == 0:
            raise ValueError("Mixture prior weights must sum to a positive value.")
        weights /= weights.sum()

        def mixture_distribution(size: Any = 1) -> Any:
            chosen_index = int(np.random.choice(len(distributions), p=weights))
            chosen_dist_callable = distributions[chosen_index]
            return chosen_dist_callable(size=size)

        return mixture_distribution

    if isinstance(config, dict):
        return get_distribution(config)

    raise TypeError(
        "Prior configuration must be a dict or a list of dicts; "
        f"got {type(config).__name__}."
    )
