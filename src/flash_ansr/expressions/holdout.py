"""Holdout management for expression skeleton sampling."""
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple
import warnings

import numpy as np

from flash_ansr.expressions.compilation import safe_f


@dataclass
class HoldoutManager:
    """Track previously sampled skeletons to avoid duplicates."""

    n_variables: int
    allow_nan: bool
    holdout_X: np.ndarray = field(default_factory=lambda: np.random.uniform(-10, 10, (512, 100)))
    holdout_C: np.ndarray = field(default_factory=lambda: np.random.uniform(-10, 10, (100,)))
    skeleton_hashes: set[Tuple[str, ...]] = field(default_factory=set)
    expression_images: set[Tuple[float, ...] | Tuple[Tuple[float, ...], ...]] = field(default_factory=set)

    def register_skeleton(
        self,
        skeleton_tokens: Sequence[str],
        compiled_fn: Callable[..., np.ndarray | float],
        num_constants: int,
        *,
        n_variables: int | None = None,
    ) -> None:
        skeleton_key = tuple(self._normalize_tokens(skeleton_tokens))
        self.skeleton_hashes.add(skeleton_key)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                key = self._evaluate_to_key(compiled_fn, num_constants, n_variables)
        except (OverflowError, NameError):
            return

        self.expression_images.add(key)

    def is_held_out(
        self,
        skeleton_tokens: Sequence[str],
        compiled_fn: Callable[..., np.ndarray | float],
        num_constants: int,
        *,
        n_variables: int | None = None,
    ) -> bool:
        skeleton_key = tuple(self._normalize_tokens(skeleton_tokens))
        if skeleton_key in self.skeleton_hashes:
            return True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            key = self._evaluate_to_key(compiled_fn, num_constants, n_variables)
        return key in self.expression_images

    @staticmethod
    def _normalize_tokens(tokens: Sequence[str]) -> list[str]:
        return [str(token) for token in tokens]

    def _evaluate_to_key(
        self,
        compiled_fn: Callable[..., np.ndarray | float],
        num_constants: int,
        n_variables: int | None = None,
    ) -> Tuple[Tuple[float, ...], ...] | Tuple[float, ...]:
        variable_count = n_variables if n_variables is not None else self.n_variables
        samples = self.holdout_X[:, :variable_count]
        constants_slice = self.holdout_C[:num_constants]
        constants_arg = None if num_constants == 0 else constants_slice

        image = safe_f(compiled_fn, samples, constants_arg)
        image = np.asarray(image, dtype=np.float64)
        image = np.round(image, 4)

        if np.isnan(image).any():
            image = image.copy()
            image[np.isnan(image)] = 0.0

        if image.ndim == 1:
            return tuple(float(value) for value in image.tolist())

        return tuple(tuple(float(value) for value in row.tolist()) for row in image.tolist())
