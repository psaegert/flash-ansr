"""Core evaluation dataclasses and protocols.

These primitives provide a shared language for plugging data sources,
model adapters, and orchestration engines together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(slots=True)
class EvaluationSample:
    """Container for a single evaluation problem.

    Attributes
    ----------
    x_support: np.ndarray
        Support inputs shaped ``(n_support, n_features)``.
    y_support: np.ndarray
        Support targets shaped ``(n_support, 1)`` or ``(n_support,)``.
    x_validation: np.ndarray
        Validation inputs (may be empty) shaped ``(n_val, n_features)``.
    y_validation: np.ndarray
        Validation targets shaped ``(n_val, 1)`` or ``(n_val,)``.
    metadata: Mapping[str, Any]
        Arbitrary metadata (ground-truth expressions, skeleton hashes, etc.).
    is_placeholder: bool
        Flag indicating that no dataset could be generated and the entry is a placeholder.
    placeholder_reason: str | None
        Optional human-readable reason describing why the placeholder was emitted.
    """

    x_support: np.ndarray
    y_support: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    y_support_noisy: np.ndarray | None = None
    y_validation_noisy: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    is_placeholder: bool = False
    placeholder_reason: str | None = None

    def clone_metadata(self) -> Dict[str, Any]:
        """Return a mutable copy of the metadata dictionary."""

        return dict(self.metadata)

    @property
    def n_support(self) -> int:
        return int(self.x_support.shape[0])

    @property
    def n_validation(self) -> int:
        return int(self.x_validation.shape[0])


@dataclass(slots=True)
class EvaluationResult:
    """Normalized result returned by model adapters."""

    values: MutableMapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> MutableMapping[str, Any]:
        return self.values


@runtime_checkable
class EvaluationDataSource(Protocol):
    """Protocol all evaluation data providers must implement."""

    def __iter__(self) -> Iterable[EvaluationSample]:
        ...

    def size_hint(self) -> Optional[int]:
        """Return the number of samples if it is cheap to compute."""

        return None

    def prepare(self, *, adapter: EvaluationModelAdapter | None = None) -> None:
        """Hook executed before iteration begins."""

        # Default implementation is a no-op.
        return None


@runtime_checkable
class EvaluationModelAdapter(Protocol):
    """Protocol for adapting arbitrary model APIs to evaluation engine."""

    def prepare(self, *, data_source: EvaluationDataSource | None = None) -> None:
        """Hook executed before the first sample is evaluated."""

        return None

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Run the model on a single sample and return normalized outputs."""

        ...


__all__ = [
    "EvaluationSample",
    "EvaluationResult",
    "EvaluationDataSource",
    "EvaluationModelAdapter",
]
