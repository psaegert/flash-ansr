"""Helpers for sampling mixed continuous/quantized support points."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict

import numpy as np

from flash_ansr.expressions.prior_factory import build_prior_callable


class SupportSamplingError(Exception):
    """Raised when support sampling fails to satisfy the configured constraints."""


def _to_scalar(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        raise ValueError("Expected scalar-compatible value, received empty array.")
    return float(array[0])


class ScaleTransform:
    """Apply a global log10 scale sampled from a prior."""

    def __init__(self, scale_prior: Callable[..., np.ndarray], *, probability: float = 1.0) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Transform probability must be within [0, 1].")

        self._scale_prior = scale_prior
        self.probability = float(probability)

    def apply(self, support: np.ndarray) -> np.ndarray:
        scale_factor = 10.0 ** _to_scalar(self._scale_prior(size=1))
        support *= scale_factor
        return support.astype(np.float32, copy=False)


class QuantizeTransform:
    """Quantize up to D-1 dimensions using sampled bin counts and edges."""

    _ABS_SPAN_EPS = 1e-12
    _REL_SPAN_EPS = 1e-9

    def __init__(
        self,
        n_variables: int,
        d_quantized_sampler: Callable[..., np.ndarray],
        n_bins_sampler: Callable[..., np.ndarray],
        *,
        max_bins: int,
        strategy: str,
        allow_all_dimensions: bool = False,
        probability: float = 1.0,
    ) -> None:
        self.n_variables = n_variables
        self._d_quantized_sampler = d_quantized_sampler
        self._n_bins_sampler = n_bins_sampler
        self.max_bins = max(2, int(max_bins))
        self.strategy = strategy
        self._is_even_strategy = strategy == "even"
        self.allow_all_dimensions = allow_all_dimensions
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Transform probability must be within [0, 1].")
        self.probability = float(probability)

    def apply(self, support: np.ndarray) -> np.ndarray:
        n_dims = support.shape[1]
        if n_dims == 0:
            return support

        max_quantizable = n_dims if self.allow_all_dimensions else max(0, n_dims - 1)
        if max_quantizable <= 0:
            return support

        raw_dims = self._d_quantized_sampler(size=1)
        d_quantized = int(round(_to_scalar(raw_dims)))
        d_quantized = max(0, min(max_quantizable, d_quantized))
        if d_quantized == 0:
            return support

        dims = np.random.choice(n_dims, size=d_quantized, replace=False)
        if dims.size == 0:
            return support.astype(np.float32, copy=False)

        columns = support[:, dims]
        mins = np.min(columns, axis=0)
        maxs = np.max(columns, axis=0)
        span = maxs - mins

        scale = np.maximum(np.abs(mins), np.abs(maxs))
        tolerance = np.maximum(self._ABS_SPAN_EPS, self._REL_SPAN_EPS * scale)

        finite = np.isfinite(mins) & np.isfinite(maxs)
        active_mask = finite & (span > tolerance)
        if not np.any(active_mask):
            return support.astype(np.float32, copy=False)

        active_dims = dims[active_mask]
        active_columns = columns[:, active_mask]
        active_mins = mins[active_mask]
        active_maxs = maxs[active_mask]

        quantized = self._quantize_dimensions(active_columns, active_mins, active_maxs)
        if quantized is not None:
            support[:, active_dims] = quantized

        return support.astype(np.float32, copy=False)

    def _quantize_dimensions(
        self,
        values: np.ndarray,
        mins: np.ndarray,
        maxs: np.ndarray,
    ) -> np.ndarray | None:
        if values.ndim != 2 or values.shape[1] == 0:
            return None

        n_dims = values.shape[1]
        raw_bins = self._n_bins_sampler(size=n_dims)
        n_bins = np.round(raw_bins).astype(int, copy=False)
        n_bins = np.clip(n_bins, 2, self.max_bins)

        if self._is_even_strategy:
            return self._quantize_even(values, mins, maxs, n_bins)

        result = np.asarray(values, dtype=np.float64, order="C", copy=True)
        updated = False

        for idx in range(n_dims):
            bins = int(n_bins[idx])
            if bins < 2:
                continue

            min_val = float(mins[idx])
            max_val = float(maxs[idx])
            if not np.isfinite(min_val) or not np.isfinite(max_val):
                continue

            edges = self._build_edges(min_val, max_val, bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            indices = np.searchsorted(edges[1:-1], result[:, idx], side="left")
            if centers.shape[0] > 1:
                np.minimum(indices, centers.shape[0] - 1, out=indices)
            result[:, idx] = centers[indices]
            updated = True

        if not updated:
            return None

        return result.astype(values.dtype, copy=False)

    def _quantize_even(
        self,
        values: np.ndarray,
        mins: np.ndarray,
        maxs: np.ndarray,
        n_bins: np.ndarray,
    ) -> np.ndarray | None:
        if values.size == 0:
            return None

        spans = maxs - mins
        # Broadcast-ready views
        mins_view = mins.reshape(1, -1)
        spans_view = spans.reshape(1, -1)
        bins_view = n_bins.astype(np.float64, copy=False).reshape(1, -1)

        values64 = np.asarray(values, dtype=np.float64, order="C")
        normalized = (values64 - mins_view) / spans_view
        np.clip(normalized, 0.0, 1.0 - np.finfo(np.float64).eps, out=normalized)

        indices = np.floor(normalized * bins_view).astype(np.int64, copy=True)
        max_indices = (n_bins - 1).reshape(1, -1)
        np.minimum(indices, max_indices, out=indices)
        centers = mins_view + (indices + 0.5) * spans_view / bins_view

        return centers.astype(values.dtype, copy=False)

    def _build_edges(self, min_val: float, max_val: float, n_bins: int) -> np.ndarray:
        if self.strategy == "even":
            return np.linspace(min_val, max_val, num=n_bins + 1, dtype=np.float64)

        if self.strategy == "uniform":
            internal_count = n_bins - 1
            if internal_count <= 0:
                return np.array([min_val, max_val], dtype=np.float64)

            for _ in range(8):
                internal = np.random.uniform(min_val, max_val, size=internal_count)
                internal = np.unique(np.sort(internal))
                if internal.size >= internal_count:
                    internal = internal[:internal_count]
                    break
            else:
                return np.linspace(min_val, max_val, num=n_bins + 1, dtype=np.float64)

            if internal.size < internal_count:
                filler = np.linspace(min_val, max_val, num=n_bins + 1, dtype=np.float64)[1:-1]
                needed = internal_count - internal.size
                internal = np.sort(np.concatenate((internal, filler[:needed])))

            return np.concatenate(([min_val], internal, [max_val]))

        raise ValueError(f"Unsupported binning strategy '{self.strategy}'.")


class SupportSampler:
    """Sample support points while mixing continuous draws with optional transformations."""

    def __init__(
        self,
        n_variables: int,
        independent_dimensions: bool,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self.n_variables = n_variables
        self.independent_dimensions = independent_dimensions
        self.config = deepcopy(config or {})

        self.require_unique = bool(self.config.get("require_unique", False))

        self.support_prior_spec = self.config.get("support_prior")
        if self.support_prior_spec is None:
            raise ValueError("Support sampler configuration must include 'support_prior'.")

        self.n_support_prior_spec = self.config.get("n_support_prior")
        if self.n_support_prior_spec is None:
            raise ValueError("Support sampler configuration must include 'n_support_prior'.")

        self._configured_max_n_support = self._extract_max_n_support(self.n_support_prior_spec)

        support_prior_callable = self.ensure_prior_callable(self.support_prior_spec)
        if support_prior_callable is None:
            raise ValueError("Failed to construct callable from 'support_prior' specification.")
        self.support_prior: Callable[..., np.ndarray] = support_prior_callable

        n_support_prior_callable = self.ensure_prior_callable(self.n_support_prior_spec)
        if n_support_prior_callable is None:
            raise ValueError("Failed to construct callable from 'n_support_prior' specification.")
        self.n_support_prior: Callable[..., np.ndarray] = n_support_prior_callable

        self.scale_transform: ScaleTransform | None = None
        self.quantize_transform: QuantizeTransform | None = None
        self._post_scale_transforms: list[Any] = []

        legacy_keys = {"support_scale_prior", "quantized"}
        unexpected = legacy_keys.intersection(self.config.keys())
        if unexpected:
            raise ValueError(
                "Support sampler configuration uses removed legacy fields: "
                + ", ".join(sorted(unexpected))
                + ". Convert configs to the transform-based schema."
            )

        transforms_cfg = self.config.get("transforms")
        if transforms_cfg is not None:
            self._initialize_transforms(transforms_cfg)

    @staticmethod
    def _extract_max_n_support(spec: Any) -> int | None:
        if spec is None:
            return None

        if isinstance(spec, dict):
            kwargs = spec.get("kwargs") or {}
            candidates: list[int] = []
            for key in ("max_value", "max", "high"):
                if key in kwargs:
                    try:
                        candidates.append(int(kwargs[key]))
                    except (TypeError, ValueError):
                        continue

            values = kwargs.get("values")
            if values is not None:
                try:
                    iter_values = list(values)
                except TypeError:
                    iter_values = []
                for value in iter_values:
                    try:
                        candidates.append(int(value))
                    except (TypeError, ValueError):
                        return None

            if candidates:
                return max(candidates)

        if isinstance(spec, list):
            maxima = [SupportSampler._extract_max_n_support(item) for item in spec]
            filtered = [value for value in maxima if value is not None]
            if filtered:
                return max(int(value) for value in filtered)

        return None

    @property
    def configured_max_n_support(self) -> int | None:
        """Return the maximum support size implied by the configuration, if available."""
        return self._configured_max_n_support

    @staticmethod
    def ensure_prior_callable(spec: Any | None) -> Callable[..., np.ndarray] | None:
        """Normalize prior specifications into callables."""
        if spec is None:
            return None
        if callable(spec):
            return spec
        if isinstance(spec, (dict, list)):
            return build_prior_callable(spec)
        raise TypeError(
            "Prior specification must be a callable or configuration dict/list; "
            f"got {type(spec).__name__}."
        )

    def sample_n_support(self) -> int:
        """Sample the number of support points to generate."""
        raw = self.n_support_prior(size=1)
        count = int(round(_to_scalar(raw)))
        return max(1, count)

    def sample(
        self,
        n_support: int,
        support_prior: Callable[..., np.ndarray] | None = None,
        support_scale_prior: Callable[..., np.ndarray] | None = None,
    ) -> np.ndarray:
        if n_support <= 0:
            raise SupportSamplingError("n_support must be positive.")

        support_prior_fn = support_prior or self.support_prior
        if support_prior_fn is None:
            raise SupportSamplingError("Support prior must be defined before sampling.")

        support = self._draw_continuous_support(n_support, support_prior_fn)

        transforms: list[Any] = []
        if support_scale_prior is not None:
            transforms.append(ScaleTransform(support_scale_prior))
        elif self.scale_transform is not None:
            transforms.append(self.scale_transform)

        transforms.extend(self._post_scale_transforms)

        for transform in transforms:
            probability = float(getattr(transform, "probability", 1.0))
            if probability <= 0.0:
                continue
            if probability < 1.0 and np.random.random() >= probability:
                continue
            support = transform.apply(support)

        self._maybe_check_unique(support, n_support)
        return support.astype(np.float32, copy=False)

    def _draw_continuous_support(
        self,
        n_support: int,
        support_prior: Callable[..., np.ndarray],
    ) -> np.ndarray:
        if self.independent_dimensions:
            columns = []
            for _ in range(self.n_variables):
                samples = support_prior(size=(n_support, 1))
                column = np.asarray(samples, dtype=np.float64).reshape(n_support)
                columns.append(column)
            support = np.stack(columns, axis=1)
        else:
            base = support_prior(size=(n_support, self.n_variables))
            support = np.asarray(base, dtype=np.float64).reshape(n_support, self.n_variables)
        return support.astype(np.float32)

    def _initialize_transforms(self, transforms_cfg: Any) -> None:
        if not isinstance(transforms_cfg, list):
            raise TypeError("'transforms' configuration must be a list of transform definitions.")

        for transform_cfg in transforms_cfg:
            if not isinstance(transform_cfg, dict):
                raise TypeError("Each transform configuration must be a mapping.")

            transform_type = str(transform_cfg.get("type", "")).strip().lower()
            probability_raw = transform_cfg.get("probability", 1.0)
            try:
                probability = float(probability_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("Transform probability must be a numeric value.") from exc

            if transform_type == "scale":
                prior_spec = transform_cfg.get("prior")
                if prior_spec is None:
                    raise ValueError("Scale transform requires a 'prior' specification.")

                prior_callable = self.ensure_prior_callable(prior_spec)
                if prior_callable is None:
                    raise ValueError("Failed to construct callable for scale transform prior.")

                if self.scale_transform is not None:
                    raise ValueError("Multiple scale transforms are not supported.")

                self.scale_transform = ScaleTransform(prior_callable, probability=probability)
            elif transform_type == "quantize":
                transform = self._build_quantize_transform(transform_cfg, probability=probability)
                if self.quantize_transform is not None:
                    raise ValueError("Multiple quantize transforms are not supported.")

                self.quantize_transform = transform
                self._post_scale_transforms.append(transform)
            else:
                raise ValueError(f"Unsupported transform type '{transform_type}'.")

    def _build_quantize_transform(self, quantize_cfg: Any, *, probability: float) -> QuantizeTransform:
        if not isinstance(quantize_cfg, dict):
            raise TypeError("Quantize transform configuration must be a mapping.")

        if "d_quantized" not in quantize_cfg or "n_bins" not in quantize_cfg:
            raise ValueError("Quantize transform requires 'd_quantized' and 'n_bins' specifications.")

        d_quantized_sampler = self.ensure_prior_callable(quantize_cfg["d_quantized"])
        if d_quantized_sampler is None:
            raise ValueError("Failed to build sampler for 'd_quantized'.")

        n_bins_sampler = self.ensure_prior_callable(quantize_cfg["n_bins"])
        if n_bins_sampler is None:
            raise ValueError("Failed to build sampler for 'n_bins'.")

        max_bins_cfg = quantize_cfg.get("max_bins")
        if max_bins_cfg is None:
            inferred_max = self._extract_max_n_support(quantize_cfg["n_bins"])
            max_bins = max(2, inferred_max) if inferred_max is not None else 64
        else:
            max_bins = int(max(2, max_bins_cfg))

        strategy = str(quantize_cfg.get("strategy", "even")).lower()
        if strategy not in {"even", "uniform"}:
            raise ValueError(f"Unsupported quantize strategy '{strategy}'.")

        allow_all_dims = bool(quantize_cfg.get("allow_all_dimensions", False))

        return QuantizeTransform(
            n_variables=self.n_variables,
            d_quantized_sampler=d_quantized_sampler,
            n_bins_sampler=n_bins_sampler,
            max_bins=max_bins,
            strategy=strategy,
            allow_all_dimensions=allow_all_dims,
            probability=probability,
        )

    def _maybe_check_unique(self, support: np.ndarray, expected: int) -> None:
        if not self.require_unique:
            return
        unique_rows = np.unique(support, axis=0)
        if unique_rows.shape[0] != expected:
            raise SupportSamplingError(
                "Duplicate support points detected after sampling; consider disabling 'require_unique'."
            )
