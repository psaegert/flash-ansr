"""Prompt feature extraction and configuration helpers."""
import math
import random
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Iterable, Mapping, Sequence, cast

import numpy as np
from simplipy import SimpliPyEngine

from flash_ansr.expressions.skeleton_pool import NoValidSampleFoundError, SkeletonPool
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing.schemas import PromptFeatures


@dataclass(frozen=True)
class DistributionSpec:
    """Declarative description of a random variable."""

    name: str = "constant"
    params: Mapping[str, object] = field(default_factory=dict)

    def sample(self) -> float:
        name = self.name.lower()
        if name in {"constant", "deterministic"}:
            return self._get_float("value", 0.0)
        if name in {"uniform_int", "randint"}:
            low = self._get_int("low", 0)
            high = self._get_int("high", low)
            if high < low:
                low, high = high, low
            return float(np.random.randint(low, high + 1))
        if name in {"uniform", "uniform_float"}:
            low_value = self._get_float("low", 0.0)
            high_value = self._get_float("high", low_value)
            if high_value < low_value:
                low_value, high_value = high_value, low_value
            return float(np.random.uniform(low_value, high_value))
        if name == "poisson":
            lam = max(0.0, self._get_float("lam", self._get_float("lambda", 0.0)))
            return float(np.random.poisson(lam))
        if name == "geometric":
            p = self._get_float("p", 0.5)
            p = min(max(p, 1e-8), 1.0)
            return float(np.random.geometric(p))
        if name in {"normal", "gaussian"}:
            mean = self._get_float("mean", self._get_float("loc", 0.0))
            std = self._get_float("std", self._get_float("scale", 1.0))
            return float(np.random.normal(mean, std))
        if name == "triangular":
            left = self._get_float("left", self._get_float("low", 0.0))
            right = self._get_float("right", self._get_float("high", 1.0))
            if right < left:
                left, right = right, left
            mode_default = (left + right) / 2
            mode = self._get_float("mode", mode_default)
            return float(np.random.triangular(left, mode, right))
        raise ValueError(f"Unsupported distribution '{self.name}'.")

    def sample_int(self, *, minimum: int = 0, maximum: int | None = None) -> int:
        value = int(round(self.sample()))
        if maximum is not None:
            if maximum < minimum:
                minimum, maximum = maximum, minimum
            value = min(value, maximum)
        return max(minimum, value)

    @staticmethod
    def constant(value: int | float) -> "DistributionSpec":
        return DistributionSpec(name="constant", params={"value": value})

    @staticmethod
    def from_range(range_like: Sequence[object]) -> "DistributionSpec":
        if len(range_like) != 2:
            raise ValueError("Range-like distributions require two entries (low, high).")
        low = DistributionSpec._coerce_int_value(range_like[0], 0)
        high = DistributionSpec._coerce_int_value(range_like[1], low)
        return DistributionSpec(name="uniform_int", params={"low": low, "high": high})

    @classmethod
    def from_dict(
        cls,
        data: "DistributionSpec" | Mapping[str, object] | Sequence[object] | int | float | str | None,
    ) -> "DistributionSpec":
        if isinstance(data, cls):
            return data
        if data is None:
            return cls()
        if isinstance(data, (int, float)):
            return cls.constant(data)
        if isinstance(data, str):
            return cls(name=data, params={})
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return cls.from_range(list(data))
        if isinstance(data, Mapping):
            params_obj = data.get("params")
            if "name" in data:
                params = cls._normalize_params(params_obj)
                return cls(name=str(data["name"]), params=params)
            if "distribution" in data:
                params = cls._normalize_params(params_obj)
                return cls(name=str(data["distribution"]), params=params)
            if len(data) == 1:
                ((name, params_val),) = data.items()
                params = cls._normalize_params(params_val)
                return cls(name=str(name), params=params)
        raise TypeError(f"Unsupported distribution specification: {data!r}")

    def _get_float(self, key: str, default: float) -> float:
        value = self.params.get(key, default)
        return self._coerce_float_value(value, default)

    def _get_int(self, key: str, default: int) -> int:
        value = self.params.get(key, default)
        return self._coerce_int_value(value, default)

    @staticmethod
    def _coerce_float_value(value: object, default: float) -> float:
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        return float(default)

    @staticmethod
    def _coerce_int_value(value: object, default: int) -> int:
        if isinstance(value, Real):
            return int(float(value))
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                pass
        return int(default)

    @staticmethod
    def _normalize_params(value: object | None) -> dict[str, object]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        return {"value": value}


DistributionInput = (
    DistributionSpec | Mapping[str, object] | Sequence[object] | int | float | str | None
)


def _normalize_distribution_input(value: object | None) -> DistributionInput:
    if value is None or isinstance(value, (DistributionSpec, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise TypeError(f"Unsupported distribution specification: {value!r}")


def _normalize_kwargs(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"Keyword arguments must be provided as a mapping, got {type(value).__name__}.")


@dataclass
class PromptSectionConfig:
    probability: float = 1.0
    kwargs: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: "PromptSectionConfig" | Mapping[str, object] | None,
    ) -> "PromptSectionConfig":
        if isinstance(data, cls):
            return data
        if data is None:
            return cls()
        if isinstance(data, PromptSectionConfig):
            return cls(probability=data.probability, kwargs=dict(data.kwargs))
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected dict for prompt section config, got {type(data).__name__}")
        probability = DistributionSpec._coerce_float_value(data.get("probability", 1.0), 1.0)
        kwargs = _normalize_kwargs(data.get("kwargs"))
        return cls(probability=probability, kwargs=kwargs)


def _normalize_section_config(value: object | None) -> PromptSectionConfig | Mapping[str, object] | None:
    if value is None or isinstance(value, PromptSectionConfig):
        return value
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"Unsupported section configuration: {value!r}")


@dataclass
class ComplexitySectionConfig(PromptSectionConfig):
    """Placeholder for future complexity-specific options."""

    @classmethod
    def from_dict(
        cls,
        data: PromptSectionConfig | Mapping[str, object] | None,
    ) -> "ComplexitySectionConfig":
        if isinstance(data, cls):
            return data
        if isinstance(data, PromptSectionConfig):
            return cls(probability=data.probability, kwargs=dict(data.kwargs))
        base = PromptSectionConfig.from_dict(data)
        return cls(probability=base.probability, kwargs=base.kwargs)


@dataclass
class AllowedTermsConfig(PromptSectionConfig):
    actual_terms: DistributionSpec = field(default_factory=lambda: DistributionSpec.constant(4))
    generated_terms: DistributionSpec = field(default_factory=lambda: DistributionSpec.constant(0))
    length: DistributionSpec = field(default_factory=lambda: DistributionSpec.constant(3))
    min_length: int = 1
    max_relative_length: float = 0.45
    force_expression_term: bool = False

    @classmethod
    def from_dict(
        cls,
        data: PromptSectionConfig | Mapping[str, object] | None,
    ) -> "AllowedTermsConfig":
        if isinstance(data, cls):
            return data
        if isinstance(data, PromptSectionConfig):
            return cls(probability=data.probability, kwargs=dict(data.kwargs))
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected dict for allowed_terms config, got {type(data).__name__}")

        probability = DistributionSpec._coerce_float_value(data.get("probability", 1.0), 1.0)
        kwargs = _normalize_kwargs(data.get("kwargs"))

        actual_spec = DistributionSpec.from_dict(_normalize_distribution_input(data.get("actual_terms")))
        generated_spec = DistributionSpec.from_dict(_normalize_distribution_input(data.get("generated_terms")))
        length_spec = DistributionSpec.from_dict(_normalize_distribution_input(data.get("length")))

        min_length = DistributionSpec._coerce_int_value(data.get("min_length", 1), 1)
        max_relative_length = DistributionSpec._coerce_float_value(data.get("max_relative_length", 0.45), 0.45)
        force_expression_term = bool(data.get("force_expression_term", True))

        return cls(
            probability=probability,
            kwargs=kwargs,
            actual_terms=actual_spec,
            generated_terms=generated_spec,
            length=length_spec,
            min_length=max(1, min_length),
            max_relative_length=max(0.0, max_relative_length),
            force_expression_term=force_expression_term,
        )


@dataclass
class TermSelectionConfig(PromptSectionConfig):
    count: DistributionSpec = field(default_factory=lambda: DistributionSpec.constant(0))
    length: DistributionSpec = field(default_factory=lambda: DistributionSpec.constant(3))
    min_length: int = 1
    max_relative_length: float = 0.45

    @classmethod
    def from_dict(
        cls,
        data: PromptSectionConfig | Mapping[str, object] | None,
    ) -> "TermSelectionConfig":
        if isinstance(data, cls):
            return data
        if isinstance(data, PromptSectionConfig):
            return cls(probability=data.probability, kwargs=dict(data.kwargs))
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected dict for term selection config, got {type(data).__name__}")

        probability = DistributionSpec._coerce_float_value(data.get("probability", 1.0), 1.0)
        kwargs = _normalize_kwargs(data.get("kwargs"))
        count_spec = DistributionSpec.from_dict(_normalize_distribution_input(data.get("count")))
        length_spec = DistributionSpec.from_dict(_normalize_distribution_input(data.get("length")))
        min_length = DistributionSpec._coerce_int_value(data.get("min_length", 1), 1)
        max_relative_length = DistributionSpec._coerce_float_value(data.get("max_relative_length", 0.45), 0.45)

        return cls(
            probability=probability,
            kwargs=kwargs,
            count=count_spec,
            length=length_spec,
            min_length=max(1, min_length),
            max_relative_length=max(0.0, max_relative_length),
        )


class IncludeTermsConfig(TermSelectionConfig):
    """Configuration for included prompt terms."""


class ExcludeTermsConfig(TermSelectionConfig):
    """Configuration for excluded prompt terms."""


@dataclass
class PromptFeatureExtractorConfig:
    """Configuration controlling randomness and term sampling."""

    prompt_probability: float = 1.0
    complexity: ComplexitySectionConfig = field(default_factory=ComplexitySectionConfig)
    allowed_terms: AllowedTermsConfig = field(default_factory=AllowedTermsConfig)
    include_terms: IncludeTermsConfig = field(default_factory=IncludeTermsConfig)
    exclude_terms: ExcludeTermsConfig = field(default_factory=ExcludeTermsConfig)
    max_random_term_attempts: int = 64

    @classmethod
    def from_dict(
        cls,
        data: "PromptFeatureExtractorConfig" | Mapping[str, object] | None,
    ) -> "PromptFeatureExtractorConfig":
        if isinstance(data, cls):
            return data
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError(f"Expected dict for prompt_feature config, got {type(data).__name__}")

        if any(
            key in data
            for key in ("max_cover_terms", "extra_allowed_range", "include_terms_range", "exclude_terms_range")
        ):
            return cls._from_legacy_dict(data)

        prompt_prob_default = DistributionSpec._coerce_float_value(data.get("prompt_prob", 1.0), 1.0)
        prompt_probability = DistributionSpec._coerce_float_value(
            data.get("prompt_probability", prompt_prob_default),
            prompt_prob_default,
        )
        max_random_term_attempts = DistributionSpec._coerce_int_value(
            data.get("max_random_term_attempts", 64),
            64,
        )

        complexity_cfg = ComplexitySectionConfig.from_dict(
            _normalize_section_config(data.get("complexity"))
        )
        allowed_cfg = AllowedTermsConfig.from_dict(
            _normalize_section_config(data.get("allowed_terms"))
        )
        include_cfg = cast(
            IncludeTermsConfig,
            IncludeTermsConfig.from_dict(_normalize_section_config(data.get("include_terms"))),
        )
        exclude_cfg = cast(
            ExcludeTermsConfig,
            ExcludeTermsConfig.from_dict(_normalize_section_config(data.get("exclude_terms"))),
        )

        return cls(
            prompt_probability=prompt_probability,
            complexity=complexity_cfg,
            allowed_terms=allowed_cfg,
            include_terms=include_cfg,
            exclude_terms=exclude_cfg,
            max_random_term_attempts=max(1, max_random_term_attempts),
        )

    @classmethod
    def _from_legacy_dict(cls, data: Mapping[str, object]) -> "PromptFeatureExtractorConfig":
        max_cover = DistributionSpec._coerce_int_value(data.get("max_cover_terms", 48), 48)
        extra_range = _normalize_distribution_input(data.get("extra_allowed_range", (0, 3)))
        include_range = _normalize_distribution_input(data.get("include_terms_range", (0, 3)))
        exclude_range = _normalize_distribution_input(data.get("exclude_terms_range", (0, 4)))
        max_attempts = DistributionSpec._coerce_int_value(data.get("max_random_term_attempts", 64), 64)

        allowed_cfg = AllowedTermsConfig(
            probability=1.0,
            kwargs={},
            actual_terms=DistributionSpec.constant(max_cover),
            generated_terms=DistributionSpec.from_dict(extra_range),
            length=DistributionSpec.constant(max_cover),
            min_length=1,
            max_relative_length=1.0,
            force_expression_term=True,
        )

        include_cfg = IncludeTermsConfig(
            probability=1.0,
            kwargs={},
            count=DistributionSpec.from_dict(include_range),
            length=DistributionSpec.constant(max_cover),
            min_length=1,
            max_relative_length=1.0,
        )

        exclude_cfg = ExcludeTermsConfig(
            probability=1.0,
            kwargs={},
            count=DistributionSpec.from_dict(exclude_range),
            length=DistributionSpec.constant(max_cover),
            min_length=1,
            max_relative_length=1.0,
        )

        return cls(
            prompt_probability=1.0,
            complexity=ComplexitySectionConfig(),
            allowed_terms=allowed_cfg,
            include_terms=include_cfg,
            exclude_terms=exclude_cfg,
            max_random_term_attempts=max(1, max_attempts),
        )

    def get_probability(self, section: str) -> float:
        section_lower = section.lower()
        if section_lower == "prompt":
            return self.prompt_probability
        if section_lower == "complexity":
            return self.complexity.probability
        if section_lower == "allowed_terms":
            return self.allowed_terms.probability
        if section_lower == "include_terms":
            return self.include_terms.probability
        if section_lower == "exclude_terms":
            return self.exclude_terms.probability
        return 1.0

    def with_section_probabilities(self, overrides: Mapping[str, float]) -> "PromptFeatureExtractorConfig":
        updated = self
        for key, value in overrides.items():
            if key == "prompt":
                updated = replace(updated, prompt_probability=float(value))
                continue
            probability = float(value)
            key_lower = key.lower()
            if key_lower == "complexity":
                updated = replace(updated, complexity=replace(updated.complexity, probability=probability))
            elif key_lower == "allowed_terms":
                updated = replace(updated, allowed_terms=replace(updated.allowed_terms, probability=probability))
            elif key_lower == "include_terms":
                updated = replace(updated, include_terms=replace(updated.include_terms, probability=probability))
            elif key_lower == "exclude_terms":
                updated = replace(updated, exclude_terms=replace(updated.exclude_terms, probability=probability))
        return updated


@dataclass
class _ExpressionNode:
    token: str
    children: list["_ExpressionNode"] = field(default_factory=list)

    def to_prefix(self) -> list[str]:
        tokens = [self.token]
        for child in self.children:
            tokens.extend(child.to_prefix())
        return tokens

    def walk(self) -> Iterable["_ExpressionNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


class PromptFeatureExtractor:
    """Derive prompt control attributes from prefix expressions."""

    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,
        config: PromptFeatureExtractorConfig | None = None,
        variables: list[str] | None = None,
        skeleton_pool: SkeletonPool | None = None,
    ) -> None:
        self.engine = simplipy_engine
        self.tokenizer = tokenizer
        self.config = config or PromptFeatureExtractorConfig()
        self.variables = variables or self._infer_variables(tokenizer)
        if skeleton_pool is None:
            raise ValueError("PromptFeatureExtractor now requires a SkeletonPool for random term generation.")
        self.skeleton_pool = skeleton_pool

        operator_arity = getattr(self.engine, "operator_arity_compat", None)
        if operator_arity is None:
            operator_arity = getattr(self.engine, "operator_arity", {})
        self.operator_arity: dict[str, int] = dict(operator_arity)

        self.operator_aliases = getattr(self.engine, "operator_aliases", {})
        self.operator_tokens = list(self.operator_arity.keys())

    def extract(self, expression_tokens: Sequence[str]) -> PromptFeatures:
        expression_list = list(expression_tokens)
        if not expression_list:
            raise ValueError("Expression tokens cannot be empty.")

        root, next_index = self._parse_prefix_expression(expression_list, 0)
        if next_index != len(expression_list):
            raise ValueError("Expression tokens contain trailing data beyond a valid prefix expression.")

        all_subtrees = [node.to_prefix() for node in root.walk()]
        unique_subtrees = self._deduplicate_terms(all_subtrees)
        expression_length = len(expression_list)

        allowed_section_enabled = self._is_section_enabled(self.config.allowed_terms.probability)

        actual_terms = self._sample_actual_allowed_terms(unique_subtrees, expression_length, allowed_section_enabled)
        existing_keys = {tuple(term) for term in actual_terms}

        generated_terms = self._sample_generated_allowed_terms(expression_length, existing_keys, allowed_section_enabled)
        allowed_terms = self._deduplicate_terms(actual_terms + generated_terms)

        include_terms = self._sample_include_terms(allowed_terms, expression_length)
        exclude_terms = self._sample_exclude_terms(expression_length, allowed_terms, include_terms)

        return PromptFeatures(
            expression_tokens=expression_list,
            complexity=len(expression_list),
            allowed_terms=allowed_terms,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )

    @staticmethod
    def _is_section_enabled(probability: float) -> bool:
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        return random.random() < probability

    def _parse_prefix_expression(self, tokens: Sequence[str], idx: int) -> tuple[_ExpressionNode, int]:
        if idx >= len(tokens):
            raise ValueError("Unexpected end of tokens while parsing prefix expression.")

        token = tokens[idx]
        idx += 1

        normalized_token = self.operator_aliases.get(token, token)
        arity = self.operator_arity.get(normalized_token, 0)

        children: list[_ExpressionNode] = []
        for _ in range(arity):
            child, idx = self._parse_prefix_expression(tokens, idx)
            children.append(child)

        return _ExpressionNode(token, children), idx

    def _sample_actual_allowed_terms(
        self,
        subtrees: list[list[str]],
        expression_length: int,
        section_enabled: bool,
    ) -> list[list[str]]:
        if not section_enabled or not subtrees:
            return []

        cfg = self.config.allowed_terms
        min_len, max_len = self._resolve_length_bounds(cfg.min_length, cfg.max_relative_length, expression_length)

        terms: list[list[str]] = []
        used_keys: set[tuple[str, ...]] = set()
        root_term: list[str] | None = list(subtrees[0]) if subtrees else None

        total_requested = cfg.actual_terms.sample_int(minimum=0)

        if cfg.force_expression_term and root_term is not None:
            terms.append(root_term)
            used_keys.add(tuple(root_term))
            remaining = max(total_requested - 1, 0)
        else:
            remaining = total_requested

        candidate_subtrees = subtrees[1:] if root_term is not None else subtrees

        for _ in range(remaining):
            desired_len = cfg.length.sample_int(minimum=min_len, maximum=max_len)
            term = self._select_term_by_length(
                candidate_subtrees,
                desired_len,
                min_len,
                max_len,
                used_keys=used_keys,
                allow_duplicates=True,
            )
            if term is None:
                break
            terms.append(term)

        if not terms and cfg.force_expression_term and root_term is not None:
            terms.append(root_term)

        if not cfg.force_expression_term and root_term is not None:
            terms = [term for term in terms if term != root_term]

        if len(terms) > 1:
            random.shuffle(terms)

        return terms

    def _sample_generated_allowed_terms(
        self,
        expression_length: int,
        existing_keys: set[tuple[str, ...]],
        section_enabled: bool,
    ) -> list[list[str]]:
        if not section_enabled:
            return []

        cfg = self.config.allowed_terms
        count = cfg.generated_terms.sample_int(minimum=0)
        if count <= 0:
            return []

        min_len, max_len = self._resolve_length_bounds(cfg.min_length, cfg.max_relative_length, expression_length)

        generated: list[list[str]] = []
        max_attempts = max(self.config.max_random_term_attempts, count * 6)
        attempts = 0
        while len(generated) < count and attempts < max_attempts:
            attempts += 1
            desired_len = cfg.length.sample_int(minimum=min_len, maximum=max_len)
            term = self._generate_term_via_skeleton_pool(
                desired_length=desired_len,
                min_length=min_len,
                max_length=max_len,
            )
            if term is None:
                continue
            key = tuple(term)
            if key in existing_keys:
                continue
            generated.append(term)
            existing_keys.add(key)

        return generated

    def _sample_include_terms(self, allowed_terms: list[list[str]], expression_length: int) -> list[list[str]]:
        cfg = self.config.include_terms
        if not self._is_section_enabled(cfg.probability):
            return []

        min_len, max_len = self._resolve_length_bounds(cfg.min_length, cfg.max_relative_length, expression_length)
        candidate_pool = [term for term in allowed_terms if min_len <= len(term) <= max_len]
        if not candidate_pool:
            return []

        max_count = len(candidate_pool)
        target = cfg.count.sample_int(minimum=0, maximum=max_count)
        if target <= 0:
            return []

        selected: list[list[str]] = []
        used_keys: set[tuple[str, ...]] = set()
        for _ in range(target):
            desired_len = cfg.length.sample_int(minimum=min_len, maximum=max_len)
            term = self._select_term_by_length(
                candidate_pool,
                desired_len,
                min_len,
                max_len,
                used_keys=used_keys,
                allow_duplicates=False,
            )
            if term is None:
                break
            selected.append(term)

        return selected

    def _sample_exclude_terms(
        self,
        expression_length: int,
        allowed_terms: list[list[str]],
        include_terms: list[list[str]],
    ) -> list[list[str]]:
        cfg = self.config.exclude_terms
        if not self._is_section_enabled(cfg.probability):
            return []

        count = cfg.count.sample_int(minimum=0)
        if count <= 0:
            return []

        min_len, max_len = self._resolve_length_bounds(cfg.min_length, cfg.max_relative_length, expression_length)

        disallowed = {tuple(term) for term in allowed_terms}
        disallowed.update(tuple(term) for term in include_terms)

        exclusions: list[list[str]] = []
        exclusion_keys: set[tuple[str, ...]] = set()
        max_attempts = max(self.config.max_random_term_attempts, count * 8)
        attempts = 0
        while len(exclusions) < count and attempts < max_attempts:
            attempts += 1
            desired_len = cfg.length.sample_int(minimum=min_len, maximum=max_len)
            term = self._generate_term_via_skeleton_pool(
                desired_length=desired_len,
                min_length=min_len,
                max_length=max_len,
            )
            if term is None:
                continue
            key = tuple(term)
            if key in disallowed or key in exclusion_keys:
                continue
            exclusions.append(term)
            exclusion_keys.add(key)

        return exclusions

    def _generate_term_via_skeleton_pool(
        self,
        *,
        desired_length: int,
        min_length: int,
        max_length: int,
    ) -> list[str] | None:
        attempts = 0
        max_attempts = max(1, self.config.max_random_term_attempts)
        fallback: list[str] | None = None

        while attempts < max_attempts:
            attempts += 1
            try:
                skeleton, _, _ = self.skeleton_pool.sample_skeleton(new=True, decontaminate=False)
            except NoValidSampleFoundError:
                continue

            tokens = list(skeleton)
            if not tokens:
                continue

            try:
                root, _ = self._parse_prefix_expression(tokens, 0)
            except ValueError:
                continue

            nodes = list(root.walk())
            if not nodes:
                continue

            random.shuffle(nodes)
            eligible: list[list[str]] = []
            fallback_candidates: list[list[str]] = []

            for node in nodes:
                term = node.to_prefix()
                length = len(term)
                if length < min_length or length > max_length:
                    continue
                if length <= desired_length:
                    eligible.append(term)
                else:
                    fallback_candidates.append(term)

            if eligible:
                return list(random.choice(eligible))
            if fallback_candidates and fallback is None:
                fallback = list(random.choice(fallback_candidates))

        return fallback

    @staticmethod
    def _resolve_length_bounds(min_length: int, max_relative: float, expression_length: int) -> tuple[int, int]:
        min_tokens = max(1, min_length)
        if max_relative <= 0:
            max_tokens = expression_length
        else:
            max_tokens = int(math.floor(expression_length * max_relative))
            max_tokens = max(min_tokens, max_tokens)
        max_tokens = max(min_tokens, min(expression_length, max_tokens))
        return min_tokens, max_tokens

    @staticmethod
    def _select_term_by_length(
        candidates: Sequence[Sequence[str]],
        desired_length: int,
        min_length: int,
        max_length: int,
        *,
        used_keys: set[tuple[str, ...]] | None = None,
        allow_duplicates: bool = True,
    ) -> list[str] | None:
        available: list[Sequence[str]] = []
        duplicate_candidates: list[Sequence[str]] = []

        for term in candidates:
            length = len(term)
            if length < min_length or length > max_length:
                continue
            key = tuple(term)
            if used_keys and key in used_keys:
                duplicate_candidates.append(term)
                continue
            available.append(term)

        pool = available
        if not pool and allow_duplicates:
            pool = duplicate_candidates

        if not pool:
            return None

        preferred = [term for term in pool if len(term) <= desired_length]
        selection_pool = preferred if preferred else pool
        choice = list(random.choice(selection_pool))

        if used_keys is not None:
            used_keys.add(tuple(choice))

        return choice

    @staticmethod
    def _deduplicate_terms(terms: Iterable[Sequence[str]]) -> list[list[str]]:
        seen: dict[tuple[str, ...], list[str]] = {}
        for term in terms:
            key = tuple(term)
            if key not in seen:
                seen[key] = list(term)
        return list(seen.values())

    @staticmethod
    def _infer_variables(tokenizer: Tokenizer) -> list[str]:
        variables: set[str] = set()
        for token in tokenizer:
            if token.startswith("x") and token[1:].isdigit():
                variables.add(token)
        return sorted(variables)
