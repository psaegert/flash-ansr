"""Preprocessing pipeline responsible for prompt enrichment."""
from __future__ import annotations  # necessary for type annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from simplipy import SimpliPyEngine

from flash_ansr.expressions.skeleton_pool import SkeletonPool
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing.feature_extractor import (
    PromptFeatureExtractor,
    PromptFeatureExtractorConfig,
)
from flash_ansr.preprocessing.prompt_serialization import PromptSerializer
from flash_ansr.preprocessing.schemas import PromptFeatures
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.numeric import merge_numeric_sequence


@dataclass
class FlashASNRPreprocessorConfig:
    """Configuration describing how prompts are serialized."""

    prompt_feature: PromptFeatureExtractorConfig = field(default_factory=PromptFeatureExtractorConfig)

    @classmethod
    def from_dict(
        cls,
        data: "FlashASNRPreprocessorConfig" | dict[str, Any] | None,
    ) -> "FlashASNRPreprocessorConfig":
        if isinstance(data, cls):
            return data
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for preprocessor config, got {type(data).__name__}")

        feature_cfg = data.get("prompt_feature")
        if isinstance(feature_cfg, str):
            feature_cfg = load_config(feature_cfg)

        prompt_feature = PromptFeatureExtractorConfig.from_dict(feature_cfg)

        section_prob_overrides: dict[str, float] = {}
        section_probs_raw = data.get("section_probs")
        if isinstance(section_probs_raw, dict):
            for key, value in section_probs_raw.items():
                try:
                    section_prob_overrides[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if section_prob_overrides:
                prompt_feature = prompt_feature.with_section_probabilities(section_prob_overrides)

        return cls(prompt_feature=prompt_feature)


class FlashANSRPreprocessor:
    """Format batch inputs and optionally enrich them with prompt metadata."""

    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,
        skeleton_pool: SkeletonPool | None = None,
        *,
        prompt_config: FlashASNRPreprocessorConfig | dict[str, Any] | None = None,
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.tokenizer = tokenizer
        self.skeleton_pool = skeleton_pool

        self.prompt_config = FlashASNRPreprocessorConfig.from_dict(prompt_config)
        self._prompt_enabled = (
            skeleton_pool is not None
            and self.prompt_config.prompt_feature.prompt_probability > 0
        )

        self._feature_extractor: PromptFeatureExtractor | None = None
        if self._prompt_enabled:
            self._feature_extractor = PromptFeatureExtractor(
                simplipy_engine=simplipy_engine,
                tokenizer=tokenizer,
                config=self.prompt_config.prompt_feature,
                skeleton_pool=skeleton_pool,
            )

        self._serializer = PromptSerializer(tokenizer)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | str | None,
        *,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,
        skeleton_pool: SkeletonPool | None = None,
    ) -> "FlashANSRPreprocessor":
        config_ = load_config(config)

        if isinstance(config_, dict) and "preprocessor" in config_.keys():
            config_ = config_["preprocessor"]

        if not isinstance(config_, dict):
            config_ = {}

        prompt_cfg = config_.get("prompt")

        return cls(
            simplipy_engine=simplipy_engine,
            tokenizer=tokenizer,
            skeleton_pool=skeleton_pool,
            prompt_config=prompt_cfg,
        )

    def format(self, batch: dict[str, Any]) -> dict[str, Any]:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return batch

        batch_size = len(input_ids)

        formatted_instances: list[dict[str, Any]] = []
        for idx in range(batch_size):
            instance = {key: self._select_batch_item(value, idx) for key, value in batch.items()}
            formatted_instances.append(self._format_single(instance))

        for key in formatted_instances[0].keys():
            batch[key] = [instance[key] for instance in formatted_instances]

        return batch

    def serialize_prompt_prefix(
        self,
        *,
        complexity: float | int | None = None,
        allowed_terms: Iterable[Sequence[Any]] | None = None,
        include_terms: Iterable[Sequence[Any]] | None = None,
        exclude_terms: Iterable[Sequence[Any]] | None = None,
    ) -> dict[str, Any]:
        return self._serializer.serialize_prompt_prefix(
            complexity=complexity,
            allowed_terms=allowed_terms,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )

    def _format_single(self, instance: dict[str, Any]) -> dict[str, Any]:
        if self._prompt_enabled and self._feature_extractor is not None and self._should_include("prompt"):
            skeleton_tokens = instance.get("skeletons")
            if skeleton_tokens is None:
                skeleton_tokens = instance.get("skeleton")
            if skeleton_tokens is None:
                return self._format_single_fallback(instance)

            if isinstance(skeleton_tokens, np.ndarray):
                skeleton_tokens = skeleton_tokens.tolist()

            skeleton_tokens = list(self._ensure_iterable_of_str(skeleton_tokens))

            try:
                features = self._feature_extractor.extract(skeleton_tokens)
                serialized = self._serialize_prompt(features)
                existing_numeric = instance.get("input_num")
                if existing_numeric is not None:
                    serialized["input_num"] = merge_numeric_sequence(existing_numeric, serialized["input_num"])
                return serialized
            except ValueError:
                return self._format_single_fallback(instance)

        return self._format_single_fallback(instance)

    def _serialize_prompt(self, features: PromptFeatures) -> dict[str, Any]:
        include_complexity = self._should_include("complexity")
        include_allowed = self._should_include("allowed_terms")
        include_include = self._should_include("include_terms")
        include_exclude = self._should_include("exclude_terms")

        return self._serializer.serialize_prompt(
            features,
            include_complexity=include_complexity,
            include_allowed_terms=include_allowed,
            include_include_terms=include_include,
            include_exclude_terms=include_exclude,
        )

    # ------------------------------------------------------------------
    # Legacy formatting path
    # ------------------------------------------------------------------
    def _format_single_fallback(self, instance: dict[str, Any]) -> dict[str, Any]:
        input_ids = instance["input_ids"]
        if hasattr(input_ids, "detach") and callable(getattr(input_ids, "detach")):
            input_ids = input_ids.detach().cpu().tolist()
        elif hasattr(input_ids, "tolist") and callable(getattr(input_ids, "tolist")):
            input_ids = input_ids.tolist()
        elif isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()

        complexity = len(input_ids)
        modified_input_ids = input_ids
        input_num = [np.nan] * len(modified_input_ids)

        serialized = {
            "complexity": complexity,
            "input_ids": modified_input_ids,
            "input_num": input_num,
            "prompt_mask": [False] * len(modified_input_ids),
            "prompt_metadata": {
                "allowed_terms": [],
                "include_terms": [],
                "exclude_terms": [],
            },
        }

        existing_numeric = instance.get("input_num")
        if existing_numeric is not None:
            serialized["input_num"] = merge_numeric_sequence(existing_numeric, serialized["input_num"])

        return serialized

    def _should_include(self, section: str) -> bool:
        probability = self.prompt_config.prompt_feature.get_probability(section)
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        return random.random() < probability

    @staticmethod
    def _ensure_iterable_of_str(tokens: Iterable[Any]) -> Iterable[str]:
        return [str(token) for token in tokens]

    @staticmethod
    def _select_batch_item(value: Any, index: int) -> Any:
        try:
            if isinstance(value, (list, tuple)):
                return value[index]
            if isinstance(value, np.ndarray):
                return value[index]
            return value[index]  # type: ignore[index]
        except (TypeError, KeyError, IndexError):
            return value
