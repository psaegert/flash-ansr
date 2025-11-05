from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from simplipy import SimpliPyEngine

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.expressions.skeleton_pool import SkeletonPool
from flash_ansr.preprocess_features import (
    PromptFeatureExtractor,
    PromptFeatureExtractorConfig,
    PromptFeatures,
)
from flash_ansr.utils.config_io import load_config


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
        input_ids = batch.get('input_ids')
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

    def _format_single(self, instance: dict[str, Any]) -> dict[str, Any]:
        if self._prompt_enabled and self._feature_extractor is not None and self._should_include("prompt"):
            skeleton_tokens = instance.get('skeletons')
            if skeleton_tokens is None:
                skeleton_tokens = instance.get('skeleton')
            if skeleton_tokens is None:
                return self._format_single_fallback(instance)

            if isinstance(skeleton_tokens, np.ndarray):
                skeleton_tokens = skeleton_tokens.tolist()

            skeleton_tokens = list(self._ensure_iterable_of_str(skeleton_tokens))

            try:
                features = self._feature_extractor.extract(skeleton_tokens)
                return self._serialize_prompt(features)
            except ValueError:
                return self._format_single_fallback(instance)

        return self._format_single_fallback(instance)

    # ------------------------------------------------------------------
    # Legacy formatting path
    # ------------------------------------------------------------------
    def _format_single_fallback(self, instance: dict[str, Any]) -> dict[str, Any]:
        input_ids = instance['input_ids']
        if hasattr(input_ids, 'detach') and callable(getattr(input_ids, 'detach')):
            input_ids = input_ids.detach().cpu().tolist()
        elif hasattr(input_ids, 'tolist') and callable(getattr(input_ids, 'tolist')):
            input_ids = input_ids.tolist()
        elif isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()

        complexity = len(input_ids)
        modified_input_ids = input_ids
        input_num = [np.nan] * len(modified_input_ids)

        return {
            'complexity': complexity,
            'input_ids': modified_input_ids,
            'input_num': input_num,
            'prompt_mask': [False] * len(modified_input_ids),
            'prompt_metadata': {
                'allowed_terms': [],
                'include_terms': [],
                'exclude_terms': [],
            },
        }

    # ------------------------------------------------------------------
    # Prompt serialization helpers
    # ------------------------------------------------------------------
    def _serialize_prompt(self, features: PromptFeatures) -> dict[str, Any]:
        tokens: list[str] = []
        numeric_values: list[float] = []
        prompt_mask: list[bool] = []

        def append(token: str, *, value: float = np.nan, is_prompt: bool = False) -> None:
            tokens.append(token)
            numeric_values.append(value)
            prompt_mask.append(is_prompt)

        append('<bos>')

        prompt_context = self._build_prompt_content(features)
        append('<prompt>', is_prompt=True)
        for token, value in prompt_context:
            append(token, value=value, is_prompt=True)
        append('</prompt>', is_prompt=True)

        append('<expression>')

        for token in features.expression_tokens:
            append(token)

        append('</expression>')

        append('<eos>')

        try:
            input_ids = [self.tokenizer[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(f"Token '{exc.args[0]}' missing from tokenizer vocabulary while serializing prompt.") from exc

        return {
            'complexity': features.complexity,
            'input_ids': input_ids,
            'input_num': numeric_values,
            'prompt_mask': prompt_mask,
            'prompt_metadata': {
                'allowed_terms': features.allowed_terms,
                'include_terms': features.include_terms,
                'exclude_terms': features.exclude_terms,
            },
        }

    def serialize_prompt_prefix(
        self,
        *,
        complexity: float | int | None = None,
        allowed_terms: Iterable[Sequence[Any]] | None = None,
        include_terms: Iterable[Sequence[Any]] | None = None,
        exclude_terms: Iterable[Sequence[Any]] | None = None,
    ) -> dict[str, Any]:
        tokens: list[str] = ['<bos>']
        numeric_values: list[float] = [np.nan]
        prompt_mask: list[bool] = [False]

        normalized_allowed = self._normalize_prompt_terms_collection(allowed_terms)
        normalized_include = self._normalize_prompt_terms_collection(include_terms)
        normalized_exclude = self._normalize_prompt_terms_collection(exclude_terms)

        metadata = {
            'allowed_terms': normalized_allowed,
            'include_terms': normalized_include,
            'exclude_terms': normalized_exclude,
        }

        has_prompt_content = (
            complexity is not None
            or bool(normalized_allowed)
            or bool(normalized_include)
            or bool(normalized_exclude)
        )

        required_prompt_tokens: set[str] = set()
        if has_prompt_content:
            required_prompt_tokens.update({'<prompt>', '</prompt>'})
            if complexity is not None:
                required_prompt_tokens.update({'<complexity>', '<float>', '</complexity>'})
            if normalized_allowed:
                required_prompt_tokens.update({'<allowed_term>', '</allowed_term>'})
            if normalized_include:
                required_prompt_tokens.update({'<include_term>', '</include_term>'})
            if normalized_exclude:
                required_prompt_tokens.update({'<exclude_term>', '</exclude_term>'})

        missing_prompt_tokens = [token for token in required_prompt_tokens if token not in self.tokenizer]
        emit_prompt = has_prompt_content and not missing_prompt_tokens

        prompt_tokens: list[tuple[str, float, bool]] = []

        if emit_prompt:
            prompt_tokens.append(('<prompt>', np.nan, True))

            if complexity is not None:
                prompt_tokens.append(('<complexity>', np.nan, True))
                prompt_tokens.append(('<float>', float(complexity), True))
                prompt_tokens.append(('</complexity>', np.nan, True))

            for section_token, terms in (
                ('allowed_term', normalized_allowed),
                ('include_term', normalized_include),
                ('exclude_term', normalized_exclude),
            ):
                for term in terms:
                    prompt_tokens.append((f'<{section_token}>', np.nan, True))
                    for token in term:
                        prompt_tokens.append((token, np.nan, True))
                    prompt_tokens.append((f'</{section_token}>', np.nan, True))

            prompt_tokens.append(('</prompt>', np.nan, True))

        for token, value, is_prompt in prompt_tokens:
            tokens.append(token)
            numeric_values.append(value)
            prompt_mask.append(is_prompt)

        missing_tokens: list[str] = list(missing_prompt_tokens)

        if '<bos>' not in self.tokenizer:
            missing_tokens.append('<bos>')

        if '<expression>' in self.tokenizer:
            tokens.append('<expression>')
            numeric_values.append(np.nan)
            prompt_mask.append(False)
        else:
            missing_tokens.append('<expression>')

        try:
            input_ids = [self.tokenizer[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(f"Token '{exc.args[0]}' missing from tokenizer vocabulary while serializing prompt prefix.") from exc

        return {
            'input_ids': input_ids,
            'input_num': numeric_values,
            'prompt_mask': prompt_mask,
            'prompt_metadata': metadata,
            'prompt_disabled': not emit_prompt,
            'missing_tokens': missing_tokens,
        }

    def _build_prompt_content(self, features: PromptFeatures) -> list[tuple[str, float]]:
        content: list[tuple[str, float]] = []

        def extend_section(tokens: Iterable[str]) -> None:
            for token in tokens:
                content.append((token, np.nan))

        if self._should_include('complexity'):
            content.append(('<complexity>', np.nan))
            content.append(('<float>', float(features.complexity)))
            content.append(('</complexity>', np.nan))

        if self._should_include('allowed_terms'):
            for term in features.allowed_terms:
                content.append(('<allowed_term>', np.nan))
                extend_section(term)
                content.append(('</allowed_term>', np.nan))

        if self._should_include('include_terms') and features.include_terms:
            for term in features.include_terms:
                content.append(('<include_term>', np.nan))
                extend_section(term)
                content.append(('</include_term>', np.nan))

        if self._should_include('exclude_terms') and features.exclude_terms:
            for term in features.exclude_terms:
                content.append(('<exclude_term>', np.nan))
                extend_section(term)
                content.append(('</exclude_term>', np.nan))

        return content

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
    def _normalize_prompt_terms_collection(terms: Iterable[Sequence[Any]] | None) -> list[list[str]]:
        if not terms:
            return []

        normalized: list[list[str]] = []
        for term in terms:
            if isinstance(term, str):
                raise TypeError("Prompt term collections must be sequences of tokens, not raw strings.")

            normalized_term = [str(token) for token in term]
            if not normalized_term:
                continue
            normalized.append(normalized_term)
        return normalized

    @staticmethod
    def _select_batch_item(value: Any, index: int) -> Any:
        try:
            if isinstance(value, (list, tuple)):
                return value[index]
            if isinstance(value, np.ndarray):
                return value[index]
            # Torch tensors support __getitem__ but we avoid importing torch here
            return value[index]  # type: ignore[index]
        except (TypeError, KeyError, IndexError):
            return value
