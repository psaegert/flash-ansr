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
from flash_ansr.utils import load_config


@dataclass
class PromptSectionProbabilities:
    """Probabilities controlling which prompt sections are emitted."""

    prompt: float = 1.0
    complexity: float = 1.0
    allowed_terms: float = 1.0
    include_terms: float = 1.0
    exclude_terms: float = 1.0

    @classmethod
    def from_dict(
        cls,
        data: "PromptSectionProbabilities" | dict[str, Any] | None,
    ) -> "PromptSectionProbabilities":
        if isinstance(data, cls):
            return data
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for section probabilities, got {type(data).__name__}")
        return cls(
            prompt=float(data.get("prompt", 1.0)),
            complexity=float(data.get("complexity", 1.0)),
            allowed_terms=float(data.get("allowed_terms", 1.0)),
            include_terms=float(data.get("include_terms", 1.0)),
            exclude_terms=float(data.get("exclude_terms", 1.0)),
        )


@dataclass
class FlashASNRPreprocessorConfig:
    """Configuration describing how prompts are serialized."""

    section_probs: PromptSectionProbabilities = field(default_factory=PromptSectionProbabilities)
    prompt_feature: PromptFeatureExtractorConfig = field(default_factory=PromptFeatureExtractorConfig)
    attach_prompt_tags: bool = True
    attach_expression_tags: bool = True

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

        section_probs = PromptSectionProbabilities.from_dict(data.get("section_probs"))

        feature_cfg = data.get("prompt_feature")
        if isinstance(feature_cfg, PromptFeatureExtractorConfig):
            prompt_feature = feature_cfg
        else:
            prompt_feature = PromptFeatureExtractorConfig(**(feature_cfg or {}))

        return cls(
            section_probs=section_probs,
            prompt_feature=prompt_feature,
            attach_prompt_tags=bool(data.get("attach_prompt_tags", True)),
            attach_expression_tags=bool(data.get("attach_expression_tags", True)),
        )


class FlashASNRPreprocessor:
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
            and self.prompt_config.section_probs.prompt > 0
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
    ) -> "FlashASNRPreprocessor":
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
        if not isinstance(batch['input_ids'][0], list):
            # The input is a single instance
            for k, modified_value in self._format_single(batch).items():
                batch[k] = modified_value

            return batch

        # The input is a batch of instances
        new_fields: set[str] = set()
        for i, instance in enumerate(zip(*batch.values())):
            for k, modified_value in self._format_single(dict(zip(batch.keys(), instance))).items():
                if k not in batch:
                    new_fields.add(k)
                    batch[k] = []

                if k in new_fields:
                    batch[k].append(modified_value)
                else:
                    batch[k][i] = modified_value

        return batch

    def format_complexity(self, input_ids: list[int], complexity: float | int) -> tuple[list[int], list[float]]:
        # Legacy interface retained for inference utilities. The returned sequence matches
        # the prepended control tokens applied during formatting when prompts are disabled.
        modified_input_ids = [self.tokenizer['<ctrl_complexity>'], self.tokenizer['<constant>']] + input_ids

        input_num = [np.nan] * len(modified_input_ids)
        input_num[1] = complexity

        return modified_input_ids, input_num

    def _format_single(self, instance: dict[str, Any]) -> dict[str, Any]:
        if self._prompt_enabled and self._feature_extractor is not None and self._should_include("prompt"):
            skeleton_tokens = instance.get('skeletons')
            if skeleton_tokens is None:
                return self._format_single_legacy(instance)

            if isinstance(skeleton_tokens, np.ndarray):
                skeleton_tokens = skeleton_tokens.tolist()

            skeleton_tokens = list(self._ensure_iterable_of_str(skeleton_tokens))

            try:
                features = self._feature_extractor.extract(skeleton_tokens)
                return self._serialize_prompt(features)
            except ValueError:
                return self._format_single_legacy(instance)

        return self._format_single_legacy(instance)

    # ------------------------------------------------------------------
    # Legacy formatting path
    # ------------------------------------------------------------------
    def _format_single_legacy(self, instance: dict[str, Any]) -> dict[str, Any]:
        input_ids = instance['input_ids']
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()

        complexity = len(input_ids)
        modified_input_ids = input_ids
        input_num = [np.nan] * len(modified_input_ids)

        return {
            'complexities': complexity,
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

        prompt_tokens_added = False
        if self.prompt_config.attach_prompt_tags:
            prompt_context = self._build_prompt_content(features)
            if prompt_context:
                append('<prompt>', is_prompt=True)
                for token, value in prompt_context:
                    append(token, value=value, is_prompt=True)
                    prompt_tokens_added = True
                append('</prompt>', is_prompt=True)
        else:
            prompt_context = self._build_prompt_content(features)
            for token, value in prompt_context:
                append(token, value=value, is_prompt=True)
                prompt_tokens_added = True

        if not prompt_tokens_added:
            # Ensure the prompt mask remains aligned even if no prompt tokens are emitted.
            prompt_mask = [False] * len(tokens)

        if self.prompt_config.attach_expression_tags:
            append('<expression>')

        for token in features.expression_tokens:
            append(token)

        if self.prompt_config.attach_expression_tags:
            append('</expression>')

        append('<eos>')

        try:
            input_ids = [self.tokenizer[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(f"Token '{exc.args[0]}' missing from tokenizer vocabulary while serializing prompt.") from exc

        return {
            'complexities': features.complexity,
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

        prompt_tokens: list[tuple[str, float, bool]] = []

        if self.prompt_config.attach_prompt_tags:
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

        if self.prompt_config.attach_prompt_tags:
            prompt_tokens.append(('</prompt>', np.nan, True))

        if not prompt_tokens and self.prompt_config.attach_prompt_tags:
            # Emit empty prompt block to maintain fixed structure
            prompt_tokens = [('<prompt>', np.nan, True), ('</prompt>', np.nan, True)]

        for token, value, is_prompt in prompt_tokens:
            tokens.append(token)
            numeric_values.append(value)
            prompt_mask.append(is_prompt)

        if self.prompt_config.attach_expression_tags:
            tokens.append('<expression>')
            numeric_values.append(np.nan)
            prompt_mask.append(False)

        try:
            input_ids = [self.tokenizer[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(f"Token '{exc.args[0]}' missing from tokenizer vocabulary while serializing prompt prefix.") from exc

        return {
            'input_ids': input_ids,
            'input_num': numeric_values,
            'prompt_mask': prompt_mask,
            'prompt_metadata': metadata,
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
        probability = getattr(self.prompt_config.section_probs, section, 1.0)
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
