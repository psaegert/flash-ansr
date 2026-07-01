"""Preprocessing pipeline responsible for prompt enrichment."""
from __future__ import annotations  # necessary for type annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from simplipy import SimpliPyEngine

from symbolic_data import LampleChartonCatalog
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
class FlashANSRPreprocessorConfig:
    """Configuration describing how prompts are serialized."""

    prompt_feature: PromptFeatureExtractorConfig = field(default_factory=PromptFeatureExtractorConfig)

    @classmethod
    def from_dict(
        cls,
        data: "FlashANSRPreprocessorConfig" | dict[str, Any] | None,
    ) -> "FlashANSRPreprocessorConfig":
        """Build a :class:`FlashANSRPreprocessorConfig` from a dict or an existing instance.

        The ``prompt_feature`` entry is parsed into a :class:`PromptFeatureExtractorConfig` (a
        string value is treated as a config path and loaded first). An optional ``section_probs``
        mapping overrides individual section probabilities via
        :meth:`PromptFeatureExtractorConfig.with_section_probabilities`.

        Parameters
        ----------
        data : FlashANSRPreprocessorConfig or dict or None
            An existing instance (passed through), ``None`` (defaults), or a dict of options.

        Returns
        -------
        FlashANSRPreprocessorConfig
            The parsed preprocessor configuration.

        Raises
        ------
        TypeError
            If ``data`` is neither ``None``, an instance, nor a dict.
        """
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
        catalog: LampleChartonCatalog | None = None,
        *,
        prompt_config: FlashANSRPreprocessorConfig | dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.tokenizer = tokenizer
        self.catalog = catalog
        self._rng = rng if rng is not None else np.random.default_rng()

        self.prompt_config = FlashANSRPreprocessorConfig.from_dict(prompt_config)
        self._prompt_enabled = (
            catalog is not None
            and self.prompt_config.prompt_feature.prompt_probability > 0
        )

        self._feature_extractor: PromptFeatureExtractor | None = None
        if self._prompt_enabled:
            self._feature_extractor = PromptFeatureExtractor(
                simplipy_engine=simplipy_engine,
                tokenizer=tokenizer,
                config=self.prompt_config.prompt_feature,
                catalog=catalog,
                rng=self._rng,
            )

        self._serializer = PromptSerializer(tokenizer)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | str | None,
        *,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,
        catalog: LampleChartonCatalog | None = None,
        rng: np.random.Generator | None = None,
    ) -> "FlashANSRPreprocessor":
        """Construct a preprocessor from a config plus the required runtime dependencies.

        Parameters
        ----------
        config : dict[str, Any] or str or None
            Config mapping or path to a config file. A top-level ``"preprocessor"`` key is
            unwrapped, and its ``"prompt"`` section configures prompt enrichment. ``None`` or a
            non-mapping config yields default (prompt-disabled) settings.
        simplipy_engine : SimpliPyEngine
            Engine used to manipulate and evaluate symbolic expressions.
        tokenizer : Tokenizer
            Tokenizer used to serialize prompts and expressions.
        catalog : LampleChartonCatalog, optional
            Catalog enabling prompt-feature extraction; prompts are only emitted when a catalog
            is supplied and the configured prompt probability is positive.
        rng : numpy.random.Generator, optional
            Random generator driving stochastic prompt inclusion. Defaults to a fresh generator.

        Returns
        -------
        FlashANSRPreprocessor
            The configured preprocessor.
        """
        config_ = load_config(config)

        if isinstance(config_, dict) and "preprocessor" in config_.keys():
            config_ = config_["preprocessor"]

        if not isinstance(config_, dict):
            config_ = {}

        prompt_cfg = config_.get("prompt")

        return cls(
            simplipy_engine=simplipy_engine,
            tokenizer=tokenizer,
            catalog=catalog,
            prompt_config=prompt_cfg,
            rng=rng,
        )

    def format(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Format a batch instance-by-instance, optionally enriching it with prompt metadata.

        Each instance in ``batch`` is formatted (adding ``input_num`` / ``prompt_mask`` /
        ``prompt_metadata`` and, when enabled, a sampled prompt prefix), then the results are
        re-stacked back into per-key lists.

        Parameters
        ----------
        batch : dict[str, Any]
            A batch mapping keys to per-instance sequences; must contain ``"input_ids"``.

        Returns
        -------
        dict[str, Any]
            The batch with formatted fields. Returned unchanged if ``"input_ids"`` is absent or
            the batch is empty.
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            return batch

        batch_size = len(input_ids)

        formatted_instances: list[dict[str, Any]] = []
        for idx in range(batch_size):
            instance = {key: self._select_batch_item(value, idx) for key, value in batch.items()}
            formatted_instances.append(self._format_single(instance))

        if not formatted_instances:
            return batch

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
        """Serialize an explicit prompt prefix constraining generation.

        Builds the token prefix (starting from ``<bos>``) that encodes the requested constraints,
        emitting the ``<prompt>`` block only when the tokenizer defines the needed special tokens.

        Parameters
        ----------
        complexity : float or int, optional
            Target expression complexity to encode in the prompt.
        allowed_terms : iterable of sequences, optional
            Terms the generated expression is restricted to.
        include_terms : iterable of sequences, optional
            Terms that must appear in the generated expression.
        exclude_terms : iterable of sequences, optional
            Terms that must not appear in the generated expression.

        Returns
        -------
        dict[str, Any]
            The serialized prefix with ``input_ids``, ``input_num``, ``prompt_mask`` and
            ``prompt_metadata`` entries.
        """
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
        return self._rng.random() < probability

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
