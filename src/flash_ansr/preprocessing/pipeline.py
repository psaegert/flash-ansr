"""Preprocessing pipeline responsible for batch formatting and prompt-prefix serialization."""
from __future__ import annotations  # necessary for type annotations

from typing import Any

import numpy as np
from simplipy import SimpliPyEngine

from symbolic_data import LampleChartonCatalog
from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing.prompt_serialization import PromptSerializer
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.numeric import merge_numeric_sequence


class FlashANSRPreprocessor:
    """Format batch inputs and serialize decoding prompt prefixes."""

    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        tokenizer: Tokenizer,
        catalog: LampleChartonCatalog | None = None,
        *,
        prompt_config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.tokenizer = tokenizer
        self.catalog = catalog
        self._rng = rng if rng is not None else np.random.default_rng()

        # Carried opaquely so dataset workers can rebuild an identical preprocessor
        # (data.py deep-copies it; streaming.py passes it back into this constructor).
        self.prompt_config: dict[str, Any] = prompt_config if prompt_config is not None else {}

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
            unwrapped. ``None`` or a non-mapping config yields default settings.
        simplipy_engine : SimpliPyEngine
            Engine used to manipulate and evaluate symbolic expressions.
        tokenizer : Tokenizer
            Tokenizer used to serialize prompt prefixes and expressions.
        catalog : LampleChartonCatalog, optional
            Catalog supplying variables to dataset workers rebuilding this preprocessor.
        rng : numpy.random.Generator, optional
            Random generator. Defaults to a fresh generator.

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
        """Format a batch instance-by-instance.

        Each instance in ``batch`` is formatted (adding ``input_num`` / ``prompt_mask``), then
        the results are re-stacked back into per-key lists.

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

    def serialize_prompt_prefix(self, *, complexity: float | int | None = None) -> dict[str, Any]:
        """Serialize the decoding prompt prefix: ``<bos>``, an optional complexity block, ``<expression>``.

        Parameters
        ----------
        complexity : float or int, optional
            Target complexity in **simplipy mu** (roughly 1e3-1e6, NOT a token count), or ``None``.

        Returns
        -------
        dict[str, Any]
            As :meth:`PromptSerializer.serialize_prompt_prefix`.
        """
        return self._serializer.serialize_prompt_prefix(complexity=complexity)

    def _format_single(self, instance: dict[str, Any]) -> dict[str, Any]:
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
        }

        existing_numeric = instance.get("input_num")
        if existing_numeric is not None:
            serialized["input_num"] = merge_numeric_sequence(existing_numeric, serialized["input_num"])

        return serialized

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
