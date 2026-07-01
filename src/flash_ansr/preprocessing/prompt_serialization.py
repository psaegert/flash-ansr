"""Prompt serialization helpers shared across preprocessing and inference."""
from typing import Any, Iterable, Sequence

import numpy as np

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing.schemas import PromptFeatures, PromptPrefix


class PromptSerializer:
    """Convert prompt features into token sequences consumable by the model."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def serialize_prompt(
        self,
        features: PromptFeatures,
        *,
        include_complexity: bool,
        include_allowed_terms: bool,
        include_include_terms: bool,
        include_exclude_terms: bool,
    ) -> dict[str, Any]:
        """Serialize a full training example (prompt plus expression) into token ids.

        Emits ``<bos>``, an optional ``<prompt>`` block (complexity and the enabled term sections),
        the ``<expression>`` body, and a trailing ``<eos>``.

        Parameters
        ----------
        features : PromptFeatures
            The extracted prompt features and the expression tokens to serialize.
        include_complexity : bool
            Whether to emit the complexity sub-section.
        include_allowed_terms : bool
            Whether to emit the allowed-terms sub-section (only if features has allowed terms).
        include_include_terms : bool
            Whether to emit the include-terms sub-section (only if features has include terms).
        include_exclude_terms : bool
            Whether to emit the exclude-terms sub-section (only if features has exclude terms).

        Returns
        -------
        dict[str, Any]
            A dict with ``complexity``, ``input_ids``, ``input_num``, ``prompt_mask`` and
            ``prompt_metadata`` (the allowed / include / exclude term lists).

        Raises
        ------
        KeyError
            If any emitted token is missing from the tokenizer vocabulary.
        """
        tokens: list[str] = []
        numeric_values: list[float] = []
        prompt_mask: list[bool] = []

        def append(token: str, *, value: float = float("nan"), is_prompt: bool = False) -> None:
            tokens.append(token)
            numeric_values.append(value)
            prompt_mask.append(is_prompt)

        append("<bos>")

        append("<prompt>", is_prompt=True)
        if include_complexity:
            append("<complexity>", is_prompt=True)
            append("<float>", value=float(features.complexity), is_prompt=True)
            append("</complexity>", is_prompt=True)

        if include_allowed_terms and features.allowed_terms:
            self._append_term_section("allowed", features.allowed_terms, append)

        if include_include_terms and features.include_terms:
            self._append_term_section("include", features.include_terms, append)

        if include_exclude_terms and features.exclude_terms:
            self._append_term_section("exclude", features.exclude_terms, append)

        append("</prompt>", is_prompt=True)

        append("<expression>")
        for token in features.expression_tokens:
            append(token)
        append("</expression>")
        append("<eos>")

        try:
            input_ids = [self.tokenizer[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(
                f"Token '{exc.args[0]}' missing from tokenizer vocabulary while serializing prompt."
            ) from exc

        return {
            "complexity": features.complexity,
            "input_ids": input_ids,
            "input_num": numeric_values,
            "prompt_mask": prompt_mask,
            "prompt_metadata": {
                "allowed_terms": features.allowed_terms,
                "include_terms": features.include_terms,
                "exclude_terms": features.exclude_terms,
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
        """Serialize a decoding prompt prefix (prompt block plus the opening ``<expression>``).

        Builds the prefix used to condition generation: ``<bos>``, an optional ``<prompt>`` block
        (emitted only when there is prompt content and all required prompt tokens exist in the
        vocabulary), and a trailing ``<expression>`` opener. Missing vocabulary tokens are reported
        rather than raised, and the prompt block is skipped when any of its tokens are absent.

        Parameters
        ----------
        complexity : float or int, optional
            Target expression complexity to encode, or ``None`` to omit it.
        allowed_terms : Iterable[Sequence[Any]], optional
            Allowed-term token sequences, or ``None``.
        include_terms : Iterable[Sequence[Any]], optional
            Required-term token sequences, or ``None``.
        exclude_terms : Iterable[Sequence[Any]], optional
            Forbidden-term token sequences, or ``None``.

        Returns
        -------
        dict[str, Any]
            A dict with ``input_ids``, ``input_num``, ``prompt_mask``, ``prompt_metadata``,
            ``prompt_disabled`` (True when the prompt block was omitted) and ``missing_tokens``.

        Raises
        ------
        KeyError
            If a token that is emitted into ``input_ids`` is missing from the tokenizer vocabulary.
        TypeError
            If a term collection is passed as a raw string rather than a sequence of tokens.
        """
        tokens: list[str] = ["<bos>"]
        numeric_values: list[float] = [np.nan]
        prompt_mask: list[bool] = [False]

        normalized_allowed = self._normalize_prompt_terms_collection(allowed_terms)
        normalized_include = self._normalize_prompt_terms_collection(include_terms)
        normalized_exclude = self._normalize_prompt_terms_collection(exclude_terms)

        metadata = {
            "allowed_terms": normalized_allowed,
            "include_terms": normalized_include,
            "exclude_terms": normalized_exclude,
        }

        has_prompt_content = (
            complexity is not None
            or bool(normalized_allowed)
            or bool(normalized_include)
            or bool(normalized_exclude)
        )

        required_prompt_tokens: set[str] = set()
        if has_prompt_content:
            required_prompt_tokens.update({"<prompt>", "</prompt>"})
            if complexity is not None:
                required_prompt_tokens.update({"<complexity>", "<float>", "</complexity>"})
            if normalized_allowed:
                required_prompt_tokens.update({"<allowed_term>", "</allowed_term>"})
            if normalized_include:
                required_prompt_tokens.update({"<include_term>", "</include_term>"})
            if normalized_exclude:
                required_prompt_tokens.update({"<exclude_term>", "</exclude_term>"})

        missing_prompt_tokens = [token for token in required_prompt_tokens if token not in self.tokenizer]
        emit_prompt = has_prompt_content and not missing_prompt_tokens

        if emit_prompt:
            tokens.append("<prompt>")
            numeric_values.append(np.nan)
            prompt_mask.append(True)

            if complexity is not None:
                tokens.extend(["<complexity>", "<float>", "</complexity>"])
                numeric_values.extend([np.nan, float(complexity), np.nan])
                prompt_mask.extend([True, True, True])

            for section_token, terms in (
                ("allowed_term", normalized_allowed),
                ("include_term", normalized_include),
                ("exclude_term", normalized_exclude),
            ):
                for term in terms:
                    tokens.append(f"<{section_token}>")
                    numeric_values.append(np.nan)
                    prompt_mask.append(True)
                    for token in term:
                        tokens.append(token)
                        numeric_values.append(np.nan)
                        prompt_mask.append(True)
                    tokens.append(f"</{section_token}>")
                    numeric_values.append(np.nan)
                    prompt_mask.append(True)

            tokens.append("</prompt>")
            numeric_values.append(np.nan)
            prompt_mask.append(True)

        missing_tokens: list[str] = list(missing_prompt_tokens)

        if "<bos>" not in self.tokenizer:
            missing_tokens.append("<bos>")

        if "<expression>" in self.tokenizer:
            tokens.append("<expression>")
            numeric_values.append(np.nan)
            prompt_mask.append(False)
        else:
            missing_tokens.append("<expression>")

        try:
            input_ids = [self.tokenizer[token] for token in tokens]
        except KeyError as exc:
            raise KeyError(
                f"Token '{exc.args[0]}' missing from tokenizer vocabulary while serializing prompt prefix."
            ) from exc

        return {
            "input_ids": input_ids,
            "input_num": numeric_values,
            "prompt_mask": prompt_mask,
            "prompt_metadata": metadata,
            "prompt_disabled": not emit_prompt,
            "missing_tokens": missing_tokens,
        }

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
    def _append_term_section(
        prefix: str,
        terms: Iterable[Sequence[str]],
        append_fn: Any,
    ) -> None:
        open_token = f"<{prefix}_term>"
        close_token = f"</{prefix}_term>"
        for term in terms:
            append_fn(open_token, is_prompt=True)
            for token in term:
                append_fn(str(token), is_prompt=True)
            append_fn(close_token, is_prompt=True)


def prepare_prompt_prefix(
        preprocessor: PromptSerializer | None,
        *,
        complexity: int | float | None,
        allowed_terms: Iterable[Sequence[Any]] | None,
        include_terms: Iterable[Sequence[Any]] | None,
        exclude_terms: Iterable[Sequence[Any]] | None) -> PromptPrefix | None:
    """Serialize prompt metadata into tokens usable by the transformer."""
    if preprocessor is None:
        return None

    serialized = preprocessor.serialize_prompt_prefix(
        complexity=complexity,
        allowed_terms=allowed_terms,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
    )

    tokens = list(serialized["input_ids"])
    numeric = [float(value) for value in serialized["input_num"]]
    mask = list(serialized["prompt_mask"])

    metadata_raw = serialized.get("prompt_metadata", {})
    if isinstance(metadata_raw, dict):
        metadata = {key: [list(term) for term in value] for key, value in metadata_raw.items()}
    else:
        metadata = {}

    return PromptPrefix(tokens=tokens, numeric=numeric, mask=mask, metadata=metadata)
