"""Prompt serialization helpers shared across preprocessing and inference.

LEGACY LANE (owner ruling 2026-08-26: keep, mark, revisit). :meth:`PromptSerializer.
serialize_prompt` and the term sections below are the FIRST-GENERATION promptable-property
mechanism: one ``<prompt>`` ... ``</prompt>`` wrapper enclosing typed sections, with the
allowed/include/exclude term lists carried out-of-band in ``prompt_metadata``. It is superseded
in spirit by the v24 grammar, where each property is a BARE prefix element the harness
force-feeds and loss-masks (``<complexity> <float> </complexity>``, ``<mask_all>``,
``<hypothesize>``) -- elements that permute per instance rather than nesting inside a wrapper.

No v24 config reaches this path: v24 datasets set no preprocessor, and the term arguments are
withdrawn from the public inference surface (they emitted tokens no v24 checkpoint saw and were
enforced nowhere at decode time). :meth:`PromptSerializer.serialize_prompt_prefix` -- the lane
inference actually uses -- already emits the bare v24 form.

WHEN MORE PROMPTABLE PROPERTIES LAND, RECONCILE THE TWO RATHER THAN EXTENDING THIS ONE. A new
property added as another ``<prompt>`` section inherits a wrapper the model was never trained to
read, a metadata side-channel with no decode-time meaning, and a fixed section order. The v24
element grammar is the target shape; this code is what the migration has to absorb or delete.
"""
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
        include_complexity: bool,   # LEGACY <prompt>-wrapper lane -- see the module docstring
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

    def serialize_prompt_prefix(self, *, complexity: float | int | None = None) -> dict[str, Any]:
        """Serialize a decoding prompt prefix: ``<bos>``, an optional complexity block, ``<expression>``.

Training emits the complexity block BARE, as a prefix element -- ``<complexity>``,
        ``<float>`` carrying mu on the numeric channel, ``</complexity>`` -- never inside a
        ``<prompt>`` wrapper.

        Parameters
        ----------
        complexity : float or int, optional
            Target complexity in **simplipy mu** (roughly 1e3-1e6, NOT a token count), or ``None``
            to omit the block.

        Returns
        -------
        dict[str, Any]
            ``input_ids``, ``input_num``, ``prompt_mask``, ``prompt_metadata``, ``prompt_disabled``
            and ``missing_tokens``.

        Raises
        ------
        KeyError
            If a token emitted into ``input_ids`` is missing from the tokenizer vocabulary.
        """
        tokens: list[str] = ["<bos>"]
        numeric_values: list[float] = [np.nan]
        prompt_mask: list[bool] = [False]

        emit_complexity = complexity is not None and all(
            token in self.tokenizer for token in ("<complexity>", "<float>", "</complexity>"))
        if emit_complexity:
            tokens.extend(["<complexity>", "<float>", "</complexity>"])
            numeric_values.extend([np.nan, float(complexity), np.nan])
            prompt_mask.extend([True, True, True])

        missing_tokens: list[str] = []
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
                f"Token '{exc.args[0]}' missing from tokenizer vocabulary while serializing "
                f"prompt prefix.") from exc

        return {
            "input_ids": input_ids,
            "input_num": numeric_values,
            "prompt_mask": prompt_mask,
            "prompt_metadata": {},
            "prompt_disabled": not emit_complexity,
            "missing_tokens": missing_tokens,
        }

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


class CapabilityUnavailable(ValueError):
    """A verb was asked for a trained capability this checkpoint's vocabulary lacks.

    Raised at CALL time, before the encoder runs -- a capability check that fires mid-decode
    has already cost the caller the expensive part of the request.
    """


#: Emission format -> the promptable flag that selects it. ``"constants"`` is the UNFLAGGED
#: default (90% of training instances): the model spells constants as ieee754 spans. The two
#: flags are harness-owned emission-format directives, force-fed at the training position
#: (immediately after ``<bos>``) and never sampled.
EMISSION_FLAGS: dict[str, str | None] = {
    "constants": None,
    "skeleton": "<mask_all>",
    "fittable": "<mask_fittable>",
}


def apply_emission_flag(prefix: PromptPrefix, emission: str, tokenizer: Any) -> PromptPrefix:
    """Insert the emission-format flag for ``emission`` directly after ``<bos>``.

    This is the public form of the monkeypatch every published T16 number was produced
    through: the capability probes reached into ``_prepare_prompt_prefix`` on the instance
    because no public verb could set the flag. Position matters -- training put the flag
    immediately after ``<bos>``, so that is where it goes, with a NaN on the numeric channel
    like every other non-payload position.

    Parameters
    ----------
    prefix : PromptPrefix
        The prefix built for this call; returned unchanged for the unflagged default.
    emission : str
        One of :data:`EMISSION_FLAGS`.
    tokenizer : Any
        The model tokenizer, used to resolve the flag and to check it exists.

    Returns
    -------
    PromptPrefix
        A new prefix carrying the flag, or ``prefix`` itself when ``emission`` needs none.

    Raises
    ------
    ValueError
        If ``emission`` is not a known mode.
    CapabilityUnavailable
        If the flag is absent from the vocabulary.
    """
    if emission not in EMISSION_FLAGS:
        raise ValueError(
            f"emission must be one of {sorted(EMISSION_FLAGS)}, got {emission!r}")

    flag_token = EMISSION_FLAGS[emission]
    if flag_token is None:
        return prefix

    if flag_token not in tokenizer:
        raise CapabilityUnavailable(
            f"emission={emission!r} needs the {flag_token} token, which this checkpoint's "
            f"vocabulary does not contain. Only mixed-representation (v24) checkpoints carry "
            f"the promptable emission flags; use emission='constants'.")

    flag_id = int(tokenizer[flag_token])
    return PromptPrefix(
        tokens=[prefix.tokens[0], flag_id] + list(prefix.tokens[1:]),
        numeric=[prefix.numeric[0], float("nan")] + list(prefix.numeric[1:]),
        mask=[True] + list(prefix.mask),
        metadata=prefix.metadata,
    )


def prepare_prompt_prefix(
        preprocessor: PromptSerializer | None,
        *,
        complexity: int | float | None) -> PromptPrefix | None:
    """Serialize the prompt prefix into tokens usable by the transformer."""
    if preprocessor is None:
        return None

    serialized = preprocessor.serialize_prompt_prefix(complexity=complexity)

    tokens = list(serialized["input_ids"])
    numeric = [float(value) for value in serialized["input_num"]]
    mask = list(serialized["prompt_mask"])

    metadata_raw = serialized.get("prompt_metadata", {})
    if isinstance(metadata_raw, dict):
        metadata = {key: [list(term) for term in value] for key, value in metadata_raw.items()}
    else:
        metadata = {}

    return PromptPrefix(tokens=tokens, numeric=numeric, mask=mask, metadata=metadata)
