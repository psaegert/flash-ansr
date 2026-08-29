"""Prompt-prefix serialization shared across preprocessing and inference.

Promptable properties are BARE prefix elements the harness force-feeds and loss-masks
(``<complexity> <float> </complexity>``, ``<mask_all>``, ``<hypothesize>``) -- elements that
permute per instance. :meth:`PromptSerializer.serialize_prompt_prefix` emits the decoding
prefix in this form; :func:`apply_emission_flag` inserts the emission-format flags.
"""
from typing import Any

import numpy as np

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing.schemas import PromptPrefix


class PromptSerializer:
    """Serialize decoding prompt prefixes into token sequences consumable by the model."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

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
