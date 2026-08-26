"""The v24 auxiliary task verbs: capabilities T16 was trained on but nothing could call.

`FlashANSR` exposes these as thin methods; the work lives here so the estimator does not grow a
second implementation of the training grammar. Each verb force-feeds exactly the circumstance the
data pipeline emitted during training -- the harness owns openers and flags, the model owns content
nibbles and closing tags -- because a prompt that drifts from the training grammar produces
confident, plausible, wrong output rather than an error.

Scope note: srbf benchmarks the traditional SR task only. These verbs are driven by custom scripts,
not by the benchmark driver.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from flash_ansr.data.serialization import (
    PREDICT_CONSTANTS_END_TOKEN,
    PREDICT_CONSTANTS_START_TOKEN,
)
from flash_ansr.preprocessing.prompt_serialization import CapabilityUnavailable
from flash_ansr.utils.ieee754 import (
    IEEE754_END_TOKEN,
    IEEE754_START_TOKEN,
    NIBBLE_TOKENS,
    nibble_tokens_to_float32,
)
from flash_ansr.utils.tensor_ops import pad_input_set

#: A float32 significand is 8 hex nibbles; the span is always exactly this long.
NIBBLES_PER_SPAN = 8

_VARIABLE_ALIAS = re.compile(r"^v(\d+)$")


@dataclass
class ConstantPrediction:
    """What :meth:`FlashANSR.predict_constants` returns.

    Attributes
    ----------
    values : list[float]
        The model's constants, in SLOT ORDER (order of appearance in the expression), as float32
        values widened to Python floats.
    nibble_logprobs : list[list[float]]
        Per slot, the log-probability the model assigned to each nibble it emitted. Low values flag
        a slot the model was unsure about -- the closest thing to a confidence the format affords.
    closed_cleanly : list[bool]
        Per slot, whether the model itself chose to close the span after 8 nibbles. ``False`` means
        the grammar had to close it, i.e. the emission was drifting.
    off_grammar_steps : int
        How many decode steps would have left the nibble alphabet had the grammar not restricted
        them. Non-zero means the prompt is out of distribution for this checkpoint.
    """

    values: list[float] = field(default_factory=list)
    nibble_logprobs: list[list[float]] = field(default_factory=list)
    closed_cleanly: list[bool] = field(default_factory=list)
    off_grammar_steps: int = 0


def _require(tokenizer: Any, token: str, verb: str) -> int:
    """Resolve ``token`` or refuse the verb before any compute is spent."""
    if token not in tokenizer:
        raise CapabilityUnavailable(
            f"{verb}() needs the {token} token, which this checkpoint's vocabulary does not "
            f"contain. Only v24 checkpoints trained on the task blocks carry it.")
    return int(tokenizer[token])


def _encoder_batch(X: np.ndarray, y: np.ndarray, n_variables: int,
                   device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``(X, y)`` into the (1, n_points, n_variables + 1) tensor the encoder reads."""
    x_arr = torch.as_tensor(np.asarray(X, dtype=np.float32))
    y_arr = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y must agree on the number of points, got {x_arr.shape[0]} and {y_arr.shape[0]}.")
    if not bool(torch.isfinite(y_arr).all()):
        raise ValueError("y contains non-finite values; drop or impute them before scoring.")

    data = torch.cat([torch.as_tensor(pad_input_set(x_arr, n_variables)), y_arr], dim=-1)
    data = data.unsqueeze(0).to(device)
    attn_mask = torch.ones(1, data.shape[1], dtype=torch.bool, device=device)
    return data, attn_mask


def score_outliers(estimator: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point outlier probability from the trained outlier head.

    Parameters
    ----------
    estimator : FlashANSR
        The loaded estimator.
    X, y : array-like
        The support set to score. One probability is returned per point.

    Returns
    -------
    np.ndarray
        Shape ``(n_points,)``, each entry in ``[0, 1]``.

    Raises
    ------
    CapabilityUnavailable
        If the checkpoint was built without an outlier head.

    Notes
    -----
    The head reads the DATA-SET ENCODER only, so a score conditions on ``(X, y)`` and never on any
    expression -- it asks "could a function have produced this point set", not "does this point fit
    that formula". Two measured caveats belong with any number it produces: the published
    AUROC 0.9888 is POOLED over instances and in-distribution (the val catalog and noise process the
    model trained on), while per-problem behaviour is far weaker -- a single lone outlier scores a
    median P of about 0.42; and the head degrades sharply above roughly 10% contamination, the
    ceiling it was trained under, collapsing to about 0.36 at 20%.
    """
    model = estimator.flash_ansr_model
    head = getattr(model, "outlier_head", None)
    if head is None:
        raise CapabilityUnavailable(
            "score_outliers() needs a checkpoint trained with an outlier head; this model.yaml "
            "declares outlier_head: false (or omits it).")

    tokenizer = model.tokenizer
    bos = _require(tokenizer, "<bos>", "score_outliers")
    eos = _require(tokenizer, "<eos>", "score_outliers")

    device = model.device
    data, attn_mask = _encoder_batch(X, y, estimator.n_variables, device)

    # A minimal <bos> <eos> decoder input: the head reads point_representations, which the ENCODER
    # populates, so the decoder side only has to be well-formed, never meaningful.
    ids = torch.tensor([[bos, eos]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        model(ids, data, data_attn_mask=attn_mask)
        representations = model.point_representations
        if representations is None:
            raise RuntimeError(
                "The encoder did not expose point_representations; the checkpoint declares an "
                "outlier head but the forward pass did not populate it.")
        logits = head(representations).squeeze(-1)
        probabilities = torch.sigmoid(logits)

    return probabilities.detach().cpu().numpy().reshape(-1)


def _normalize_expression(estimator: Any, expression: Sequence[str] | str) -> list[str]:
    """Accept prefix tokens or an infix string; return prefix tokens over ``x1..xN``."""
    if isinstance(expression, str):
        tokens = list(estimator.simplipy_engine.parse_expression(expression))
    else:
        tokens = [str(token) for token in expression]

    # FastSRB and several catalogs name variables v1..vn; the vocabulary is x1..xN. Resolve the
    # alias at the boundary (design principle 3) instead of making every caller do it by hand.
    return [f"x{match.group(1)}" if (match := _VARIABLE_ALIAS.match(token)) else token
            for token in tokens]


def predict_constants(
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        expression: Sequence[str] | str) -> ConstantPrediction:
    """Infill the ``<constant>`` slots of ``expression`` with the model's own constant prediction.

    Force-feeds the training circumstance exactly::

        <bos> <mask_all> <expression> ...tokens... </expression> <predict_constants> <ieee754> ...

    then greedy-decodes 8 nibbles per slot with the alphabet restricted to the nibble tokens, as in
    training. The model owns the closing tag, so whether it CHOSE to close is reported rather than
    hidden (:attr:`ConstantPrediction.closed_cleanly`).

    Parameters
    ----------
    estimator : FlashANSR
        The loaded estimator.
    X, y : array-like
        The support set the constants should explain.
    expression : sequence of str or str
        Prefix tokens or an infix string, with ``'<constant>'`` marking each slot to fill.
        ``v1..vn`` variable names are accepted and mapped to ``x1..xN``.

    Returns
    -------
    ConstantPrediction
        Values in slot order, plus per-slot nibble log-probabilities and grammar diagnostics.

    Raises
    ------
    CapabilityUnavailable
        If the checkpoint lacks the ``<predict_constants>`` block or the ieee754 vocabulary.
    ValueError
        If the expression has no ``'<constant>'`` slots, or carries a token the vocabulary lacks.
    """
    model = estimator.flash_ansr_model
    tokenizer = model.tokenizer

    bos = _require(tokenizer, "<bos>", "predict_constants")
    pc_start = _require(tokenizer, PREDICT_CONSTANTS_START_TOKEN, "predict_constants")
    _require(tokenizer, PREDICT_CONSTANTS_END_TOKEN, "predict_constants")
    span_start = _require(tokenizer, IEEE754_START_TOKEN, "predict_constants")
    span_end = _require(tokenizer, IEEE754_END_TOKEN, "predict_constants")
    mask_all = _require(tokenizer, "<mask_all>", "predict_constants")
    nibble_ids = [int(tokenizer[token]) for token in NIBBLE_TOKENS]

    tokens = _normalize_expression(estimator, expression)
    n_slots = sum(1 for token in tokens if token == "<constant>")
    if n_slots == 0:
        raise ValueError(
            "expression has no '<constant>' slots to infill. Mark each constant position with "
            "'<constant>' -- predict_constants fills slots, it does not choose where they are.")

    unknown = sorted({token for token in tokens if token not in tokenizer})
    if unknown:
        raise ValueError(
            f"expression carries tokens this checkpoint's vocabulary lacks: {unknown}")

    body = tokenizer.encode(["<expression>", *tokens, "</expression>"], add_bos=False, add_eos=False)
    sequence = [bos, mask_all, *body, pc_start, span_start]

    device = model.device
    data, attn_mask = _encoder_batch(X, y, estimator.n_variables, device)

    result = ConstantPrediction()
    nibble_index = torch.tensor(nibble_ids, dtype=torch.long, device=device)

    model.eval()
    memory = None
    with torch.no_grad():
        for slot in range(n_slots):
            emitted: list[str] = []
            logprobs: list[float] = []
            for _ in range(NIBBLES_PER_SPAN):
                ids = torch.tensor([sequence], dtype=torch.long, device=device)
                numeric = torch.full((1, len(sequence)), float("nan"), device=device)
                if memory is None:
                    logits = model(ids, data, input_num=numeric, data_attn_mask=attn_mask)
                    memory = model.memory
                else:
                    logits = model(ids, None, input_num=numeric, memory=memory,
                                   data_attn_mask=attn_mask)

                step = logits[0, -1]
                # Grammar restriction, as in training: only nibbles are legal here. Record when the
                # unrestricted argmax would have left the alphabet -- that is the observable that
                # says the prompt is out of distribution.
                if int(torch.argmax(step)) not in nibble_ids:
                    result.off_grammar_steps += 1
                restricted = step[nibble_index]
                choice = int(torch.argmax(restricted))
                logprobs.append(float(torch.log_softmax(restricted, dim=-1)[choice]))

                token_id = nibble_ids[choice]
                emitted.append(str(tokenizer[token_id]))
                sequence.append(token_id)

            # The model owns the closing tag; ask whether it wanted to close before forcing it.
            ids = torch.tensor([sequence], dtype=torch.long, device=device)
            numeric = torch.full((1, len(sequence)), float("nan"), device=device)
            logits = model(ids, None, input_num=numeric, memory=memory, data_attn_mask=attn_mask)
            result.closed_cleanly.append(bool(int(torch.argmax(logits[0, -1])) == span_end))
            sequence.append(span_end)

            result.values.append(float(nibble_tokens_to_float32(emitted)))
            result.nibble_logprobs.append(logprobs)

            if slot < n_slots - 1:
                sequence.append(span_start)

    return result
