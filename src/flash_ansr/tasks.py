"""The v24 auxiliary task verbs: capabilities T16 was trained on but nothing could call.

`FlashANSR` exposes these as thin methods; the work lives here so the estimator does not grow a
second implementation of the training grammar. Each verb force-feeds exactly the circumstance the
data pipeline emitted during training -- the harness owns openers and flags, the model owns content
nibbles and closing tags -- because a prompt that drifts from the training grammar produces
confident, plausible, wrong output rather than an error.

**Every value here is a DISTRIBUTION, never a float.** A value is spelled as eight hex nibbles and
each nibble is drawn from a softmax, so one decode is one sample from a factorised distribution
over values. Greedy decoding returns that distribution's mode, which is neither its centre nor any
indication of its width -- and the width matters enormously here, because the leading nibbles carry
the exponent, so a single flipped nibble moves the value by orders of magnitude. Conclusions drawn
from a single decode are conclusions drawn from one sample.

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
    COMPACT_CONSTANT_TOKEN,
    COMPLEXITY_END_TOKEN,
    COMPLEXITY_START_TOKEN,
    HYPOTHESIS_TOKEN,
    POINT_END_TOKEN,
    POINT_START_TOKEN,
    PREDICT_CONSTANTS_END_TOKEN,
    PREDICT_CONSTANTS_START_TOKEN,
    PREDICT_Y_END_TOKEN,
    PREDICT_Y_START_TOKEN,
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

#: Draws taken by default whenever a verb reads a value out of the model. NOT 1: a single decode is
#: one sample, and a sample quoted as an answer is how a wide belief gets reported as a point.
DEFAULT_SAMPLES = 32

_VARIABLE_ALIAS = re.compile(r"^v(\d+)$")


@dataclass
class ValueDistribution:
    """Draws of one predicted value, plus the summaries that may legitimately be quoted from them.

    Attributes
    ----------
    draws : list[float]
        Every sampled value, in draw order.
    off_grammar_steps : int
        Decode steps whose unrestricted argmax would have left the nibble alphabet, summed over
        draws. Non-zero means the prompt is out of distribution for this checkpoint.
    closed_cleanly_fraction : float
        Fraction of draws where the model itself chose to close the span after 8 nibbles.
    """

    draws: list[float] = field(default_factory=list)
    off_grammar_steps: int = 0
    closed_cleanly_fraction: float = 0.0

    @property
    def n(self) -> int:
        """Number of draws taken."""
        return len(self.draws)

    @property
    def finite_draws(self) -> list[float]:
        """Draws that are finite.

        A nibble pattern can decode to nan or inf. Those are genuine outcomes of the model's
        distribution -- ``non_finite_fraction`` reports how often -- but they cannot enter a
        quantile, so the summaries below are taken over the finite draws only.
        """
        return [value for value in self.draws if np.isfinite(value)]

    @property
    def non_finite_fraction(self) -> float:
        """Fraction of draws that decoded to nan or inf."""
        if not self.draws:
            return 0.0
        return 1.0 - len(self.finite_draws) / len(self.draws)

    @property
    def median(self) -> float:
        """Median of the finite draws -- the point summary to quote, if one must be."""
        finite = self.finite_draws
        return float(np.median(finite)) if finite else float("nan")

    @property
    def q05(self) -> float:
        """5th percentile of the finite draws."""
        finite = self.finite_draws
        return float(np.quantile(finite, 0.05)) if finite else float("nan")

    @property
    def q95(self) -> float:
        """95th percentile of the finite draws."""
        finite = self.finite_draws
        return float(np.quantile(finite, 0.95)) if finite else float("nan")

    @property
    def mode(self) -> float:
        """The most frequently drawn exact value (what greedy decoding approximates)."""
        finite = self.finite_draws
        if not finite:
            return float("nan")
        values, counts = np.unique(np.asarray(finite), return_counts=True)
        return float(values[int(np.argmax(counts))])

    @property
    def agreement(self) -> float:
        """Fraction of finite draws equal to :attr:`mode` -- how concentrated the belief is.

        Near 1.0 the model is effectively certain and the mode is a fair summary; near 0 the draws
        disagree and no single number represents them.
        """
        finite = self.finite_draws
        if not finite:
            return 0.0
        mode = self.mode
        return float(sum(1 for value in finite if value == mode) / len(finite))

    def __repr__(self) -> str:
        return (f"ValueDistribution(n={self.n}, median={self.median:.6g}, "
                f"q05={self.q05:.6g}, q95={self.q95:.6g}, agreement={self.agreement:.2f})")


@dataclass
class ComplexityDistribution(ValueDistribution):
    """A :class:`ValueDistribution` over complexity, with the hypothesis diagnostic.

    Attributes
    ----------
    self_initiated_fraction : float
        Fraction of draws where the model, given only the ``<hypothesize>`` licence, would have
        opened a ``<complexity>`` block on its own rather than needing the opener forced.
    """

    self_initiated_fraction: float = 0.0


def _require(tokenizer: Any, token: str, verb: str) -> int:
    """Resolve ``token`` or refuse the verb before any compute is spent."""
    if token not in tokenizer:
        raise CapabilityUnavailable(
            f"{verb}() needs the {token} token, which this checkpoint's vocabulary does not "
            f"contain. This harness serves v24 checkpoints only.")
    return int(tokenizer[token])


def _seeded_generator(device: Any, seed: int | None) -> torch.Generator | None:
    """A seeded generator so a published distribution reproduces; ``None`` means fresh entropy."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


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


def _start_state(prefix_tokens: list[int], prefix_numeric: list[float], n_samples: int,
                 device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``n_samples`` identical rows of the prompt prefix, ready to diverge."""
    sequences = torch.tensor([prefix_tokens], dtype=torch.long, device=device).repeat(n_samples, 1)
    numerics = torch.tensor([prefix_numeric], dtype=torch.float32, device=device).repeat(n_samples, 1)
    return sequences, numerics


def _append(state: tuple[torch.Tensor, torch.Tensor], token_id: int,
            numeric: float = float("nan")) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one harness-owned token to every row."""
    sequences, numerics = state
    rows = sequences.shape[0]
    device = sequences.device
    return (
        torch.cat([sequences, torch.full((rows, 1), token_id, dtype=torch.long, device=device)], dim=1),
        torch.cat([numerics, torch.full((rows, 1), numeric, dtype=torch.float32, device=device)], dim=1),
    )


def _conditioning(estimator: Any, X: Any, y: Any, *, conditioned: bool, n_rows: int,
                  verb: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build the encoder inputs and the per-row condition mask for one verb call.

    ``conditioned=True`` is the ordinary path: the model reads the ``(X, y)`` support set.
    ``conditioned=False`` selects the trained UNCONDITIONED mode -- ``forward`` swaps the learned
    ``null_memory`` in for the encoder memory, so the verb answers from the token stream alone.
    The encoder still runs (that is the path training took, and its output is then discarded),
    which is why ``X``/``y`` may be omitted entirely: a placeholder support set is synthesized.
    """
    model = estimator.flash_ansr_model
    if conditioned:
        if X is None or y is None:
            raise ValueError(f"{verb}() needs X and y unless conditioned=False.")
        data, attn_mask = _encoder_batch(X, y, estimator.n_variables, model.device)
        return data, attn_mask, None

    if not getattr(model, "optional_condition", False):
        raise CapabilityUnavailable(
            f"{verb}(conditioned=False) needs a model trained with optional_condition=True "
            f"(the learned null_memory); this checkpoint has no unconditioned mode.")
    if X is None or y is None:
        # Discarded by the null substitution -- shape is all that matters.
        X = np.zeros((1, estimator.n_variables), dtype=np.float32)
        y = np.zeros((1,), dtype=np.float32)
    data, attn_mask = _encoder_batch(X, y, estimator.n_variables, model.device)
    mask = torch.zeros(n_rows, dtype=torch.bool, device=model.device)
    return data, attn_mask, mask


def _sample_span(
        model: Any,
        state: tuple[torch.Tensor, torch.Tensor],
        data: torch.Tensor,
        attn_mask: torch.Tensor,
        nibble_ids: list[int],
        span_end: int,
        temperature: float,
        generator: torch.Generator | None,
        condition_mask: torch.Tensor | None = None,
) -> tuple[ValueDistribution, tuple[torch.Tensor, torch.Tensor]]:
    """Draw one ieee754 span for every row of ``state``, decoded in parallel.

    Each row keeps its OWN history, so a multi-slot decode samples the joint distribution over
    slots rather than a product of per-slot marginals. Each nibble is drawn from the softmax
    restricted to the nibble alphabet -- the same restriction training imposed. The closing tag is
    appended by the harness, but whether the model WANTED to close is measured first.
    """
    device = data.device
    nibble_index = torch.tensor(nibble_ids, dtype=torch.long, device=device)
    tokenizer = model.tokenizer
    sequences, numerics = state
    n_rows = sequences.shape[0]

    off_grammar = 0
    memory = None
    for _ in range(NIBBLES_PER_SPAN):
        if memory is None:
            # The mask applies on the FIRST pass only: forward substitutes null_memory into
            # `model.memory` in place, so the cached memory below is already the routed one.
            logits = model(sequences, data, input_num=numerics, data_attn_mask=attn_mask,
                           condition_mask=condition_mask)
            memory = model.memory
        else:
            logits = model(sequences, None, input_num=numerics, memory=memory,
                           data_attn_mask=attn_mask)

        step = logits[:, -1]
        off_grammar += int((~torch.isin(step.argmax(dim=-1), nibble_index)).sum().item())

        restricted = step[:, nibble_index]
        if temperature <= 0:
            choice = restricted.argmax(dim=-1)
        else:
            probabilities = torch.softmax(restricted / float(temperature), dim=-1)
            choice = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)

        sequences = torch.cat([sequences, nibble_index[choice].unsqueeze(-1)], dim=1)
        numerics = torch.cat(
            [numerics, torch.full((n_rows, 1), float("nan"), device=device)], dim=1)

    logits = model(sequences, None, input_num=numerics, memory=memory, data_attn_mask=attn_mask)
    closed = (logits[:, -1].argmax(dim=-1) == span_end)

    draws = [
        float(nibble_tokens_to_float32([str(tokenizer[int(t)]) for t in row]))
        for row in sequences[:, -NIBBLES_PER_SPAN:].tolist()
    ]

    distribution = ValueDistribution(
        draws=draws,
        off_grammar_steps=off_grammar,
        closed_cleanly_fraction=float(closed.float().mean().item()),
    )
    return distribution, _append((sequences, numerics), span_end)


def score_outliers(estimator: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point outlier probability from the trained outlier head.

    Unlike the other verbs this one is NOT a sampled decode -- it is a single deterministic forward
    pass through a sigmoid head, so one call is the answer, not a draw.

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
    AUROC 0.9888 is POOLED over instances and in-distribution, while per-problem behaviour is far
    weaker; and the head degrades sharply above roughly 10% contamination, the ceiling it was
    trained under.
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
        probabilities = torch.sigmoid(head(representations).squeeze(-1))

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
        X: "np.ndarray | None",
        y: "np.ndarray | None",
        expression: Sequence[str] | str,
        *,
        conditioned: bool = True,
        n_samples: int = DEFAULT_SAMPLES,
        temperature: float = 1.0,
        seed: int | None = None) -> list[ValueDistribution]:
    """Infill the ``<constant>`` slots of ``expression`` from the model's own distribution.

    Force-feeds the training circumstance exactly::

        <bos> <mask_all> <expression> ...tokens... </expression> <predict_constants> <ieee754> ...

    then draws ``n_samples`` values per slot. Rows decode in parallel and each keeps its own
    history, so the draws are JOINT across slots.

    Parameters
    ----------
    estimator : FlashANSR
        The loaded estimator.
    X, y : array-like
        The support set the constants should explain.
    expression : sequence of str or str
        Prefix tokens or an infix string, with ``'<constant>'`` marking each slot to fill.
    conditioned : bool, optional
        ``False`` selects the trained unconditioned mode -- the constants the model considers
        typical for this SHAPE, with no data to fit, by default ``True``.
    n_samples : int, optional
        Draws, by default :data:`DEFAULT_SAMPLES`.
    temperature : float, optional
        Softmax temperature, by default 1.0 (the training distribution). ``0`` decodes greedily,
        which returns the mode of every slot and is only useful for a determinism check.
    seed : int, optional
        Seeds the sampling so a published distribution reproduces.

    Returns
    -------
    list[ValueDistribution]
        One distribution per slot, in slot order.

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
    prefix_tokens = [bos, mask_all, *body, pc_start, span_start]

    data, attn_mask, condition_mask = _conditioning(
        estimator, X, y, conditioned=conditioned, n_rows=int(n_samples), verb="predict_constants")
    generator = _seeded_generator(model.device, seed)

    model.eval()
    results: list[ValueDistribution] = []
    with torch.no_grad():
        state = _start_state(prefix_tokens, [float("nan")] * len(prefix_tokens),
                             int(n_samples), model.device)
        for slot in range(n_slots):
            distribution, state = _sample_span(
                model, state, data, attn_mask, nibble_ids, span_end, float(temperature), generator,
                condition_mask=condition_mask)
            results.append(distribution)
            if slot < n_slots - 1:
                state = _append(state, span_start)

    return results


def predict_y(
        estimator: Any,
        X: np.ndarray | None,
        y: np.ndarray | None,
        x_query: np.ndarray,
        *,
        expression: "Sequence[str] | str | None" = None,
        conditioned: bool = True,
        n_samples: int = DEFAULT_SAMPLES,
        temperature: float = 1.0,
        seed: int | None = None) -> list[ValueDistribution]:
    """Predict the target at held-out points, from the model's ``<predict_y>`` block.

    Training writes this block in two placements, and both are reachable here. Without an
    expression the block sits in the PREFIX -- the model interpolates the point set it was given::

        <bos> <predict_y> <point> <float>...<float> </point> <ieee754> [8 nibbles of y*] </ieee754>

    With one it sits in the SUFFIX, after the expression, so the expression is in scope::

        <bos> <expression> ...tokens... </expression> <predict_y> <point> ... </point> <ieee754> ...

    The query coordinates ride the NUMERIC channel exactly as training wrote them.

    Combining ``expression`` with ``conditioned=False`` gives FUNCTION EVALUATION: the encoder
    memory is replaced by the learned ``null_memory``, so the only thing the model can answer from
    is the expression and the query point. Measured 2026-08-26: with the data present the
    expression buys nothing (median normalized error 0.270 vs 0.277, paired p = 0.56), and
    expression-only sat at the no-information floor -- but T16 was trained with the block WITHHELD
    from unconditioned instances, so that floor measures an untrained circumstance, not a limit.

    Parameters
    ----------
    estimator : FlashANSR
        The loaded estimator.
    X, y : array-like or None
        The support set the model conditions on. May be ``None`` only when ``conditioned=False``.
    x_query : array-like
        Query coordinates, ``(n_queries, n_variables)`` or a single ``(n_variables,)`` point.
    expression : sequence of str or str, optional
        Prefix tokens or an infix string. When given, the block is placed AFTER the expression --
        the trained suffix circumstance. When omitted, the prefix circumstance.
    conditioned : bool, optional
        ``False`` selects the trained unconditioned mode (``null_memory`` replaces the encoder
        memory), by default ``True``.
    n_samples : int, optional
        Draws per query point, by default :data:`DEFAULT_SAMPLES`.
    temperature : float, optional
        Softmax temperature, by default 1.0.
    seed : int, optional
        Seeds the sampling.

    Returns
    -------
    list[ValueDistribution]
        One distribution per query point, in query order.

    Raises
    ------
    CapabilityUnavailable
        If the checkpoint lacks the ``<predict_y>`` block, or ``conditioned=False`` on a model
        without a ``null_memory``.
    ValueError
        If a query point's dimensionality does not match ``X``, or ``X``/``y`` are omitted while
        ``conditioned=True``.
    """
    model = estimator.flash_ansr_model
    tokenizer = model.tokenizer

    bos = _require(tokenizer, "<bos>", "predict_y")
    py_start = _require(tokenizer, PREDICT_Y_START_TOKEN, "predict_y")
    _require(tokenizer, PREDICT_Y_END_TOKEN, "predict_y")
    point_start = _require(tokenizer, POINT_START_TOKEN, "predict_y")
    point_end = _require(tokenizer, POINT_END_TOKEN, "predict_y")
    compact = _require(tokenizer, COMPACT_CONSTANT_TOKEN, "predict_y")
    span_start = _require(tokenizer, IEEE754_START_TOKEN, "predict_y")
    span_end = _require(tokenizer, IEEE754_END_TOKEN, "predict_y")
    nibble_ids = [int(tokenizer[token]) for token in NIBBLE_TOKENS]

    queries = np.atleast_2d(np.asarray(x_query, dtype=np.float64))
    if X is not None:
        x_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if queries.shape[-1] != x_arr.shape[-1]:
            raise ValueError(
                f"x_query has {queries.shape[-1]} coordinate(s) but X has "
                f"{x_arr.shape[-1]} variable(s).")

    body: list[int] = []
    if expression is not None:
        tokens = _normalize_expression(estimator, expression)
        unknown = sorted({token for token in tokens if token not in tokenizer})
        if unknown:
            raise ValueError(
                f"expression carries tokens this checkpoint's vocabulary lacks: {unknown}")
        _require(tokenizer, "<expression>", "predict_y")
        _require(tokenizer, "</expression>", "predict_y")
        body = tokenizer.encode(["<expression>", *tokens, "</expression>"],
                                add_bos=False, add_eos=False)

    data, attn_mask, condition_mask = _conditioning(
        estimator, X, y, conditioned=conditioned, n_rows=int(n_samples), verb="predict_y")
    generator = _seeded_generator(model.device, seed)

    model.eval()
    results: list[ValueDistribution] = []
    with torch.no_grad():
        for point in queries:
            prefix_tokens = [bos, *body, py_start, point_start]
            # <bos> + body + <predict_y> + <point>: the numeric channel is NaN across all of it
            # (a spelled constant rides its own ieee754 span, never this prefix).
            prefix_numeric = [float("nan")] * len(prefix_tokens)
            for coordinate in point:
                prefix_tokens.append(compact)
                prefix_numeric.append(float(coordinate))
            prefix_tokens += [point_end, span_start]
            prefix_numeric += [float("nan"), float("nan")]

            state = _start_state(prefix_tokens, prefix_numeric, int(n_samples), model.device)
            distribution, _ = _sample_span(
                model, state, data, attn_mask, nibble_ids, span_end, float(temperature), generator,
                condition_mask=condition_mask)
            results.append(distribution)

    return results


def predict_complexity(
        estimator: Any,
        X: "np.ndarray | None",
        y: "np.ndarray | None",
        *,
        conditioned: bool = True,
        n_samples: int = DEFAULT_SAMPLES,
        temperature: float = 1.0,
        seed: int | None = None) -> ComplexityDistribution:
    """Ask the model how complex it thinks the generating expression is.

    Uses the trained HYPOTHESIS circumstance: the harness utters ``<hypothesize>`` -- a licence only
    it may give -- and everything after it is the model's own. Training supervised the opener too,
    so the verb records how often the model would have opened ``<complexity>`` unprompted before
    forcing it, rather than presenting a forced number as a spontaneous one.

    Parameters
    ----------
    estimator : FlashANSR
        The loaded estimator.
    X, y : array-like or None
        The support set to judge. May be ``None`` only when ``conditioned=False``.
    conditioned : bool, optional
        ``False`` selects the trained unconditioned mode -- the model's PRIOR over complexity with
        no data to judge, which is the reference any conditioned reading should be compared
        against, by default ``True``.
    n_samples : int, optional
        Draws, by default :data:`DEFAULT_SAMPLES`.
    temperature : float, optional
        Softmax temperature, by default 1.0.
    seed : int, optional
        Seeds the sampling.

    Returns
    -------
    ComplexityDistribution
        Draws in simplipy complexity units -- the unit ``fit(complexity=...)`` consumes, running
        roughly 1e3-1e6, NOT a token count.

    Raises
    ------
    CapabilityUnavailable
        If the checkpoint lacks the ``<hypothesize>`` or ``<complexity>`` tokens.
    """
    model = estimator.flash_ansr_model
    tokenizer = model.tokenizer

    bos = _require(tokenizer, "<bos>", "predict_complexity")
    hypothesize = _require(tokenizer, HYPOTHESIS_TOKEN, "predict_complexity")
    cx_start = _require(tokenizer, COMPLEXITY_START_TOKEN, "predict_complexity")
    _require(tokenizer, COMPLEXITY_END_TOKEN, "predict_complexity")
    span_start = _require(tokenizer, IEEE754_START_TOKEN, "predict_complexity")
    span_end = _require(tokenizer, IEEE754_END_TOKEN, "predict_complexity")
    nibble_ids = [int(tokenizer[token]) for token in NIBBLE_TOKENS]

    data, attn_mask, condition_mask = _conditioning(
        estimator, X, y, conditioned=conditioned, n_rows=int(n_samples), verb="predict_complexity")
    generator = _seeded_generator(model.device, seed)

    model.eval()
    with torch.no_grad():
        state = _start_state([bos, hypothesize], [float("nan"), float("nan")],
                             int(n_samples), model.device)
        sequences, numerics = state
        logits = model(sequences, data, input_num=numerics, data_attn_mask=attn_mask,
                       condition_mask=condition_mask)
        self_initiated = float((logits[:, -1].argmax(dim=-1) == cx_start).float().mean().item())

        state = _append(_append(state, cx_start), span_start)
        distribution, _ = _sample_span(
            model, state, data, attn_mask, nibble_ids, span_end, float(temperature), generator,
            condition_mask=condition_mask)

    return ComplexityDistribution(
        draws=distribution.draws,
        off_grammar_steps=distribution.off_grammar_steps,
        closed_cleanly_fraction=distribution.closed_cleanly_fraction,
        self_initiated_fraction=self_initiated,
    )
