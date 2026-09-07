"""Candidates from the training prior: the generative catalog a model trained on, serialized the
way the model emits expressions, so refinement and ranking run on them unchanged.

The control for "what is the posterior worth". ``method: prior_sampling`` swaps the decoder for
draws from the catalog (``catalog_train.yaml`` beside the checkpoint by default) and leaves every
later stage -- fittable-constant refinement, the ranking, the candidate ledger -- exactly as it is
for a decoded beam. A draw goes through the training stream's own steps: positional literal
masking, the fittable-slot policy, ieee754 serialization of the kept structural literals, and the
tokenizer, so a prior candidate is what the target for that draw would have been. The data never
reaches the sampler; it reaches the refiner, which fits the constants, and the ranking, which
scores the fits: "propose from the prior, fit to the data".

``decontaminate=True`` (the default) draws from the distribution the model was trained on, with
the benchmark holdout applied; ``False`` draws from the bare prior, which can propose a benchmark
law's own family. With ``n_variables`` (the problem's input columns; ``match_variables`` in the
generation config) a draw is conditioned on the one thing every regressor is told, the number of
inputs: draws with more distinct variables than columns are rejected, and the others are relabeled
onto a uniformly random injection into the columns -- the prior is exchangeable over variable
names, so this is the conditional draw, not a new prior. Without it the raw prior over the
catalog's full variable set is proposed as is.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from symbolic_data import NoValidSampleFoundError
from symbolic_data.generative import GenerativeCatalog, build_catalog
from symbolic_data.token_ops import substitute_constants

from flash_ansr.data.serialization import serialize_constant_tokens
from flash_ansr.utils.paths import substitute_root_path
from flash_ansr.utils.skeleton import NonFiniteExpressionError, fittable_slots, mask_literals_positional

CATALOG_FILENAME = "catalog_train.yaml"
"""The training prior's spec, shipped beside ``model.yaml`` in a model bundle."""


def resolve_prior_catalog(catalog: Any, model_directory: str | None) -> Any:
    """The catalog spec ``prior_sampling`` draws from: the config's ``catalog`` (a path, a
    ``name[@version]`` ref, or an inline mapping), else ``catalog_train.yaml`` beside the model."""
    if catalog is None:
        if not model_directory:
            raise ValueError(
                "prior_sampling needs the training prior: pass catalog=<path or spec> in the generation "
                f"config, or load the model from a directory that carries {CATALOG_FILENAME}")
        path = os.path.join(substitute_root_path(str(model_directory)), CATALOG_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"prior_sampling: {path} is missing; the model bundle carries no training prior. "
                "Pass catalog=<path or spec> in the generation config.")
        return path
    if isinstance(catalog, str):
        return substitute_root_path(catalog)
    return catalog


class PriorSampler:
    """Draw candidate expressions from a generative catalog as token-id beams.

    Parameters
    ----------
    catalog : str | Mapping | Catalog
        The training prior: a spec path, a catalog ref, an inline mapping with a ``type`` key, or
        a built generative catalog.
    engine : SimpliPyEngine
        The model's engine (literal sites, the fittable policy, validity).
    tokenizer : Tokenizer
        The model's tokenizer; a draw the vocabulary cannot spell (a variable beyond the model's
        range) is discarded.
    decontaminate : bool
        Apply the catalog's holdout to every draw (the training distribution) or not (the bare prior).
    seed : int, optional
        Seed of the draw stream; ``None`` is fresh entropy.
    """

    def __init__(self, catalog: Any, *, engine: Any, tokenizer: Any, decontaminate: bool = True,
                 seed: int | None = None) -> None:
        built = build_catalog(catalog)
        if not isinstance(built, GenerativeCatalog):
            raise TypeError(
                f"prior_sampling needs a generative catalog (one with a sampler, e.g. type: lample_charton); "
                f"got {type(built).__name__}")
        self.catalog = built
        self.engine = engine
        self.tokenizer = tokenizer
        self.decontaminate = bool(decontaminate)
        self.rng = np.random.default_rng(seed)
        vocab = getattr(tokenizer, "token2idx", {})
        self._wrap = "<expression>" in vocab and "</expression>" in vocab
        self.n_attempts = 0
        self.n_discarded = 0

    def _relabel(self, expression: list[str], n_variables: int) -> list[str] | None:
        """Map the draw's distinct variables onto a random injection into ``x1..x<n_variables>``;
        ``None`` when the draw uses more variables than the problem has columns."""
        seen: list[str] = []
        for token in expression:
            if token.startswith("x") and token[1:].isdigit() and token not in seen:
                seen.append(token)
        if len(seen) > n_variables:
            return None
        targets = self.rng.choice(n_variables, size=len(seen), replace=False)
        mapping = {old: f"x{int(new) + 1}" for old, new in zip(seen, targets)}
        return [mapping.get(token, token) for token in expression]

    def draw_one(self, n_variables: int | None = None) -> tuple[list[int], list[str]] | None:
        """One prior candidate as ``(token ids, masked expression)``, or ``None`` for an unusable draw."""
        self.n_attempts += 1
        try:
            skeleton, _code, constants = self.catalog.sample_skeleton(
                new=True, decontaminate=self.decontaminate, rng=self.rng)
        except NoValidSampleFoundError:
            self.n_discarded += 1
            return None
        expression = substitute_constants(list(skeleton), values=list(constants), inplace=False)
        if n_variables is not None:
            relabeled = self._relabel(expression, int(n_variables))
            if relabeled is None:
                self.n_discarded += 1
                return None
            expression = relabeled
        try:
            masked, values = mask_literals_positional(self.engine, expression)
            placeheld = fittable_slots(self.engine, expression)
        except NonFiniteExpressionError:
            self.n_discarded += 1
            return None
        if len(placeheld) != len(values) or any(not math.isfinite(v) for v in values):
            self.n_discarded += 1
            return None
        # The emission format under the fittable flag: a fittable literal is the model's
        # <constant> placeholder, a structural literal (pow exponent, rootn index) is spelled.
        tokens, _numeric = serialize_constant_tokens(
            masked, [None if kept else value for value, kept in zip(values, placeheld)])
        body = ["<expression>", *tokens, "</expression>"] if self._wrap else tokens
        try:
            ids = self.tokenizer.encode(body, oov="raise")
        except KeyError:
            self.n_discarded += 1
            return None
        return [int(i) for i in ids], list(masked)

    def draw(self, choices: int, *, unique: bool = True, valid_only: bool = True,
             max_tries: int | None = None, n_variables: int | None = None,
             ) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
        """``choices`` candidates in the ``generate`` contract: ``(beams, log_probs, completed, rewards)``.

        A prior draw carries no log-probability (``nan``); ``unique`` keys on the token ids, so two
        draws that differ only in a spelled literal are two candidates. ``n_variables`` conditions
        the draws on the problem's input count (see the module docstring). ``max_tries`` bounds the
        attempts (default: 32 per requested candidate, at least 256, since a low-dimensional problem
        rejects most raw draws).
        """
        target = int(choices)
        budget = int(max_tries) if max_tries is not None else max(256, 32 * target)
        beams: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len(beams) < target and attempts < budget:
            attempts += 1
            drawn = self.draw_one(n_variables)
            if drawn is None:
                continue
            ids, masked = drawn
            if valid_only and not self.engine.is_valid(masked):
                self.n_discarded += 1
                continue
            key = tuple(ids)
            if unique:
                if key in seen:
                    continue
                seen.add(key)
            beams.append(ids)
        n = len(beams)
        return beams, [float("nan")] * n, [True] * n, [float("nan")] * n


__all__ = ["CATALOG_FILENAME", "PriorSampler", "resolve_prior_catalog"]
