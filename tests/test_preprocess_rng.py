"""Completeness test: prompt feature extraction must be fully controlled by the
injected ``numpy.random.Generator`` with NO leak to global ``np.random`` / stdlib
``random`` state.

The two files under test (``feature_extractor.py`` and ``pipeline.py``) were
refactored so that every source of randomness threads through a single injected
``Generator``. These tests prove that perturbing the global RNG states between two
otherwise-identical runs does not change the output: only the injected Generator
matters.
"""
import random
from pathlib import Path

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing.feature_extractor import (
    AllowedTermsConfig,
    DistributionSpec,
    ExcludeTermsConfig,
    IncludeTermsConfig,
    PromptFeatureExtractor,
    PromptFeatureExtractorConfig,
)
from flash_ansr.preprocessing.pipeline import FlashANSRPreprocessor
from flash_ansr import LampleChartonCatalog


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "test"


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("dev_7-3", install=True)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(str(CONFIG_DIR / "tokenizer.yaml"))


@pytest.fixture(scope="module")
def skeleton_pool() -> LampleChartonCatalog:
    return LampleChartonCatalog.from_config(str(CONFIG_DIR / "skeleton_pool_test.yaml"))


def _exercising_config() -> PromptFeatureExtractorConfig:
    """A config that forces EVERY random path to fire:

    - fractional section probabilities -> ``_is_section_enabled`` / ``_should_include``
      use ``self._rng.random()``
    - ``uniform_int`` distributions -> ``DistributionSpec.sample`` integer path
    - multiple actual terms -> the in-place shuffle (permutation) path
    - ``generated_terms`` >= 1 and ``exclude`` count >= 1 -> ``sample_skeleton(rng=...)``
      + ``_generate_term_via_skeleton_pool`` shuffle + choice paths
    - include selection -> ``_select_term_by_length`` choice path
    """
    return PromptFeatureExtractorConfig(
        prompt_probability=0.85,
        allowed_terms=AllowedTermsConfig(
            probability=0.85,
            actual_terms=DistributionSpec.from_dict(
                {"distribution": "uniform_int", "params": {"low": 2, "high": 5}}
            ),
            generated_terms=DistributionSpec.from_dict(
                {"distribution": "uniform_int", "params": {"low": 1, "high": 3}}
            ),
            length=DistributionSpec.from_dict(
                {"distribution": "uniform_int", "params": {"low": 1, "high": 4}}
            ),
            min_length=1,
            max_relative_length=1.0,
            force_expression_term=True,
        ),
        include_terms=IncludeTermsConfig(
            probability=0.7,
            count=DistributionSpec.from_dict(
                {"distribution": "uniform_int", "params": {"low": 1, "high": 3}}
            ),
            length=DistributionSpec.from_dict(
                {"distribution": "uniform_int", "params": {"low": 1, "high": 3}}
            ),
            min_length=1,
            max_relative_length=1.0,
        ),
        exclude_terms=ExcludeTermsConfig(
            probability=0.7,
            count=DistributionSpec.from_dict(
                {"distribution": "uniform_int", "params": {"low": 1, "high": 3}}
            ),
            length=DistributionSpec.from_dict(
                {"distribution": "uniform_int", "params": {"low": 1, "high": 3}}
            ),
            min_length=1,
            max_relative_length=1.0,
        ),
        max_random_term_attempts=16,
    )


EXPRESSIONS = [
    ['+', 'x1', 'x2'],
    ['*', '+', 'x1', 'x2', 'x3'],
    ['/', '*', 'x1', 'x2', '+', 'x3', 'x4'],
    ['-', 'sin', 'x1', 'cos', 'x2'],
    ['+', '*', 'x1', 'x2', '*', 'x3', 'x4'],
]


def _run_extractor_features(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: LampleChartonCatalog,
    seed: int,
) -> list[tuple]:
    extractor = PromptFeatureExtractor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        config=_exercising_config(),
        skeleton_pool=skeleton_pool,
        rng=np.random.default_rng(seed),
    )

    results: list[tuple] = []
    # Many draws per expression so any leaked global-RNG draw would diverge.
    for _ in range(8):
        for expression in EXPRESSIONS:
            features = extractor.extract(expression)
            results.append(
                (
                    tuple(tuple(term) for term in features.allowed_terms),
                    tuple(tuple(term) for term in features.include_terms),
                    tuple(tuple(term) for term in features.exclude_terms),
                )
            )
    return results


def test_extractor_output_invariant_to_global_rng(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: LampleChartonCatalog,
) -> None:
    # Run 1: clean global state.
    np.random.seed(0)
    random.seed(0)
    first = _run_extractor_features(simplipy_engine, tokenizer, skeleton_pool, seed=123)

    # Run 2: perturb BOTH global RNG states, fresh injected Generator with same seed.
    np.random.seed(999)
    random.seed(999)
    _ = [np.random.random() for _ in range(37)]  # advance global numpy state further
    _ = [random.random() for _ in range(37)]     # advance global stdlib state further
    second = _run_extractor_features(simplipy_engine, tokenizer, skeleton_pool, seed=123)

    # Identical output despite different global RNG states => no global RNG leak.
    assert first == second
    # Sanity: the random paths actually produced varied, non-trivial output.
    assert len(set(first)) > 1


def test_preprocessor_should_include_invariant_to_global_rng(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: LampleChartonCatalog,
) -> None:
    # _should_include drives serialization-section inclusion via self._rng.random().
    def _collect(seed: int) -> list[tuple[bool, bool, bool, bool]]:
        preprocessor = FlashANSRPreprocessor(
            simplipy_engine=simplipy_engine,
            tokenizer=tokenizer,
            skeleton_pool=skeleton_pool,
            prompt_config={
                'section_probs': {
                    'prompt': 0.5,
                    'complexity': 0.5,
                    'allowed_terms': 0.5,
                    'include_terms': 0.5,
                    'exclude_terms': 0.5,
                },
            },
            rng=np.random.default_rng(seed),
        )
        return [
            (
                preprocessor._should_include('prompt'),
                preprocessor._should_include('complexity'),
                preprocessor._should_include('allowed_terms'),
                preprocessor._should_include('exclude_terms'),
            )
            for _ in range(200)
        ]

    np.random.seed(0)
    random.seed(0)
    first = _collect(seed=123)

    np.random.seed(999)
    random.seed(999)
    _ = [np.random.random() for _ in range(37)]
    _ = [random.random() for _ in range(37)]
    second = _collect(seed=123)

    assert first == second
    # Sanity: the fractional probability actually produced a mix of decisions.
    flattened = [flag for row in first for flag in row]
    assert any(flattened) and not all(flattened)


def test_distribution_spec_sample_invariant_to_global_rng() -> None:
    # DistributionSpec.sample must use ONLY the passed Generator.
    specs = [
        DistributionSpec.from_dict({"distribution": "uniform_int", "params": {"low": 0, "high": 10}}),
        DistributionSpec.from_dict({"distribution": "uniform", "params": {"low": -3.0, "high": 3.0}}),
        DistributionSpec.from_dict({"distribution": "poisson", "params": {"lam": 2.5}}),
        DistributionSpec.from_dict({"distribution": "geometric", "params": {"p": 0.3}}),
        DistributionSpec.from_dict({"distribution": "normal", "params": {"mean": 1.0, "std": 2.0}}),
        DistributionSpec.from_dict({"distribution": "triangular", "params": {"left": 0.0, "mode": 1.0, "right": 2.0}}),
    ]

    def _draw(seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        return [spec.sample(rng) for spec in specs for _ in range(50)]

    np.random.seed(0)
    random.seed(0)
    first = _draw(seed=123)

    np.random.seed(999)
    random.seed(999)
    _ = [np.random.random() for _ in range(37)]
    _ = [random.random() for _ in range(37)]
    second = _draw(seed=123)

    assert first == second

    # uniform_int [0, 10] inclusive: the +1 in integers(low, high+1) must be preserved.
    int_spec = specs[0]
    rng = np.random.default_rng(7)
    draws = [int(int_spec.sample(rng)) for _ in range(5000)]
    assert min(draws) == 0
    assert max(draws) == 10
