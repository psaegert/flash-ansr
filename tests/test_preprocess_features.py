import math
import random
from pathlib import Path

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocess import FlashASNRPreprocessor
from flash_ansr.preprocess_features import (
    PromptFeatureExtractor,
    PromptFeatureExtractorConfig,
)
from flash_ansr.expressions.skeleton_pool import SkeletonPool


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "test"


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("dev_7-3", install=True)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(str(CONFIG_DIR / "tokenizer.yaml"))


@pytest.fixture(scope="module")
def skeleton_pool() -> SkeletonPool:
    return SkeletonPool.from_config(str(CONFIG_DIR / "skeleton_pool_test.yaml"))


def test_extract_basic_prompt_features(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: SkeletonPool,
) -> None:
    random.seed(0)
    np.random.seed(0)
    config = PromptFeatureExtractorConfig(
        max_cover_terms=8,
        extra_allowed_range=(0, 0),
        include_terms_range=(1, 1),
        exclude_terms_range=(1, 1),
        max_random_term_attempts=8,
    )

    extractor = PromptFeatureExtractor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        config=config,
        skeleton_pool=skeleton_pool,
    )

    expression = ['+', 'x1', 'x2']
    features = extractor.extract(expression)

    assert features.expression_tokens == expression
    assert features.complexity == len(expression)
    assert expression in features.allowed_terms
    assert len(features.include_terms) == 1
    assert all(tuple(term) in {tuple(t) for t in features.allowed_terms} for term in features.include_terms)
    assert all(
        tuple(term) not in {tuple(t) for t in features.allowed_terms} for term in features.exclude_terms
    )


def test_exclude_terms_skip_existing_terms(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: SkeletonPool,
) -> None:
    random.seed(1)
    np.random.seed(1)
    config = PromptFeatureExtractorConfig(
        max_cover_terms=8,
        extra_allowed_range=(0, 0),
        include_terms_range=(0, 0),
        exclude_terms_range=(1, 1),
        max_random_term_attempts=8,
    )

    extractor = PromptFeatureExtractor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        config=config,
        skeleton_pool=skeleton_pool,
    )

    expression = ['+', 'x1', 'x2']
    features = extractor.extract(expression)

    allowed = {tuple(term) for term in features.allowed_terms}
    include = {tuple(term) for term in features.include_terms}
    assert all(tuple(term) not in allowed for term in features.exclude_terms)
    assert all(tuple(term) not in include for term in features.exclude_terms)
    assert len(features.exclude_terms) <= 1


def test_include_terms_subset_of_allowed(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: SkeletonPool,
) -> None:
    random.seed(2)
    np.random.seed(2)
    config = PromptFeatureExtractorConfig(
        max_cover_terms=8,
        extra_allowed_range=(1, 1),
        include_terms_range=(2, 2),
        exclude_terms_range=(1, 1),
        max_random_term_attempts=8,
    )

    extractor = PromptFeatureExtractor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        config=config,
        skeleton_pool=skeleton_pool,
    )

    expression = ['*', '+', 'x1', 'x2', 'x3']
    features = extractor.extract(expression)

    include_set = {tuple(term) for term in features.include_terms}
    allowed_set = {tuple(term) for term in features.allowed_terms}
    assert include_set.issubset(allowed_set)
    assert all(tuple(term) not in include_set and tuple(term) not in allowed_set for term in features.exclude_terms)
    assert len(features.include_terms) >= 1
    assert len(features.allowed_terms) >= len(include_set)


def test_preprocessor_prompt_mask_alignment(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: SkeletonPool,
) -> None:
    random.seed(3)
    np.random.seed(3)

    preprocessor = FlashASNRPreprocessor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        skeleton_pool=skeleton_pool,
        prompt_config={
            'prompt_feature': {
                'max_cover_terms': 8,
                'extra_allowed_range': (0, 0),
                'include_terms_range': (0, 0),
                'exclude_terms_range': (0, 0),
                'max_random_term_attempts': 8,
            },
        },
    )

    features = preprocessor._feature_extractor.extract(['+', 'x1', 'x2'])  # type: ignore[union-attr]
    serialized = preprocessor._serialize_prompt(features)

    tokens = [tokenizer[idx] for idx in serialized['input_ids']]
    mask = serialized['prompt_mask']

    assert len(tokens) == len(mask)
    assert tokens[0] == '<bos>'
    assert mask[0] is False
    assert '<prompt>' in tokens
    assert '</prompt>' in tokens

    start = tokens.index('<prompt>')
    end = len(tokens) - 1 - tokens[::-1].index('</prompt>')

    assert all(mask[i] for i in range(start, end + 1))
    assert not any(mask[i] for i in range(0, start))
    assert not any(mask[i] for i in range(end + 1, len(tokens)))


def test_preprocessor_prompt_mask_disabled(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: SkeletonPool,
) -> None:
    preprocessor = FlashASNRPreprocessor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        skeleton_pool=skeleton_pool,
        prompt_config={
            'section_probs': {
                'prompt': 0.0,
            }
        },
    )

    instance = {
        'input_ids': tokenizer.encode(['+', 'x1', 'x2'], add_bos=True, add_eos=True),
        'skeletons': ['+', 'x1', 'x2'],
    }

    formatted = preprocessor._format_single(instance)

    assert 'prompt_mask' in formatted
    assert len(formatted['prompt_mask']) == len(formatted['input_ids'])
    assert all(flag is False for flag in formatted['prompt_mask'])


def test_serialize_prompt_prefix(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: SkeletonPool,
) -> None:
    preprocessor = FlashASNRPreprocessor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        skeleton_pool=skeleton_pool,
    )

    serialized = preprocessor.serialize_prompt_prefix(
        complexity=5,
        allowed_terms=[['+', 'x1']],
    )

    tokens = [tokenizer[idx] for idx in serialized['input_ids']]
    assert tokens[:5] == ['<bos>', '<prompt>', '<complexity>', '<float>', '</complexity>']
    assert tokens[5:9] == ['<allowed_term>', '+', 'x1', '</allowed_term>']
    assert tokens[9:] == ['</prompt>', '<expression>']

    numeric = serialized['input_num']
    assert len(numeric) == len(tokens)
    assert math.isnan(numeric[0])
    assert numeric[3] == pytest.approx(5.0)
    assert all(math.isnan(value) for idx, value in enumerate(numeric) if idx != 3)

    mask = serialized['prompt_mask']
    assert len(mask) == len(tokens)
    assert mask[0] is False  # <bos>
    assert all(mask[i] is True for i in range(1, len(tokens) - 1))
    assert mask[-1] is False  # <expression>

    metadata = serialized['prompt_metadata']
    assert metadata['allowed_terms'] == [['+', 'x1']]
    assert metadata['include_terms'] == []
    assert metadata['exclude_terms'] == []
