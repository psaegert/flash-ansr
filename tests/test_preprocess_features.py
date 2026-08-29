import math
from pathlib import Path

import pytest
from simplipy import SimpliPyEngine

from flash_ansr.model.tokenizer import Tokenizer
from flash_ansr.preprocessing.pipeline import FlashANSRPreprocessor
from flash_ansr import LampleChartonCatalog


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "test"


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(str(CONFIG_DIR / "tokenizer.yaml"))


@pytest.fixture(scope="module")
def catalog() -> LampleChartonCatalog:
    return LampleChartonCatalog.from_config(str(CONFIG_DIR / "catalog_test.yaml"))


def test_preprocessor_prompt_mask_disabled(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    catalog: LampleChartonCatalog,
) -> None:
    preprocessor = FlashANSRPreprocessor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        catalog=catalog,
        prompt_config={
            'section_probs': {
                'prompt': 0.0,
            }
        },
    )

    instance = {
        'input_ids': [
            tokenizer['<bos>'],
            *tokenizer.encode(['+', 'x1', 'x2']),
            tokenizer['<eos>'],
        ],
        'skeletons': ['+', 'x1', 'x2'],
    }

    formatted = preprocessor._format_single(instance)

    assert 'prompt_mask' in formatted
    assert len(formatted['prompt_mask']) == len(formatted['input_ids'])
    assert all(flag is False for flag in formatted['prompt_mask'])


def test_serialize_prompt_prefix(
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    catalog: LampleChartonCatalog,
) -> None:
    preprocessor = FlashANSRPreprocessor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        catalog=catalog,
    )

    serialized = preprocessor.serialize_prompt_prefix(complexity=5)

    # This fixture is a v24 vocabulary (it carries <ieee754>), and v24 training emitted the
    # complexity block BARE -- `<complexity> <float> </complexity>` as a prefix element, never
    # inside a `<prompt>` wrapper. The wrapper costs two token ids that carry no training signal
    # for such a checkpoint. The term sections are withdrawn from the surface and emit nothing.
    tokens = [tokenizer[idx] for idx in serialized['input_ids']]
    assert tokens == ['<bos>', '<complexity>', '<float>', '</complexity>', '<expression>']
    assert '<prompt>' not in tokens
    assert '<allowed_term>' not in tokens
    assert serialized['prompt_disabled'] is False
    assert serialized['missing_tokens'] == []

    numeric = serialized['input_num']
    assert len(numeric) == len(tokens)
    assert math.isnan(numeric[0])
    assert numeric[2] == pytest.approx(5.0)   # mu rides the <float> position, as in training
    assert all(math.isnan(value) for idx, value in enumerate(numeric) if idx != 2)

    mask = serialized['prompt_mask']
    assert len(mask) == len(tokens)
    assert mask[0] is False  # <bos>
    assert all(mask[i] is True for i in range(1, len(tokens) - 1))
    assert mask[-1] is False  # <expression>

    # The term collections are withdrawn: documented as constraining generation,
    # emitting tokens no v24 checkpoint saw, enforced nowhere at decode time.
    assert serialized['prompt_metadata'] == {}


def test_serialize_prompt_prefix_without_prompt_tokens(
    simplipy_engine: SimpliPyEngine,
    catalog: LampleChartonCatalog,
) -> None:
    tokenizer = Tokenizer(
        vocab=['+', 'x1'],
        special_tokens=[
            '<pad>',
            '<bos>',
            '<eos>',
            '<unk>',
            '<cls>',
            '<mask>',
            '<constant>',
            '<expression>',
        ],
    )

    preprocessor = FlashANSRPreprocessor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        catalog=catalog,
    )

    serialized = preprocessor.serialize_prompt_prefix(complexity=5)

    tokens = [tokenizer[idx] for idx in serialized['input_ids']]
    assert tokens == ['<bos>', '<expression>']

    assert all(flag is False for flag in serialized['prompt_mask'])
    assert serialized['prompt_disabled'] is True
    # A vocabulary without the complexity trio simply gets no block; the <prompt> wrapper is
    # no longer part of the contract, so it is not reported missing either.
    assert serialized['missing_tokens'] == []

    assert serialized['prompt_metadata'] == {}
