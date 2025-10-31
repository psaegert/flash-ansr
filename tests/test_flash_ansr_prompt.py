import numpy as np
import pytest
import torch
from pathlib import Path

from simplipy import SimpliPyEngine

from flash_ansr import FlashANSR, GenerationConfig
from flash_ansr.preprocess import FlashASNRPreprocessor
from flash_ansr.model.tokenizer import Tokenizer
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


class _DummyModel:
    def __init__(self, tokenizer: Tokenizer, preprocessor: FlashASNRPreprocessor) -> None:
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.device = torch.device("cpu")
        self.encoder_max_n_variables = 2

    def eval(self) -> "_DummyModel":
        return self


class _DummyRefiner:
    def __init__(self, simplipy_engine: SimpliPyEngine, n_variables: int) -> None:
        self.valid_fit = True
        self.expression_lambda = lambda X: np.zeros((X.shape[0], 1))
        self._all_constants_values = [
            (np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float), 0.1)
        ]

    def fit(self, *args, **kwargs) -> "_DummyRefiner":
        return self

    def predict(self, X: np.ndarray, nth_best_constants: int = 0) -> np.ndarray:
        return np.zeros((X.shape[0], 1), dtype=float)

    def transform(
        self,
        expression: list[str],
        *,
        nth_best_constants: int = 0,
        return_prefix: bool = False,
        precision: int = 2,
        variable_mapping: dict[str, str] | None = None,
        **_: dict,
    ) -> list[str] | str:
        if return_prefix:
            return expression
        return " ".join(expression)


def test_flash_ansr_fit_uses_prompt_prefix(
    monkeypatch: pytest.MonkeyPatch,
    simplipy_engine: SimpliPyEngine,
    tokenizer: Tokenizer,
    skeleton_pool: SkeletonPool,
) -> None:
    preprocessor = FlashASNRPreprocessor(
        simplipy_engine=simplipy_engine,
        tokenizer=tokenizer,
        skeleton_pool=skeleton_pool,
    )

    model = _DummyModel(tokenizer=tokenizer, preprocessor=preprocessor)

    captured: dict[str, list[float] | list[int] | None] = {}

    def _fake_generate(
        self: FlashANSR,
        data: torch.Tensor,
        complexity: int | float | None = None,
        verbose: bool = False,
        prompt_tokens: list[int] | None = None,
        prompt_numeric: list[float] | None = None,
    ) -> tuple[list[list[int]], list[float], list[bool], list[float]]:
        captured['prompt_tokens'] = list(prompt_tokens) if prompt_tokens is not None else None
        captured['prompt_numeric'] = list(prompt_numeric) if prompt_numeric is not None else None

        beam_tokens = [
            tokenizer['<bos>'],
            tokenizer['<expression>'],
            tokenizer['x1'],
            tokenizer['</expression>'],
            tokenizer['<eos>'],
        ]
        return [beam_tokens], [0.0], [True], [float('nan')]

    monkeypatch.setattr(FlashANSR, 'generate', _fake_generate)
    monkeypatch.setattr('flash_ansr.flash_ansr.Refiner', _DummyRefiner)

    ansr = FlashANSR(
        simplipy_engine=simplipy_engine,
        flash_ansr_transformer=model,
        tokenizer=tokenizer,
        generation_config=GenerationConfig(method='beam_search', beam_width=1, max_len=8),
        n_restarts=1,
    )

    X = np.ones((4, 1), dtype=float)
    y = np.ones((4, 1), dtype=float)

    ansr.fit(
        X,
        y,
        prompt_complexity=7,
        prompt_allowed_terms=[["x1"]],
        prompt_include_terms=[["x1"]],
        prompt_exclude_terms=[["x2"]],
    )

    assert 'prompt_tokens' in captured and captured['prompt_tokens'] is not None
    assert 'prompt_numeric' in captured and captured['prompt_numeric'] is not None

    prompt_tokens = [tokenizer[idx] for idx in captured['prompt_tokens']]  # type: ignore[arg-type]
    expected_prefix = [
        '<bos>',
        '<prompt>',
        '<complexity>',
        '<float>',
        '</complexity>',
        '<allowed_term>',
        'x1',
        '</allowed_term>',
        '<include_term>',
        'x1',
        '</include_term>',
        '<exclude_term>',
        'x2',
        '</exclude_term>',
        '</prompt>',
        '<expression>',
    ]
    assert prompt_tokens[: len(expected_prefix)] == expected_prefix

    prompt_numeric = captured['prompt_numeric']  # type: ignore[assignment]
    assert prompt_numeric[3] == pytest.approx(7.0)
    assert all(
        np.isnan(value) for idx, value in enumerate(prompt_numeric) if idx != 3
    )

    assert ansr._prompt_metadata is not None
    assert ansr._prompt_metadata['allowed_terms'] == [['x1']]
    assert ansr._prompt_metadata['include_terms'] == [['x1']]
    assert ansr._prompt_metadata['exclude_terms'] == [['x2']]

    assert 'prompt_metadata' in ansr.results
    assert not ansr.results['prompt_metadata'].isnull().all()