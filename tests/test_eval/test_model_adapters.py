import numpy as np
import pytest

from flash_ansr.eval.core import EvaluationResult, EvaluationSample
from flash_ansr.eval import model_adapters


class _DummyModel:
    parsimony = None

    def to(self, device: str):  # noqa: D401 - simple stub
        return self

    def eval(self):  # noqa: D401 - simple stub
        return self


class _DummyEngine:
    def infix_to_prefix(self, expression):  # noqa: D401 - simple stub
        return [expression]


@pytest.mark.parametrize(
    "pred_key",
    ["best_bfgs_preds", "best_preds"],
)
def test_extract_first_prediction_handles_empty_sequences(pred_key: str):
    output = {pred_key: [None, None]}
    assert model_adapters._extract_first_prediction(
        output,
        preferred_key=pred_key,
        fallback_key=None,
    ) is None


def test_nesymres_adapter_handles_missing_predictions(monkeypatch):
    monkeypatch.setattr(model_adapters, "_HAVE_NESYMRES", True)

    def failing_fitfunc(X_support, y_fit):  # noqa: D401 - simple stub
        return {"best_bfgs_preds": [], "best_bfgs_consts": []}

    adapter = model_adapters.NeSymReSAdapter(
        model=_DummyModel(),
        fitfunc=failing_fitfunc,
        simplipy_engine=_DummyEngine(),
    )

    sample = EvaluationSample(
        x_support=np.zeros((1, 1), dtype=float),
        y_support=np.zeros((1, 1), dtype=float),
        x_validation=np.zeros((0, 1), dtype=float),
        y_validation=np.zeros((0, 1), dtype=float),
    )

    adapter.prepare()
    result: EvaluationResult = adapter.evaluate_sample(sample)
    mapping = result.to_mapping()

    assert mapping["prediction_success"] is False
    assert "error" in mapping
    assert "no expression" in mapping["error"].lower()
