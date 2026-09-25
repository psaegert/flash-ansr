"""``oracle``: the problem's ground truth as the only candidate, through the unchanged refinement and
ranking. The generation config, the candidate's emission format (the prior sampler's), and end-to-end
``fit`` on a model whose weights are never consulted for the candidate."""
import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import FlashANSR, FlashANSRModel, OracleConfig, Tokenizer
from flash_ansr.oracle import oracle_beams
from flash_ansr.scoring import RankingError
from flash_ansr.utils.generation import create_generation_config
from flash_ansr.utils.ieee754 import IEEE754_END_TOKEN, IEEE754_START_TOKEN
from flash_ansr.utils.paths import get_path

# 2.5 * x1^2 + sin(x2): one fittable literal (2.5), one structural one (the exponent 2)
LAW = ["+", "*", "2.5", "pow", "x1", "2", "sin", "x2"]


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("base")


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


def _spans(tokens):
    """The float64 values of the ieee754 spans in a token list (8 byte tokens, big-endian)."""
    import struct
    values, i = [], 0
    while i < len(tokens):
        if tokens[i] == IEEE754_START_TOKEN:
            j = tokens.index(IEEE754_END_TOKEN, i)
            raw = bytes(int(token[2:4], 16) for token in tokens[i + 1:j])
            values.append(struct.unpack(">d", raw)[0])
            i = j
        i += 1
    return values


class TestConfig:
    def test_factory_builds_the_oracle(self):
        config = create_generation_config(method="oracle", expression=LAW)
        assert isinstance(config, OracleConfig) and config.method == "oracle"
        assert config.to_kwargs() == {"expression": tuple(LAW)}
        assert OracleConfig().expression is None


class TestBeams:
    def test_the_ground_truth_is_one_beam_in_the_emission_format(self, engine, tokenizer):
        beams, log_probs, completed, rewards = oracle_beams(LAW, engine=engine, tokenizer=tokenizer)
        assert len(beams) == 1 and completed == [True]
        assert np.isnan(log_probs[0]) and np.isnan(rewards[0])  # no log-probability
        tokens = [tokenizer.idx2token[i] for i in beams[0]]
        # the fittable literal travels as the <constant> placeholder the refiner fits ...
        assert tokens.count("<constant>") == 1 and "2.5" not in tokens
        # ... the structural exponent stays spelled: one ieee754 span holding exactly 2
        assert _spans(tokens) == [2.0]

    def test_a_ground_truth_the_vocabulary_cannot_spell_is_no_candidate(self, engine, tokenizer):
        assert oracle_beams(["+", "x999", "1.5"], engine=engine, tokenizer=tokenizer) == ([], [], [], [])


class TestFit:
    def _model(self, engine, tokenizer, **kwargs) -> FlashANSR:
        model = FlashANSRModel.from_config(get_path("configs", "test", "model.yaml"))  # random weights: never used for the candidate
        kwargs.setdefault("refine", {"n_restarts": 4})
        return FlashANSR(simplipy_engine=engine, flash_ansr_model=model, tokenizer=tokenizer,
                         compute={"workers": 0}, model_directory=None, **kwargs)

    def test_fit_recovers_the_law_by_fitting_its_constants(self, engine, tokenizer):
        nsr = self._model(engine, tokenizer, generation_config=OracleConfig(expression=LAW))
        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, size=(64, 2))
        y = (2.5 * X[:, 0] ** 2 + np.sin(X[:, 1])).reshape(-1, 1)
        result = nsr.fit(X, y, seed=0)
        # one beam, the ground truth; refinement may add its usual variants of it (a re-spelled constant)
        assert len({tuple(c.raw_beam) for c in result.candidates}) == 1
        best = result.best
        assert best is not None and np.isnan(best.log_prob)
        assert best.fvu < 1e-12

    def test_fit_refuses_to_run_without_a_ground_truth(self, engine, tokenizer):
        nsr = self._model(engine, tokenizer, generation_config=OracleConfig())
        with pytest.raises(ValueError, match="ground truth"):
            nsr.fit(np.ones((8, 2)), np.ones((8, 1)))

    def test_the_oracle_refuses_a_log_prob_ranking(self, engine, tokenizer):
        nsr = self._model(engine, tokenizer, generation_config=OracleConfig(expression=LAW),
                          ranking={"mode": "weighted", "weights": {"n_nodes": 0.05, "neg_log_prob": 0.1}})
        with pytest.raises(RankingError, match="log-probability"):
            nsr.fit(np.ones((8, 2)), np.ones((8, 1)))
