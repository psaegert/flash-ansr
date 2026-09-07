"""``prior_sampling``: candidates from the training prior instead of the decoder, through the
unchanged refinement and ranking. The generation config, the sampler's emission format, and an
end-to-end ``infer`` on a model whose weights are never consulted for the candidates."""
import numpy as np
import pytest
from simplipy import SimpliPyEngine

from flash_ansr import FlashANSR, FlashANSRModel, Tokenizer
from flash_ansr.data.serialization import replace_ieee754_spans_with_constants
from flash_ansr.prior import PriorSampler, resolve_prior_catalog
from flash_ansr.scoring import RankingError
from flash_ansr.utils.generation import PriorSamplingConfig, SoftmaxSamplingConfig, create_generation_config
from flash_ansr.utils.ieee754 import BYTE_TOKENS, IEEE754_END_TOKEN, IEEE754_START_TOKEN
from flash_ansr.utils.paths import get_path


def _catalog_config() -> dict:
    """A small generation-2 catalog over three variables: arithmetic, sin, and pow with a typed exponent."""
    return {
        "type": "lample_charton",
        "simplipy_engine": "base",
        "holdout_pools": [],
        "sample_strategy": {
            "n_operator_distribution": "length_proportional",
            "min_operators": 1, "max_operators": 5, "power": 1,
            "max_length": 21, "max_tries": 4, "independent_dimensions": True,
        },
        "allow_nan": False,
        "simplify": True,
        "literal_prior": {"name": "normal", "kwargs": {"loc": 0, "scale": 5}},
        "support_sampler": {
            "support_prior": {"name": "uniform", "kwargs": {"low": -5, "high": 5, "min_value": -5, "max_value": 5}},
            "n_support_prior": {"name": "uniform", "kwargs": {"low": 4, "high": 16, "min_value": 4, "max_value": 16}},
        },
        "variables": ["x1", "x2", "x3"],
        "operator_weights": {"+": 10, "-": 10, "*": 10, "sin": 2, "pow": 2},
    }


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("base")


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_config(get_path("configs", "test", "tokenizer.yaml"))


class TestConfig:
    def test_factory_dispatches_and_refuses_retired_methods(self):
        config = create_generation_config(method="prior_sampling", choices=8, seed=3)
        assert isinstance(config, PriorSamplingConfig)
        assert config.method == "prior_sampling"
        assert config.to_kwargs() == {"choices": 8, "unique": True, "valid_only": True, "catalog": None,
                                      "decontaminate": True, "match_variables": True, "seed": 3, "max_tries": None}
        assert isinstance(create_generation_config(method="softmax_sampling", choices=2), SoftmaxSamplingConfig)
        with pytest.raises(ValueError, match="retired"):
            create_generation_config(method="beam_search")
        with pytest.raises(ValueError):
            create_generation_config(method="prior_sampling", choices=0)

    def test_catalog_resolution(self, tmp_path):
        assert resolve_prior_catalog({"type": "lample_charton"}, None) == {"type": "lample_charton"}
        with pytest.raises(ValueError, match="training prior"):
            resolve_prior_catalog(None, None)
        with pytest.raises(FileNotFoundError):
            resolve_prior_catalog(None, str(tmp_path))
        (tmp_path / "catalog_train.yaml").write_text("type: lample_charton\n")
        assert resolve_prior_catalog(None, str(tmp_path)) == str(tmp_path / "catalog_train.yaml")


class TestSampler:
    def test_draws_are_unique_valid_and_in_the_emission_format(self, engine, tokenizer):
        sampler = PriorSampler(_catalog_config(), engine=engine, tokenizer=tokenizer, seed=0)
        beams, log_probs, completed, rewards = sampler.draw(24)
        assert len(beams) == 24 and len({tuple(b) for b in beams}) == 24
        assert all(np.isnan(lp) for lp in log_probs) and all(completed) and all(np.isnan(r) for r in rewards)
        byte_ids = [int(tokenizer[token]) for token in BYTE_TOKENS]
        n_placeholders = n_spans = 0
        for beam in beams:
            body = tokenizer.extract_expression_from_beam(beam)[0]
            mapped, span_values = replace_ieee754_spans_with_constants(
                body, start_id=int(tokenizer[IEEE754_START_TOKEN]), end_id=int(tokenizer[IEEE754_END_TOKEN]),
                byte_ids=byte_ids, constant_id=int(tokenizer["<constant>"]))
            tokens = tokenizer.decode_expression(mapped)
            assert engine.is_valid(tokens), tokens
            raw = tokenizer.decode_expression(body)
            n_placeholders += raw.count("<constant>")
            n_spans += len(span_values or [])
        # fittable literals travel as <constant> placeholders; a spelled literal only rides a span
        assert n_placeholders > 0
        assert sampler.n_attempts >= 24

    def test_variable_matching_relabels_onto_the_problem_columns(self, engine, tokenizer):
        sampler = PriorSampler(_catalog_config(), engine=engine, tokenizer=tokenizer, seed=2)
        beams, *_ = sampler.draw(20, n_variables=1)
        used = set()
        for beam in beams:
            tokens = tokenizer.decode_expression(tokenizer.extract_expression_from_beam(beam)[0])
            used |= {t for t in tokens if t.startswith("x") and t[1:].isdigit()}
        assert used <= {"x1"}
        sampler = PriorSampler(_catalog_config(), engine=engine, tokenizer=tokenizer, seed=2)
        beams, *_ = sampler.draw(40, n_variables=2)
        used = set()
        for beam in beams:
            tokens = tokenizer.decode_expression(tokenizer.extract_expression_from_beam(beam)[0])
            used |= {t for t in tokens if t.startswith("x") and t[1:].isdigit()}
        assert used == {"x1", "x2"}  # a random injection, so both columns appear across 40 draws
        assert sampler.n_discarded > 0  # three-variable draws are rejected for a two-column problem

    def test_seed_fixes_the_draw_stream(self, engine, tokenizer):
        a = PriorSampler(_catalog_config(), engine=engine, tokenizer=tokenizer, seed=7).draw(6)[0]
        b = PriorSampler(_catalog_config(), engine=engine, tokenizer=tokenizer, seed=7).draw(6)[0]
        assert a == b

    def test_max_tries_bounds_the_attempts(self, engine, tokenizer):
        sampler = PriorSampler(_catalog_config(), engine=engine, tokenizer=tokenizer, seed=1)
        beams, *_ = sampler.draw(1000, max_tries=20)
        assert len(beams) <= 20 and sampler.n_attempts == 20


class TestInfer:
    def _model(self, engine, tokenizer, **generation) -> FlashANSR:
        model = FlashANSRModel.from_config(get_path("configs", "test", "model.yaml"))  # random weights: never used for the draws
        return FlashANSR(
            simplipy_engine=engine, flash_ansr_model=model, tokenizer=tokenizer,
            generation_config=create_generation_config(method="prior_sampling", catalog=_catalog_config(), **generation),
            n_restarts=2, refiner_workers=0, model_directory=None)

    def test_infer_fits_prior_candidates_to_the_data(self, engine, tokenizer):
        nsr = self._model(engine, tokenizer, choices=48, seed=0)
        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, size=(64, 2))
        y = (2.0 * X[:, 0] - 0.5 * X[:, 1] + 1.0).reshape(-1, 1)
        result = nsr.infer(X, y, emission="constants", top_k="all")  # the test vocabulary has no emission flags
        assert len(result.ledger) >= len(result.candidates) > 0
        assert all(np.isnan(c.log_prob) for c in result.candidates)  # the prior carries no log-probability
        assert result.candidates[0].score <= result.candidates[-1].score  # the ranking's order (MDL: fvu plus a length price)
        assert np.isfinite(result.candidates[0].fvu)
        assert result.generation_time >= 0.0

    def test_prior_mode_refuses_a_log_prob_ranking(self, engine, tokenizer):
        model = FlashANSRModel.from_config(get_path("configs", "test", "model.yaml"))
        nsr = FlashANSR(
            simplipy_engine=engine, flash_ansr_model=model, tokenizer=tokenizer,
            generation_config=create_generation_config(method="prior_sampling", catalog=_catalog_config(), choices=4),
            ranking_mode="weighted", ranking_weights={"n_nodes": 0.05, "neg_log_prob": 0.1}, refiner_workers=0)
        with pytest.raises(RankingError, match="log-probability"):
            nsr.infer(np.ones((8, 2)), np.ones((8, 1)), emission="constants")
