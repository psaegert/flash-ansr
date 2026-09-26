"""Stand-ins for the hybrid's tests: a toy problem and a fake FlashANSR."""
import time as _time
from types import SimpleNamespace

import numpy as np


def toy_problem(seed: int = 0, n_support: int = 48, n_validation: int = 12):
    """A two-variable problem: X, y, X_val, y_val."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_support, 2))
    x_val = rng.uniform(-2.0, 2.0, size=(n_validation, 2))
    law = lambda a: 1.5 * a[:, 0] - 0.5 * a[:, 1] + 2.0  # noqa: E731
    return x, law(x), x_val, law(x_val)


def fake_model(seconds_per_candidate: float, call_overhead: float, engine=None, ranking=None):
    """A stand-in for ``FlashANSR`` (0.17): ``fit(draws=n)`` sleeps for its chunk and returns ``n`` distinct
    candidates of the form ``c * x1`` in score order (rank 0 = best of the chunk, scores worsening
    slightly with every draw), evaluable through ``result.predict``; ``ranking`` is an MDL ranking."""

    def fit(X, y, *, draws=None, **kw):
        n = int(draws if draws is not None else model.generation_config.draws)
        _time.sleep(call_overhead + seconds_per_candidate * n)
        cands = []
        for i in range(n):
            j = fit.counter + i
            cands.append(SimpleNamespace(
                raw_beam=[j], expression=["*", "<constant>", "x1"], expression_prefix=["*", str(1.0 + 1e-3 * j), "x1"],
                expression_infix=f"{1.0 + 1e-3 * j} * x1", skeleton_prefix=["*", "<constant>", "x1"], constants=[1.0 + 1e-3 * j],
                score=-1.0 + 1e-6 * j, fvu=1e-3, mdl=1000.0, n_nodes=3, log_prob=-1.0, pareto_rank=-1,
                spelling=None, rank=i))
        fit.counter += n
        return SimpleNamespace(candidates=cands, generation_time=seconds_per_candidate * n, refinement_time=0.0,
                               predict=lambda X, rank=0: np.ones(np.asarray(X).shape[0]))

    fit.counter = 0
    model = SimpleNamespace(
        fit=fit, generation_config=SimpleNamespace(draws=1024, emission="fittable"),
        simplipy_engine=engine if engine is not None else SimpleNamespace(operator_arity={"*": 2, "+": 2}),
        ranking=SimpleNamespace(as_dict=lambda: dict(ranking or {"mode": "mdl", "mdl_strength": 1e-2})),
        n_restarts=8, refiner_method="curve_fit_lm", refiner_p0_noise="normal", refiner_p0_noise_kwargs={"loc": 0.0, "scale": 5},
        refine_scope="fittable", constant_ladder=True,
    )
    return model
