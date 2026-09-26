"""The seed bridge to Julia syntax, the engine-side evaluation, the data fingerprint."""
import math

import numpy as np

from flash_ansr.hybrid.bridge import PYSR_OPERATORS, data_fingerprint, evaluate_prefix, prefix_to_julia
from flash_ansr.hybrid.pysr_model import BINARY_OPERATORS, UNARY_OPERATORS

from hybrid_fakes import toy_problem

ARITY = {"+": 2, "-": 2, "*": 2, "/": 2, "pow": 2, "rootn": 2, "neg": 1, "inv": 1, "abs": 1,
         "sin": 1, "cos": 1, "exp": 1, "log": 1}


class TestBridge:
    def test_renders_pysr_syntax(self):
        assert prefix_to_julia(["*", "2.5", "pow", "x1", "2"], ARITY, ["v1", "v2"]) == "(2.5 * (v1 ^ 2.0))"
        assert prefix_to_julia(["neg", "+", "x1", "1.0"], ARITY, ["v1"]) == "neg((v1 + 1.0))"
        assert prefix_to_julia(["rootn", "x2", "3"], ARITY, ["v1", "v2"]) == "rootn(v2, 3.0)"
        assert prefix_to_julia(["inv", "sin", "x1"], ARITY, ["a"]) == "inv(sin(a))"

    def test_negative_literals_and_special_constants(self):
        text = prefix_to_julia(["+", "*", "-1.5e-09", "x1", "np.pi"], ARITY, ["v1"])
        assert text == "(((-1.5e-09) * v1) + 3.141592653589793)"
        assert prefix_to_julia(["*", "np.e", "x1"], ARITY, ["v1"]) == f"({math.e!r} * v1)"

    def test_refuses_what_pysr_cannot_read(self):
        assert prefix_to_julia(["sqrt", "x1"], ARITY, ["v1"]) is None          # outside the vocabulary
        assert prefix_to_julia(["*", "<constant>", "x1"], ARITY, ["v1"]) is None  # an unrealized slot
        assert prefix_to_julia(["*", "x3", "x1"], ARITY, ["v1", "v2"]) is None    # a variable the problem lacks
        assert prefix_to_julia(["+", "x1"], ARITY, ["v1"]) is None                # malformed

    def test_vocabulary_matches_the_gp_stage(self):
        """The bridge offers exactly what the PySRRegressor is built with: 17 unaries + 6 binaries."""
        assert len(UNARY_OPERATORS) == 17 and len(BINARY_OPERATORS) == 6
        assert PYSR_OPERATORS == set(UNARY_OPERATORS) | {"+", "-", "*", "/", "pow", "rootn"}


class TestEvaluatePrefix:
    def test_columns_from_the_engine_realizations(self, engine):
        x, y, x_val, y_val = toy_problem()
        cols = evaluate_prefix(engine, ["+", "-", "*", "1.5", "x1", "*", "0.5", "x2", "2.0"], ["x1", "x2"], x, x_val, np.empty((0, 2)))
        assert cols[0].shape == (48, 1) and cols[1].shape == (12, 1) and cols[2].shape == (0, 1)
        np.testing.assert_allclose(cols[0].reshape(-1), y, rtol=1e-12)
        np.testing.assert_allclose(cols[1].reshape(-1), y_val, rtol=1e-12)


class TestFingerprint:
    def test_follows_the_arrays(self):
        a, b = toy_problem(0), toy_problem(1)
        assert data_fingerprint(a[0], a[1], a[2]) == data_fingerprint(*toy_problem(0)[:3])
        assert data_fingerprint(a[0], a[1], a[2]) != data_fingerprint(b[0], b[1], b[2])
        assert data_fingerprint(a[0], a[1], None) != data_fingerprint(a[0], a[1], a[2])
