"""In-process refinement must not disturb the caller's global RNG state.

``_refine_candidate_worker`` seeds ``np.random``/``torch`` per candidate so a fit is
reproducible from its expression hash. In the pool that is contained -- separate processes --
but the same worker also runs in-process (serial refine, and the pool-broken fallback), where
the seeding lands on the caller's global RNGs.
"""
import numpy as np
import torch
from simplipy import SimpliPyEngine

from flash_ansr.flash_ansr import _refine_candidate_worker


def _payload(seed):
    """A minimal refine job: fit `C * x1` against a line."""
    X = np.linspace(1.0, 5.0, 32).reshape(-1, 1)
    y = (2.5 * X[:, 0]).reshape(-1, 1)
    return {
        'simplipy_engine': SimpliPyEngine.load('acj-4-3', install=True),
        'expression': ['*', '<constant>', 'x1'],
        'n_variables': 1,
        'X': X,
        'y': y,
        'n_restarts': 2,
        'method': 'curve_fit_lm',
        'p0_noise': 'normal',
        'p0_noise_kwargs': {'loc': 0, 'scale': 1},
        'converge_error': 'ignore',
        'numpy_errors': 'ignore',
        'y_variance': float(np.var(y)),
        'length_penalty': 0.0,
        'constants_penalty': 0.0,
        'likelihood_penalty': 0.0,
        'log_prob': None,
        'constant_count': 0,
        'complexity': None,
        'metadata_snapshot': None,
        'raw_beam': [],
        'beam': ['*', '<constant>', 'x1'],
        'raw_beam_decoded': '',
        'seed': seed,
    }


def test_caller_numpy_rng_is_unchanged() -> None:
    np.random.seed(1234)
    expected = np.random.rand(4)

    np.random.seed(1234)
    _refine_candidate_worker(_payload(987654321))
    actual = np.random.rand(4)

    np.testing.assert_array_equal(expected, actual)


def test_caller_torch_rng_is_unchanged() -> None:
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    _refine_candidate_worker(_payload(987654321))
    actual = torch.randn(4)

    assert torch.equal(expected, actual)


def test_per_candidate_fit_is_still_reproducible() -> None:
    """The point of the seed: the same seed must still produce the same fit."""
    a, _ = _refine_candidate_worker(_payload(2024))
    b, _ = _refine_candidate_worker(_payload(2024))
    assert a is not None and b is not None
    np.testing.assert_array_equal(a['fits'][0][0], b['fits'][0][0])
