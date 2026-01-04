import numpy as np
import pytest

from flash_ansr import get_path
from flash_ansr.utils.config_io import load_config
from flash_ansr.expressions.support_sampling import SupportSampler, SupportSamplingError


@pytest.fixture
def quantized_sampler_config():
    return load_config(get_path('configs', 'test', 'skeleton_pool_quantized.yaml'))


@pytest.fixture
def quantized_sampler_config_no_transform():
    return load_config(get_path('configs', 'test', 'skeleton_pool_quantized_no_transform.yaml'))


def _build_sampler(config: dict[str, object]) -> SupportSampler:
    sampler_cfg = config['support_sampler']
    n_variables = len(config['variables'])
    independent = config['sample_strategy'].get('independent_dimensions', False)
    return SupportSampler(
        n_variables=n_variables,
        independent_dimensions=independent,
        config=sampler_cfg,
    )


def test_quantized_sampler_even_bins_mixes_axes(quantized_sampler_config):
    np.random.seed(0)
    sampler = _build_sampler(quantized_sampler_config)

    configured_limit = sampler.configured_max_n_support
    assert configured_limit == 16

    assert sampler.scale_transform is None
    assert sampler.quantize_transform is not None
    support = sampler.sample(n_support=configured_limit)

    assert support.shape == (configured_limit, sampler.n_variables)
    assert np.unique(support, axis=0).shape[0] == configured_limit

    per_dim_unique_counts = [np.unique(support[:, dim]).shape[0] for dim in range(support.shape[1])]
    # At least one dimension should remain largely continuous.
    assert any(count >= configured_limit // 2 for count in per_dim_unique_counts)

    if sampler.quantize_transform is not None:
        # Quantized dimensions are capped by the sampled number of bins.
        assert any(2 <= count <= sampler.quantize_transform.max_bins for count in per_dim_unique_counts)

    assert np.all((support >= -10) & (support <= 10))


def test_quantized_sampler_uniform_bins_fractional_values(quantized_sampler_config_no_transform):
    np.random.seed(2)
    sampler = _build_sampler(quantized_sampler_config_no_transform)

    configured_limit = sampler.configured_max_n_support
    assert configured_limit == 20

    assert sampler.scale_transform is None
    assert sampler.quantize_transform is not None
    support = sampler.sample(n_support=configured_limit)

    assert support.shape == (configured_limit, sampler.n_variables)
    assert np.unique(support, axis=0).shape[0] == configured_limit

    per_dim_unique_counts = [np.unique(support[:, dim]).shape[0] for dim in range(support.shape[1])]
    quantized_dim = int(np.argmin(per_dim_unique_counts))
    fractional_parts = np.abs(support[:, quantized_dim] - np.round(support[:, quantized_dim]))
    assert np.any(fractional_parts > 1e-4)

    if sampler.quantize_transform is not None:
        assert per_dim_unique_counts[quantized_dim] <= sampler.quantize_transform.max_bins
        assert any(count > sampler.quantize_transform.max_bins for count in per_dim_unique_counts)


def test_unique_guard_detects_duplicates():
    np.random.seed(3)
    sampler = SupportSampler(
        n_variables=1,
        independent_dimensions=False,
        config={
            'support_prior': {
                'name': 'constant',
                'kwargs': {'value': 0},
            },
            'n_support_prior': {
                'name': 'constant',
                'kwargs': {'value': 4},
            },
            'require_unique': True,
            'transforms': [
                {
                    'type': 'scale',
                    'prior': {
                        'name': 'constant',
                        'kwargs': {'value': 0},
                    },
                }
            ],
        },
    )

    with pytest.raises(SupportSamplingError):
        sampler.sample(n_support=4)
