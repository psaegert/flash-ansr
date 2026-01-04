import numpy as np


def test_sample_generates_outputs(tmp_path):
    yaml_content = """
example:
  prepared: v1 + 2
  vars:
    v1:
      sample_type: [uni, pos]
      sample_range: [0.1, 1.0]
"""
    yaml_path = tmp_path / "fastsrb.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    from flash_ansr.benchmarks import FastSRBBenchmark

    benchmark = FastSRBBenchmark(yaml_path, random_state=0)

    sample = benchmark.sample("example", n_points=4, random_state=0)

    assert sample["data"]["X"].shape == (4, 1)
    assert sample["data"]["y"].shape == (4,)
    np.testing.assert_allclose(sample["data"]["y"], sample["data"]["X"][:, 0] + 2.0, rtol=1e-6)
