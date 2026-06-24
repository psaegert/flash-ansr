<h1 align="center" style="margin-top: 0px;">⚡Flash-ANSR:<br>Fast Amortized Neural Symbolic Regression</h1>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/flash-ansr.svg)](https://pypi.org/project/flash-ansr/)
[![PyPI license](https://img.shields.io/pypi/l/flash-ansr.svg)](https://pypi.org/project/flash-ansr/)
[![Documentation Status](https://readthedocs.org/projects/flash-ansr/badge/?version=latest)](https://flash-ansr.readthedocs.io/en/latest/?badge=latest)

</div>

<div align="center">

[![pytest](https://github.com/psaegert/flash-ansr/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/pytest.yml)
[![quality checks](https://github.com/psaegert/flash-ansr/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/pre-commit.yml)
[![CodeQL Advanced](https://github.com/psaegert/flash-ansr/actions/workflows/codeql.yaml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/codeql.yaml)

</div>

# Publications
- Saegert & Köthe 2026, _Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression_ (preprint, under review) [https://arxiv.org/abs/2602.08885](https://arxiv.org/abs/2602.08885)


# Usage

```sh
pip install flash-ansr
```

```python
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import flash_ansr
from flash_ansr import (
  FlashANSR,
  SoftmaxSamplingConfig,
  install_model,
  get_path,
)

# Select a model from Hugging Face
# https://huggingface.co/models?search=flash-ansr-v23.0
MODEL = "psaegert/flash-ansr-v23.0-120M"

# Download the latest snapshot of the model
# By default, the model is downloaded to the directory `./models/` in the package root
install_model(MODEL)

# Load the model (KV-cache, auto-batching and static decoding are on by default; see "Inference speed")
model = FlashANSR.load(
  directory=get_path('models', MODEL),
  generation_config=SoftmaxSamplingConfig(choices=1024),  # or BeamSearchConfig / MCTSGenerationConfig
  length_penalty=0.05,  # prefer shorter expressions when scoring candidates (renamed from `parsimony` in v0.5)
).to(device)

# Define data: a small synthetic example, y = 2.5 * sin(x) + x^2 / 3
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2.5 * np.sin(X[:, 0]) + X[:, 0] ** 2 / 3

# Fit the model to the data
model.fit(X, y, verbose=True)

# Show the best expression
print(model.get_expression())

# Predict with the best expression
y_pred = model.predict(X)
```

Explore more in the [Demo Notebook](https://github.com/psaegert/flash-ansr/blob/main/experimental/demo.ipynb).

# Inference speed

Flash-ANSR v0.5 ships several inference-speed improvements, **enabled by default** and designed to be quality-neutral, so the quickstart above already runs in the fast regime. The speed-relevant settings live on the generation config:

| Setting | Default | What it does |
|---|---|---|
| `use_cache` | `True` | KV-cache decoding |
| `batch_size` | `'auto'` | candidate-budget-adaptive batching (pass an `int` to override) |
| `static_decode` | `None` | static decoding, auto-enabled for capable models (set `True`/`False` to force) |

```python
from flash_ansr import SoftmaxSamplingConfig

config = SoftmaxSamplingConfig(
  choices=1024,        # number of candidate expressions to sample
  use_cache=True,      # KV cache (default)
  batch_size='auto',   # candidate-budget-adaptive chunking (default)
  static_decode=None,  # auto for capable models (default)
)
```

Constant refinement runs in parallel; control it via `FlashANSR.load(..., refiner_workers=N, persistent_refine_pool=True)`.

To reproduce v0.4.x inference behavior, opt out of the new defaults:

```python
SoftmaxSamplingConfig(choices=1024, use_cache=False, batch_size=128, static_decode=False)
```

> **Breaking change (v0.5):** the candidate-selection penalty `parsimony` was renamed to `length_penalty`. Replace any `parsimony=` arguments with `length_penalty=`.

# Overview

<table>
  <tr>
    <td align="center">
      <h3>SRSD/FastSRB Results</h3>
      <img src="https://raw.githubusercontent.com/psaegert/flash-ansr/refs/heads/main/assets/images/small_test_time_compute_fastsrb.svg" width="500">
      <p>Results on the SRSD/FastSRB benchmark <a href="https://arxiv.org/abs/2206.10540">[Matsubara et al. 2022]</a>, <a href="https://arxiv.org/abs/2508.14481">[Martinek 2025]</a> <strong>Left:</strong> Validation Numeric Recovery Rate (vNRR) as a function of inference time (log scale). FLASH-ANSR models (shades of blue) scale monotonically with compute, with the 120M model partially surpassing the PySR baseline (red). Baselines NeSymReS <a href="https://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf">[Biggio et al. 2021]</a> and E2E <a href="https://arxiv.org/abs/2204.10532">[Kamienny et al. 2022]</a> fail to generalize to the benchmark. <strong>Right:</strong> Expression Length Ratio (predicted vs ground truth) versus compute. We observe a parsimony inversion: while PySR <a href="https://arxiv.org/abs/2305.01582">[Cranmer 2023]</a> increases complexity to minimize error over time, FLASH-ANSR converges toward simpler, more canonical expressions as the sampling budget increases. Shaded regions denote 95% confidence intervals.</p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <h3>Training</h3>
      <img src="https://raw.githubusercontent.com/psaegert/flash-ansr/refs/heads/main/assets/images/flash-ansr-training.png" width="420">
      <p><strong>The Flash-ANSR training pipeline.</strong> Following the
established standard encoder-decoder paradigm, our framework
integrates <a href="https://github.com/psaegert/simplipy">SimpliPy</a> (top center) into the loop for synchronous
simplification of on-the-fly generated training expressions.</p>
    </td>
    <td align="center">
      <h3>Architecture</h3>
      <img src="https://raw.githubusercontent.com/psaegert/flash-ansr/refs/heads/main/assets/images/flash-ansr.svg" width="420">
      <p><strong>Flash-ANSR model architecture.</strong> The Set Transformer <a href="https://arxiv.org/abs/1810.00825">[Lee et al. 2019]</a> encoder ingests a variable-sized set of input-output pairs and produces a fixed-size latent representation via Induced Set Attention Blocks (ISAB) and Set Attention Blocks (SAB). The Transformer decoder <a href="https://arxiv.org/abs/1706.03762">[Vaswani et al. 2017]</a>, <a href="https://arxiv.org/abs/2002.04745">[Xiong et al. 2020]</a> autoregressively generates a symbolic expression token-by-token, attending to the encoded dataset at each step.</p>
    </td>
  </tr>
</table>


# Citation
```bibtex
@misc{saegert2026breakingsimplificationbottleneckamortized,
  title   = {Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression},
  author  = {Paul Saegert and Ullrich Köthe},
  year    = {2026},
  eprint  = {2602.08885},
  archivePrefix =  {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2602.08885},
}

% Optionally
@mastersthesis{flash-ansr2024-thesis,
  author  = {Paul Saegert},
  title   = {Flash Amortized Neural Symbolic Regression},
  school  = {Heidelberg University},
  year    = {2025},
  url     = {https://github.com/psaegert/flash-ansr-thesis}
}
@software{flash-ansr2024,
  author  = {Paul Saegert},
  title   = {Flash Amortized Neural Symbolic Regression},
  year    = {2024},
  publisher   = {GitHub},
  version = {0.5.0},
  url     = {https://github.com/psaegert/flash-ansr}
}
```
