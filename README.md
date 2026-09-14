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

Flash-ANSR is a library for **amortized neural symbolic regression**: load a pretrained model, call `fit(X, y)`, and recover a symbolic expression for your tabular data, or train your own model. It is built for fast, ready-to-use inference.

# Publications
- Saegert & Köthe 2026, _Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression_ (ICML 2026) [https://arxiv.org/abs/2602.08885](https://arxiv.org/abs/2602.08885)


# Usage

Requires Python >= 3.12.

```sh
pip install flash-ansr
flash_ansr install psaegert/flash-ansr-v25.0-T8-20M   # the reference checkpoint (see "Models")
```

```python
import numpy as np
import torch
from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path

device = "cuda" if torch.cuda.is_available() else "cpu"

# The estimator's policy is fixed at construction: the sampler, the refiner, the ranking, the compute.
model = FlashANSR.load(
  directory=get_path("models", "psaegert/flash-ansr-v25.0-T8-20M"),
  generation_config=SoftmaxSamplingConfig(draws=1024),  # the search budget: expressions drawn per problem
  ranking="mdl",                                        # log10(FVU) + 1e-2 per bit of description length (default)
  compute={"device": device},
)

# Define data: a small synthetic example, y = 2 * x + sin(3 * x)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2 * X[:, 0] + np.sin(3 * X[:, 0])

# One call: draw candidates, fit their constants, rank them
result = model.fit(X, y)

print(result.best.expression_infix)   # the answer
print(model.get_expression())         # the same, read back from the estimator
y_pred = model.predict(X)             # evaluate the answer on new data
```

**The result.** `fit` returns a `FitResult` and keeps it as `model.result_`: the score-sorted refined `candidates` (each a `Candidate` with its expression, constants, `fvu`, `score`, `mdl`, `log_prob`, ...), the full `ledger` (every draw, classified `FIT_OK` / `FIT_FAILED` / `INVALID`), the ranking that ordered them and the generation / refinement times. Everything else is a view of it:

```python
result.predict(X, rank=3)                                   # evaluate the candidate at rank 3
result.get_expression(rank=3, precision=3)                  # render it, constants rounded for display
result.to_dataframe()                                       # one row per refined candidate
result.rerank("weighted", weights={"n_nodes": 0.05})        # a NEW result under another ranking, no refit
result.save("result.pkl")                                   # plain data: no model objects inside
FitResult.load("result.pkl", engine=model.simplipy_engine)  # ... and back, evaluable again
```

**The call.** Everything that changes with the problem is an argument of `fit`: `draws=` overrides the budget for this call, `seed=` makes the draw and the refinement reproducible, `complexity=` hints the target complexity, `on_empty="raise"` raises `ConvergenceError` instead of returning an empty result when nothing fitted, `variable_names=` names the columns. Everything else is policy and lives on the estimator.

Explore more in the [Demo Notebook](https://github.com/psaegert/flash-ansr/blob/main/demo.ipynb).

**Train your own:** see the [training guide](https://flash-ansr.readthedocs.io/en/latest/training/).

# Models

The v25.0-T8 series: one recipe and one data prior at three sizes. Pick by the hardware you have; every one of them runs the examples above unchanged.

| Checkpoint | Parameters | Training | Notes |
|---|---|---|---|
| [`psaegert/flash-ansr-v25.0-T8-3M`](https://huggingface.co/psaegert/flash-ansr-v25.0-T8-3M) | 3.5M | 1.5M steps, batch 128, `configs/v25.0-T8-3M` | the smallest; comfortable on a CPU |
| [`psaegert/flash-ansr-v25.0-T8-20M`](https://huggingface.co/psaegert/flash-ansr-v25.0-T8-20M) | 23.7M | 1.5M steps, batch 128, `configs/v25.0-T8-20M` | the reference checkpoint for this release |
| [`psaegert/flash-ansr-v25.0-T8-120M`](https://huggingface.co/psaegert/flash-ansr-v25.0-T8-120M) | 123.6M | 1.5M steps, batch 128, `configs/v25.0-T8-120M` | the largest; a GPU is advisable |

```sh
flash_ansr install psaegert/flash-ansr-v25.0-T8-20M
```

Every catalog that [srbf](https://github.com/psaegert/srbf) evaluates on is held out of the training data by canonical form (6,660 expressions across 29 catalogs).

# Inference speed

Several inference-speed features are **enabled by default** and designed to be quality-neutral, so the quickstart above already runs in the fast regime. The speed-relevant settings are the compute group of the generation config:

| Setting | Default | What it does |
|---|---|---|
| `use_cache` | `True` | KV-cache decoding |
| `batch_size` | `'auto'` | budget-adaptive batching (pass an `int` to override) |
| `static_decode` | `None` | static decoding, auto-enabled for capable models (set `True`/`False` to force) |

```python
from flash_ansr import SoftmaxSamplingConfig

config = SoftmaxSamplingConfig(
  draws=1024,          # number of candidate expressions to draw per problem (fit(draws=) overrides it)
  use_cache=True,      # KV cache (default)
  batch_size='auto',   # budget-adaptive chunking (default)
  static_decode=None,  # auto for capable models (default)
)
```

Constant refinement runs in parallel; control it via `compute={"workers": N, "persistent_pool": True}` on `FlashANSR.load`. By default (`workers=None`) the pool uses every available CPU core, which oversubscribes shared machines; pass an explicit integer to cap it (`0` disables multiprocessing).

To opt out of these defaults:

```python
SoftmaxSamplingConfig(draws=1024, use_cache=False, batch_size=128, static_decode=False)
```

> **Candidate ranking.** Three modes, one sort: `ranking="mdl"` (default; `log10(FVU)` plus `mdl_strength` decades per bit of the refined expression's description length), `{"mode": "weighted", "weights": {...}}` (weights over `n_nodes`, `n_constants`, `n_constant_placeholders`, `n_typed_literals`, `mdl`, `neg_log_prob`) and `{"mode": "pareto", "metrics": [...], "tie_break": ...}` (the non-dominated front over the metrics). Each knob belongs to one mode and raises under another. A fitted result can be re-ordered under any ranking without refitting: `result.rerank(...)`.

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


# Related projects

- [**SimpliPy**](https://github.com/psaegert/simplipy): the expression simplification engine integrated into the Flash-ANSR training loop.
- [**symbolic-data**](https://github.com/psaegert/symbolic-data): the model-agnostic symbolic-regression data layer (catalogs, `ProblemSource`, holdouts) that feeds Flash-ANSR training. It is an unconditional runtime dependency and the backbone of the training loop.
- [**srbf**](https://github.com/psaegert/srbf): the companion symbolic-regression evaluation and benchmarking framework (engine, model adapters, benchmarks, metrics), developed alongside Flash-ANSR.

# Citation
```bibtex
@inproceedings{saegert2026breakingsimplificationbottleneckamortized,
  title   = {Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression},
  author  = {Paul Saegert and Ullrich Köthe},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
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
  version = {0.14.0},
  url     = {https://github.com/psaegert/flash-ansr}
}
```
