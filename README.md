<h1 align="center" style="margin-top: 0px;">⚡ANSR:<br>Flash Amortized Neural Symbolic Regression</h1>

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

# Papers
- WIP


# Usage

```sh
pip install flash-ansr
```

```python
import torch
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

# Load the model
model = FlashANSR.load(
  directory=get_path('models', MODEL),
  generation_config=SoftmaxSamplingConfig(choices=32),  # or BeamSearchConfig / MCTSGenerationConfig
  n_restarts=8,
).to(device)

# Define data
X = ...
y = ...

# Fit the model to the data
model.fit(X, y, verbose=True)

# Show the best expression
print(model.get_expression())

# Predict with the best expression
y_pred = model.predict(X)
```

# Overview

### Training

<img src="https://raw.githubusercontent.com/psaegert/flash-ansr/refs/heads/main/assets/images/flash-ansr-training.png" width="300">

> **⚡ANSR Training on Fully Procedurally Generated Data** Inspired by NeSymReS ([Biggio et al. 2021](https://arxiv.org/abs/2106.06427))

### Architecture

<img src="https://raw.githubusercontent.com/psaegert/flash-ansr/refs/heads/main/assets/images/flash-ansr.png">

> **FlashANSR Architecture.** The model consists of an upgraded version of the SetTransformer ([Lee et al. 2019](https://arxiv.org/abs/1810.00825)) as an encoder, and a Pre-Norm Transformer decoder ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) as a generative model over symbolic expressions.

### Results
Coming soon
<!-- <img src="https://raw.githubusercontent.com/psaegert/flash-ansr/refs/heads/main/assets/images/test_time_compute_fastsrb.svg">

> **Test Time Compute scaling.** ⚡ANSR, NeSymReS ([Biggio et al. 2021](https://arxiv.org/abs/2106.06427)), PySR ([Cranmer 2023](https://arxiv.org/abs/2305.01582)), and E2E ([Kamienny et al. 2022](https://arxiv.org/abs/2204.10532)) are evaluated on the FastSRB benchmark with 10 datasets per equation, $n_{support}=512$, noise level 0.0.\
> AMD 9950X (16C32T), RTX 4090 (24GB). -->



# Citation
```bibtex
@software{flash-ansr2024,
    author = {Paul Saegert},
    title = {Flash Amortized Neural Symbolic Regression},
    year = 2024,
    publisher = {GitHub},
    version = {0.4.3},
    url = {https://github.com/psaegert/flash-ansr}
}
```
