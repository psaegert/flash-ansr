<h1 align="center" style="margin-top: 0px;">⚡ANSR:<br>Flash Amortized Neural Symbolic Regression</h1>

<div align="center">

[![pytest](https://github.com/psaegert/flash-ansr/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/pytest.yml)
[![quality checks](https://github.com/psaegert/flash-ansr/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/pre-commit.yml)
[![CodeQL Advanced](https://github.com/psaegert/flash-ansr/actions/workflows/codeql.yaml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/codeql.yaml)

</div>

<!-- TODO: Visual Abstract -->

<img src="./assets/images/nsr-training.drawio.svg" width="100%">

> **⚡ANSR Training on Fully Procedurally Generated Data** Inspired by NeSymReS ([Biggio et al. 2021](https://arxiv.org/abs/2106.06427))

# Introduction

### Abstract
Symbolic Regression has been approached with many different methods and paradigms. The overwhelming success of transformer-based language models in recent years has since motivated researchers to solve Symbolic Regression with large-scale pre-training of data-conditioned "equation generators" at competitive levels. However, as most traditional methods, the majority of these Amortized Neural Symbolic Regression methods rely on SymPy to simplify and compile randomly generated training equations, a choice that inevitably brings tradeoffs and requires workarounds to efficiently work at scale. I show that replacing SymPy with a novel token-based simplification algorithm with hand-crafted transformation rules enables training on _fully-procedurally_ generated and _higher-quality_ synthetic data, and thus develop ⚡ANSR. On various test sets, my method perfectly recovers $+80$% more equations numerically than the NeSymReS baseline while being 84 times faster natively, and yields comparable recovery rates to PySR in a quarter of its time. I provide an in-depth performance analysis of my method on stricter and more meaningful metrics than previous work. ⚡ANSR is open-source and available on GitHub and Huggingface, and allows for straight-forward replicability on consumer-grade hardware.

### Main Results
<img src="./assets/images/results.png" width="100%">

> **Model Comparison.** Up to 3 variables. Default Model Configurations (32 threads / beams).\
> Bootstrapped Median, 5p, 95p and AR-p ([Noreen 1989](https://scholar.google.com/scholar?hl=en&q=Computer-intensive+methods+for+testing+hypotheses)) values (n=1000).\
> N = 5000 ([⚡ v7.0](#usage)), 1000 ([PySR](https://github.com/MilesCranmer/PySR), [NeSymReS 100M](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales?tab=readme-ov-file#pretrained-models)).\
> AMD 9950X (16C32T), RTX 4090.

# Table of Contents
- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
  - [Hardware](#hardware)
  - [Software](#software)
- [Getting Started](#getting-started)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Install the package](#2-install-the-package)
- [Usage](#usage)
- [Training](#training)
  - [Express](#express)
  - [Manual](#manual)
    - [0. Prerequisites](#0-prerequisites)
    - [1. Import test data](#1-import-test-data)
    - [2. Generate validation data](#2-generate-validation-data)
    - [3. Train the model](#3-train-the-model)
    - [4. Evaluate the model](#4-evaluate-the-model)
      - [4.1 Evaluate NeSymReS](#41-evaluate-nesymres)
      - [4.2 Evaluate PySR](#42-evaluate-pysr)
- [Development](#development)
  - [Setup](#setup)
  - [Tests](#tests)
- [Citation](#citation)

# Requirements

## Hardware
- `32` GB Memory
- CUDA-enabled GPU
- `12` GB VRAM
- `64` GB Storage (subject to change)

## Software
- Python $\geq$ 3.11
- `pip` $\geq$ 21.3 with PEP 660 (see https://pip.pypa.io/en/stable/news/#v21-3)
- (Ubuntu 22.04.3 LTS)

# Getting Started
## 1. Clone the repository

```sh
git clone https://github.com/psaegert/flash-ansr
cd flash-ansr
```

## 2. Install the package

Create a virtual environment (optional):

**conda:**

```sh
conda create -n ansr python=3.11 ipykernel ipywidgets
conda activate ansr
```

Then, install the package via

```sh
pip install -e .
pip install -e ./nsrops
```

# Usage

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import flash_ansr
from flash_ansr import FlashANSR, GenerationConfig, install_model, get_path

# Specify the model
# Here: https://huggingface.co/psaegert/flash-ansr-v7.0
MODEL = "psaegert/flash-ansr-v7.0"

# Download the latest snapshot of the model
# By default, the model is downloaded to the directory `./models/` in the package root
install_model(MODEL)

# Load the model
ansr = FlashANSR.load(
    directory=get_path('models', MODEL),
    generation_config=GenerationConfig(method='beam_search', beam_width=32),  # optional
    n_restarts=32,  # optional
).to(device)

# Define data
X = ...
y = ...

# Fit the model to the data
ansr.fit(X, y, verbose=True)

# Show the best expression
print(ansr.get_expression())

# Predict with the best expression
y_pred = ansr.predict(X)
```


# Training

## Express

Use, copy or modify a config in `./configs`:

```
./configs
├── my_config
│   ├── dataset_train.yaml          # Link to skeleton pool and padding for training
│   ├── dataset_val.yaml            # Link to skeleton pool and padding for validation
│   ├── evaluation.yaml             # Evaluation settings
│   ├── expression_space.yaml       # Operators and variables
│   ├── nsr.yaml                    # Model settings and link to expression space
│   ├── skeleton_pool_train.yaml    # Sampling and holdout settings for training
│   ├── skeleton_pool_val.yaml      # Sampling and holdout settings for validation
│   └── train.yaml                  # Data and schedule for training
```

Run the training and evaluation pipeline with

```sh
./scripts/run.sh my_config
```

For more information see below.

## Manual

### 0. Prerequisites

Test data structured as follows:

```sh
./data/ansr-data/test_set
├── feynman
│   └── FeynmanEquations.csv
├── nguyen
│   └── nguyen.csv
└── soose_nc
    └── nc.csv
```

The test data can be cloned from the Hugging Face data repository:

```sh
git clone https://huggingface.co/psaegert/ansr-data data/ansr-data
```

### 1. Import test data

External datasets must be imported into the ANSR format:

```sh
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/soose_nc/nc.csv" -p "soose" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/soose_nc/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/feynman/FeynmanEquations.csv" -p "feynman" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/feynman/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/nguyen/nguyen.csv" -p "nguyen" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/nguyen/skeleton_pool" -v
```

with

- `-i` the input file

- `-p` the name of the parser implemented in `./src/flash_ansr/compat/convert_data.py`

- `-e` the expression space

- `-b` the config of a base skeleton pool to add the data to

- `-o` the output directory for the resulting skeleton pool

- `-v` verbose output

This will create and save a skeleton pool with the parsed imported skeletons in the specified directory:

```sh
./data/ansr-data/test_set/<test_set>
└── skeleton_pool
    ├── expression_space.yaml
    ├── skeleton_pool.yaml
    └── skeletons.pkl
```

### 2. Generate validation data

Validation data is generated by randomly sampling according to the settings in the skeleton pool config:

```sh
flash_ansr generate-skeleton-pool -c {{ROOT}}/configs/${CONFIG}/skeleton_pool_val.yaml -o {{ROOT}}/data/ansr-data/${CONFIG}/skeleton_pool_val -s 5000 -v
```

with

- `-c` the skeleton pool config
- `-o` the output directory to save the skeleton pool
- `-s` the number of unique skeletons to sample
- `-v` verbose output

### 3. Train the model

```sh
flash_ansr train -c {{ROOT}}/configs/${CONFIG}/train.yaml -o {{ROOT}}/models/ansr-models/${CONFIG} -v -ci 100000 -vi 10000
```

with

- `-c` the training config
- `-o` the output directory to save the model and checkpoints
- `-v` verbose output
- `-ci` the interval to save checkpoints
- `-vi` the interval for validation

### 4. Evaluate the model


```sh
flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/soose_nc/dataset.yaml" -n 5000 -o {{ROOT}}/results/evaluation/${CONFIG}/soose_nc.pickle -v
flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/feynman/dataset.yaml" -n 5000 -o {{ROOT}}/results/evaluation/${CONFIG}/feynman.pickle -v
flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/nguyen/dataset.yaml" -n 5000 -o {{ROOT}}/results/evaluation/${CONFIG}/nguyen.pickle -v
flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/configs/${CONFIG}/dataset_val.yaml" -n 5000 -o {{ROOT}}/results/evaluation/${CONFIG}/val.pickle -v
flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/pool_15/dataset.yaml" -n 5000 -o {{ROOT}}/results/evaluation/${CONFIG}/pool_15.pickle -v
flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/configs/${CONFIG}/dataset_train.yaml" -n 5000 -o {{ROOT}}/results/evaluation/${CONFIG}/train.pickle -v
```

with

- `-c` the evaluation config
- `-m` the model to evaluate
- `-d` the dataset to evaluate on
- `-n` the number of samples to evaluate
- `-o` the output file for results
- `-v` verbose output

#### 4.1 Evaluate NeSymReS
1. Clone [NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales) to a directory of your choice.
2. Download the `100M` model as described [here](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales?tab=readme-ov-file#pretrained-models)
3. Move the `100M` model into `flash-ansr/models/nesymres/`
4. Create a Python 3.10 (!) environment and install flash-ansr as in the previous steps.
5. Install NeSymReS in the same environment:
```sh
cd NeuralSymbolicRegressionThatScales
pip install -e src/
pip install lightning
```
1. Navigate back to this repository and run the evaluation
```sh
cd flash-ansr
./scripts/evaluate_nesymres <test_set>
```

#### 4.2 Evaluate PySR
1. Install [PySR](https://github.com/MilesCranmer/PySR) in the same environment as flash-ansr.
2. Run the evaluation
```sh
./scripts/evaluate_pysr <test_set>
```

# Development

## Setup
To set up the development environment, run the following commands:

```sh
pip install -e .[dev]
pip install -e ./nsrops
pre-commit install
```

## Tests

Test the package with `./scripts/pytest.sh`. Run pylint with `./scripts/pylint.sh`.

# Citation
```bibtex
@software{flash-ansr2024,
    author = {Paul Saegert},
    title = {Flash Amortized Neural Symbolic Regression},
    year = 2024,
    publisher = {GitHub},
    version = {0.3.0},
    url = {https://github.com/psaegert/flash-ansr}
}
```
