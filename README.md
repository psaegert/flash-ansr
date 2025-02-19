<h1 align="center" style="margin-top: 0px;">🏗️Work In Progress🏗️</h1>

<h1 align="center" style="margin-top: 0px;">⚡ANSR:<br>Flash Amortized Neural Symbolic Regression</h1>

<div align="center">

[![pytest](https://github.com/psaegert/flash-ansr/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/pytest.yml)
[![quality checks](https://github.com/psaegert/flash-ansr/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/pre-commit.yml)
[![CodeQL Advanced](https://github.com/psaegert/flash-ansr/actions/workflows/codeql.yaml/badge.svg)](https://github.com/psaegert/flash-ansr/actions/workflows/codeql.yaml)

</div>

<!-- TODO: Visual Abstract -->

# Introduction

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

Optional: Create a virtual environment:

**conda:**

```sh
conda create -n ansr python=3.11 ipykernel ipywidgets
conda activate ansr
```

Then, install the package via

```sh
pip install -e .
pip install -e nsrops
```

# Usage

## Use a pre-trained model

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import flash_ansr
from flash_ansr import FlashANSR, install_model, get_path

# Specify the model
# Here: https://huggingface.co/psaegert/flash-ansr-v7.0
MODEL = "psaegert/flash-ansr-v7.0"

# Download the latest snapshot of the model
# By default, the model is downloaded to the directory `./models/` in the package root
install_model(MODEL)

# Load the model
ansr = FlashANSR.load(
    directory=get_path('models', MODEL),
    beam_width=256,
    n_restarts=32,
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

#### 4.1 Evaluate NeSymRes
1. Clone [NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales) to a directory of your choice.
2. Download the `100M` model as described [here](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales?tab=readme-ov-file#pretrained-models)
3. Move the `100M` model into `flash-ansr/models/nesymres/`
4. Create a Python 3.10 (!) environment and install flash-ansr as in the previous steps.
5. Install NeSymRes in the same environment:
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

Test the package with

```sh
./scripts/pytest.sh
```

for convenience.

# Citation
```bibtex
@software{flash-ansr2024,
    author = {Paul Saegert},
    title = {Flash Amortized Neural Symbolic Regression},
    year = 2024,
    publisher = {GitHub},
    version = {0.1.0},
    url = {https://github.com/psaegert/flash-ansr}
}
```
