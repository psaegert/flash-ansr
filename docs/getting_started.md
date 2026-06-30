# Getting Started

```bash
pip install flash-ansr
```

Check the installed version with `flash_ansr.__version__`.

## Download a checkpoint
```bash
flash_ansr install psaegert/flash-ansr-v23.0-120M
```
By default models are cached under `./models/` relative to the package root and can be uninstalled with `flash_ansr remove <repo>`.
Models can also be managed with the Python API via `flash_ansr.model.manage.install_model` and `flash_ansr.model.manage.remove_model`.

See [all available models on Hugging Face](https://huggingface.co/models?search=flash-ansr-v23.0):

## Minimal inference Example
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

# Load the model (KV-cache, auto-batching and static decoding are on by default in v0.5)
model = FlashANSR.load(
  directory=get_path('models', MODEL),
  generation_config=SoftmaxSamplingConfig(choices=1024),  # or BeamSearchConfig / MCTSGenerationConfig
  length_penalty=0.05,  # prefer shorter expressions when scoring candidates (renamed from `parsimony` in v0.5)
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

## Getting all candidates with `infer`
`fit` / `get_expression` / `predict` keep the fitted state on the model for read-back. To get every candidate in one call instead, use `infer`, which returns an `InferenceResult` and writes nothing to the model:

```python
result = model.infer(X, y)

# Best refined candidate (or None if nothing fitted)
best = result.best
print(best.expression_infix)   # human-readable prediction
print(best.fvu, best.score, best.log_prob, best.constants)

# All refined survivors, score-sorted (best first)
for candidate in result.candidates:
    print(candidate.score, candidate.expression_infix)

# The full candidate ledger: the generation pool joined with the refined
# survivors, each classified FIT_OK / FIT_FAILED / INVALID
ledger = result.ledger
print(len(ledger))                       # total candidates considered
print(ledger.fit_status, ledger.fvu)     # per-candidate columns

# Timing of the two phases
print(result.generation_time, result.refinement_time)
```

A `Candidate` carries `expression` (skeleton tokens), `expression_prefix`, `expression_infix`, `skeleton_prefix`, `constants`, `score`, `log_prob`, `fvu`, `complexity`, and optional `y_pred` / `y_pred_val` (populated for the top `top_k` candidates). The `FIT_OK` / `FIT_FAILED` / `INVALID` codes live in `flash_ansr.inference`.

Find more details in the [API Reference](api.md).


## Evaluation
As of v0.6, evaluation, baseline comparisons, and benchmarking moved out of flash-ansr into the standalone `srbf` (Symbolic Regression Benchmark Framework) package.

```bash
pip install srbf
```

See the [srbf repository](https://github.com/psaegert/srbf) for usage.

## Next steps
- See [Concepts & Architecture](concepts.md) for how the pieces fit together.
- For training your own checkpoints, jump to [Training](training.md).
- For baseline comparisons and sweeps, see the [srbf repository](https://github.com/psaegert/srbf) (evaluation moved to the standalone `srbf` package in v0.6).
