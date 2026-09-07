# Getting Started

Requires Python >= 3.12.

```bash
pip install flash-ansr
```

This also pulls in `symbolic-data` and `simplipy` automatically, so no manual sequencing is needed. Check the installed version with `flash_ansr.__version__`.

## Download a checkpoint
```bash
flash_ansr install psaegert/flash-ansr-v25.0-T7-3M
```
The reference checkpoint for this release is [`psaegert/flash-ansr-v25.0-T7-3M`](https://huggingface.co/psaegert/flash-ansr-v25.0-T7-3M) (3.5M parameters, trained with `configs/v25.0-T7`).
By default models are cached under `./models/` relative to the package root and can be uninstalled with `flash_ansr remove <repo>`.
Models can also be managed with the Python API via `flash_ansr.model.manage.install_model` and `flash_ansr.model.manage.remove_model`, and `flash_ansr.get_path('models', repo)` resolves the cached directory.

`FlashANSR.load` requires a checkpoint whose vocabulary carries the `<ieee754>` constant spans and the `<b00>`..`<bff>` byte tokens they are built from; anything else is refused at load, naming what is missing.

## Minimal inference Example
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import flash_ansr
from flash_ansr import (
  FlashANSR,
  SoftmaxSamplingConfig,
)

# The installed checkpoint directory
from flash_ansr import get_path
CHECKPOINT = get_path("models", "psaegert/flash-ansr-v25.0-T7-3M")

# Load the model (KV-cache, auto-batching and static decoding are on by default)
model = FlashANSR.load(
  directory=CHECKPOINT,
  generation_config=SoftmaxSamplingConfig(choices=1024),
  # Candidate ranking (default): log10(FVU) + 1e-2 per bit of the refined expression's description
  # length. Alternatives: ranking_mode="weighted" with ranking_weights={"n_nodes": 0.05}, or
  # ranking_mode="pareto" with ranking_metrics=("fvu", "n_nodes").
  ranking_mode="mdl",
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

# A tabular view of the refined survivors (one row per candidate in result.candidates)
df = result.to_dataframe()
```

A `Candidate` carries `expression` (skeleton tokens), `expression_prefix`, `expression_infix`, `skeleton_prefix`, `constants` (refined), `constants_emitted` (as predicted by the model), `score`, `log_prob`, `fvu`, `n_nodes`, `mu` (simplipy complexity of the skeleton), `mdl` (description length of the refined expression, in milli-bits), `constant_count`, `pruned_variant`, `pareto_rank`, `rank`, and optional `y_pred` / `y_pred_val` (populated for the top `top_k` candidates). The `FIT_OK` / `FIT_FAILED` / `INVALID` codes live in `flash_ansr.inference`.

`result.to_dataframe()` returns a pandas DataFrame of the refined survivors (one row per candidate in `result.candidates`, i.e. `FIT_OK` fits), not the full ledger. To control which candidates get predictions, `infer` takes `top_k` (compute `y_pred` / `y_pred_val` for the top `top_k` candidates; `None` = the best only), `predict_val` (toggle validation-set prediction), and `X_val` (out-of-sample features for `y_pred_val`).

Find more details in the [API Reference](api.md).


## Evaluation
Evaluation, baseline comparisons, and benchmarking live in the standalone `srbf` (Symbolic Regression Benchmark Framework) package.

```bash
pip install srbf
```

See the [srbf repository](https://github.com/psaegert/srbf) for usage.

## Next steps
- See [Concepts & Architecture](concepts.md) for how the pieces fit together.
- For training your own checkpoints, jump to [Training](training.md).
- For baseline comparisons and sweeps, see the [srbf repository](https://github.com/psaegert/srbf).

## The prior baseline: candidates from the training prior

How much is the trained decoder worth? Swap it for the prior it was trained on and keep everything
else: the same constant refinement, the same MDL ranking, the same candidate ledger.

```python
from flash_ansr import FlashANSR, PriorSamplingConfig

prior = FlashANSR.load(
  directory=CHECKPOINT,                      # catalog_train.yaml beside the checkpoint is the prior
  generation_config=PriorSamplingConfig(choices=1024),
  ranking_mode="mdl",
)
result = prior.infer(X, y)                   # no GPU needed: the sampler and the refiner are CPU work
```

The draws are conditioned on one thing every regressor is told, the number of input columns
(`match_variables=True`: a draw is relabeled onto the columns, a wider draw is rejected); pass
`match_variables=False` for the raw prior over the catalog's whole variable set, `decontaminate=False`
for the bare prior without the benchmark holdout, and `catalog=` for a prior other than the
checkpoint's own. A prior candidate carries no log-probability, so the ranking must not weight it
(the default `mdl` ranking does not). The unconditioned decode, `infer(..., conditioned=False)`, is
the other control: the model's own learned prior through the same pipeline.

