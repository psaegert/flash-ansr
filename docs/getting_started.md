# Getting Started

Requires Python >= 3.12.

```bash
pip install flash-ansr
```

This also pulls in `symbolic-data` and `simplipy` automatically, so no manual sequencing is needed. Check the installed version with `flash_ansr.__version__`.

## Download a checkpoint
```bash
flash_ansr install psaegert/flash-ansr-v25.0-T8-20M
```
The reference checkpoint for this release is [`psaegert/flash-ansr-v25.0-T8-20M`](https://huggingface.co/psaegert/flash-ansr-v25.0-T8-20M) (23.7M parameters, trained with `configs/v25.0-T8-20M`). The same series has a [3M](https://huggingface.co/psaegert/flash-ansr-v25.0-T8-3M) checkpoint for CPU-bound use and a [120M](https://huggingface.co/psaegert/flash-ansr-v25.0-T8-120M) checkpoint for a GPU; all three run every example on this page unchanged.
By default models are cached under `./models/` relative to the package root and can be uninstalled with `flash_ansr remove <repo>`.
Models can also be managed with the Python API via `flash_ansr.model.manage.install_model` and `flash_ansr.model.manage.remove_model`, and `flash_ansr.get_path('models', repo)` resolves the cached directory.

`FlashANSR.load` requires a checkpoint whose vocabulary carries the `<ieee754>` constant spans and the `<b00>`..`<bff>` byte tokens they are built from; anything else is refused at load, naming what is missing.

## Minimal inference Example
```python
import torch
from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path

device = "cuda" if torch.cuda.is_available() else "cpu"

# The estimator's POLICY, fixed at construction: the sampler, the refiner, the ranking, the compute.
model = FlashANSR.load(
  directory=get_path("models", "psaegert/flash-ansr-v25.0-T8-20M"),
  generation_config=SoftmaxSamplingConfig(draws=1024),   # the search budget: expressions drawn per problem
  refine={"n_restarts": 8},                              # RefineConfig: the constant optimizer (defaults shown)
  ranking="mdl",                                         # the two-part code: (n/2) log2 FVU + description length in bits (default)
  compute={"device": device, "workers": None},           # ComputeConfig: device and refiner workers (None = every core)
)

# Define data
X = ...
y = ...

# One call: draw candidates, fit their constants, rank them. Everything that changes with the
# PROBLEM is an argument here: the budget for this call, a seed, a complexity hint, the names.
result = model.fit(X, y, draws=None, seed=None, complexity=None, on_empty="return", verbose=True)

print(result.best.expression_infix)   # the answer
print(model.get_expression())         # the same, read back from the estimator
y_pred = model.predict(X)             # evaluate the answer on new data
```

`on_empty="return"` (the default) hands back an empty result with its ledger when no candidate fitted, so a caller can see why; `on_empty="raise"` raises `ConvergenceError` instead. A single candidate whose refinement fails is never an error: it is a `FIT_FAILED` row of the ledger.

## The result
`fit` returns a `FitResult` and keeps it as `model.result_`; `predict`, `get_expression` and `results` are views of it. The result is plain data (no model objects inside), so it pickles and travels:

```python
result = model.fit(X, y)

# Best refined candidate (or None if nothing fitted)
best = result.best
print(best.expression_infix)   # human-readable prediction
print(best.fvu, best.score, best.log_prob, best.constants)

# All refined survivors, score-sorted (best first)
for candidate in result.candidates:
    print(candidate.score, candidate.expression_infix)

# Evaluate and render any candidate by rank
y3 = result.predict(X, rank=3)
print(result.get_expression(rank=3, precision=3))

# The full candidate ledger: the generation pool joined with the refined survivors,
# each classified FIT_OK / FIT_FAILED / INVALID
ledger = result.ledger
print(len(ledger))                       # total candidates considered
print(ledger.fit_status, ledger.fvu)     # per-candidate columns

# Timing of the two phases
print(result.generation_time, result.refinement_time)

# A tabular view of the refined survivors (one row per candidate in result.candidates)
df = result.to_dataframe()

# Another ranking, no refit: a NEW result, this one untouched
weighted = result.rerank("weighted", weights={"n_nodes": 0.05})

# Persist and restore (the engine makes a loaded result evaluable again)
result.save("result.pkl")
from flash_ansr import FitResult
again = FitResult.load("result.pkl", engine=model.simplipy_engine)
```

A `Candidate` carries `expression` (the candidate as the model stated it, refined sites as `<constant>`), `slots` (the positions the refiner fitted), `expression_prefix`, `expression_infix`, `skeleton_prefix`, `constants` (refined), `constants_emitted` (as predicted by the model), `score`, `log_prob`, `fvu`, `n_nodes`, `mu` (simplipy complexity of the skeleton), `mdl` (description length of the refined expression, in milli-bits), `constant_count`, `pruned_variant`, `pareto_rank`, `rank`, and the provenance of a constant-ladder or typed-span variant (`spelling`, `typed_frozen`, `typed_thaw`). The `FIT_OK` / `FIT_FAILED` / `INVALID` codes live in `flash_ansr.inference`.

## The generation alone
`model.generate(X, y, draws=..., seed=...)` runs only the first phase and returns a `Generation`: the raw draws as token ids with their log-likelihoods, the encoder memory and the prompt they continued. It writes nothing to the estimator.

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
  directory=get_path("models", "psaegert/flash-ansr-v25.0-T8-20M"),   # catalog_train.yaml beside the checkpoint is the prior
  generation_config=PriorSamplingConfig(draws=1024),
  ranking="mdl",
)
result = prior.fit(X, y, seed=0)             # no GPU needed: the sampler and the refiner are CPU work
```

The draws are conditioned on one thing every regressor is told, the number of input columns
(`match_variables=True`: a draw is relabeled onto the columns, a wider draw is rejected); pass
`match_variables=False` for the raw prior over the catalog's whole variable set, `decontaminate=False`
for the bare prior without the benchmark holdout, and `catalog=` for a prior other than the
checkpoint's own. A prior candidate carries no log-probability, so the ranking must not weight it
(the default `mdl` ranking does not). The other control is the model's own unconditioned decode,
`SoftmaxSamplingConfig(draws=1024, guidance_weight=0.0)`: the learned null memory replaces the
encoder's, so the decoder proposes from its learned prior and the refiner still fits the data.

## The oracle: the ground truth as the only candidate

The ceiling of the fitting stage: hand the refiner the true expression and see how often fitting alone
recovers the law. `OracleConfig(expression=...)` takes the ground truth in prefix notation with its
literals spelled, over the columns as `x1..xN`; it becomes one candidate in the model's own emission
format (a fittable literal is a `<constant>` the refiner fits, a pow exponent or root index stays
spelled), and refinement and ranking run on it unchanged. Its budget is the refiner's restarts.

```python
from flash_ansr import FlashANSR, OracleConfig

oracle = FlashANSR.load(
  directory=get_path("models", "psaegert/flash-ansr-v25.0-T8-3M"),   # only the tokenizer and the engine are used
  generation_config=OracleConfig(expression=["+", "*", "2.5", "pow", "x1", "2", "sin", "x2"]),
  refine={"n_restarts": 8},
)
result = oracle.fit(X, y)                  # a harness sets a new OracleConfig for every problem
```
