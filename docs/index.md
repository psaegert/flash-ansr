# Flash-ANSR

Flash Amortized Neural Symbolic Regression combines a SetTransformer encoder, Transformer decoder, and constant refiner to map tabular data to symbolic expressions. These docs are structured for first-time users and contributors.

- **New here?** Start with [Getting Started](getting_started.md).
- **How does it work?** Read [Concepts & Architecture](concepts.md).
- **Train or finetune?** See [Training](training.md).
- **Benchmarks/baselines?** Evaluation, baselines, and benchmarking live in the standalone `srbf` package (`pip install srbf`); see https://github.com/psaegert/srbf.
- **API details?** Browse [API Reference](api.md).
- **Contributing?** Check [Contributing](contributing.md) and [FAQ](faq.md).

## Quick inference
Requires Python >= 3.12.

```bash
pip install flash-ansr
flash_ansr install psaegert/flash-ansr-v25.0-T8-20M
```
```python
import torch
from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path

device = "cuda" if torch.cuda.is_available() else "cpu"

# The estimator's policy: the sampler (with its draw budget), the ranking, the compute
model = FlashANSR.load(
  directory=get_path("models", "psaegert/flash-ansr-v25.0-T8-20M"),
  generation_config=SoftmaxSamplingConfig(draws=1024),
  ranking="mdl",
  compute={"device": device},
)

# Define data
X = ...
y = ...

# One call: draw candidates, fit their constants, rank them
result = model.fit(X, y, verbose=True)

# The answer, and its evaluation on new data
print(result.best.expression_infix)
y_pred = model.predict(X)
```

`fit` returns a `FitResult` (the score-sorted refined `candidates`, the full classified `ledger`, the ranking, the timings) and keeps it as `model.result_`; `predict`, `get_expression` and `results` read it. See [Getting Started](getting_started.md).

## Serving these docs locally
```bash
pip install -r docs/requirements.txt
mkdocs serve
```
Visit http://127.0.0.1:8000.
