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
flash_ansr install psaegert/flash-ansr-v25.0-T7-3M
```
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

To get every candidate in one call instead of reading back from the model, use `model.infer(X, y)`, which returns an `InferenceResult` (best `Candidate`, the score-sorted `candidates`, and the full `CandidateLedger`). See [Getting Started](getting_started.md).

## Serving these docs locally
```bash
pip install -r docs/requirements.txt
mkdocs serve
```
Visit http://127.0.0.1:8000.
