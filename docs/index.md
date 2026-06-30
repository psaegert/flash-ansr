# Flash-ANSR

Flash Amortized Neural Symbolic Regression combines a SetTransformer encoder, Transformer decoder, and constant refiner to map tabular data to symbolic expressions. These docs are structured for first-time users and contributors.

- **New here?** Start with [Getting Started](getting_started.md).
- **How does it work?** Read [Concepts & Architecture](concepts.md).
- **Train or finetune?** See [Training](training.md).
- **Benchmarks/baselines?** Evaluation, baselines, and benchmarking moved to the standalone `srbf` package in v0.6 (`pip install srbf`); see https://github.com/psaegert/srbf.
- **API details?** Browse [API Reference](api.md).
- **Contributing?** Check [Contributing](contributing.md) and [FAQ](faq.md).

## Quick inference
```bash
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

To get every candidate in one call instead of reading back from the model, use `model.infer(X, y)`, which returns an `InferenceResult` (best `Candidate`, the score-sorted `candidates`, and the full `CandidateLedger`). See [Getting Started](getting_started.md).

## Serving these docs locally
```bash
pip install -r docs/requirements.txt
mkdocs serve
```
Visit http://127.0.0.1:8000.
