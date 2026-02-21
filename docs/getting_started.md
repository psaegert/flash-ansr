# Getting Started

```bash
pip install flash-ansr
```

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

# Load the model
model = FlashANSR.load(
  directory=get_path('models', MODEL),
  generation_config=SoftmaxSamplingConfig(choices=1024),  # or BeamSearchConfig / MCTSGenerationConfig
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

Find more details in the [API Reference](api.md).


## One-command evaluation
```bash
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M_fastsrb.yaml -v
```
Produces a pickle under `results/evaluation/...` with entries like:

- `predicted_expression`
- `predicted_log_prob`
- `y_pred`
- `...`

For more details, see [Evaluation](evaluation.md).

## Next steps
- See [Concepts & Architecture](concepts.md) for how the pieces fit together.
- For training your own checkpoints, jump to [Training](training.md).
- For baseline comparisons and sweeps, read [Evaluation](evaluation.md).
