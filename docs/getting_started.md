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

See [all available models on Hugging Face](https://huggingface.co/search/full-text?q=psaegert%2Fflash-ansr-v23.0):

## Minimal inference Example
```python
import numpy as np
from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path

# Define some data
X = np.random.randn(256, 2)
y = X[:, 0] + X[:, 1]

# Load the model (assuming v23.0-120M is installed)
model = FlashANSR.load(
    directory=get_path('models', 'psaegert/flash-ansr-v23.0-120M'),
    generation_config=SoftmaxSamplingConfig(choices=256),
)  # .to(device) for GPU. Highly recommended.

# Find an expression that fits the data by sampling from the model
model.fit(X, y, verbose=True)

print("Expression:", model.get_expression())

y_pred = model.predict(X)
print("Predictions:", y_pred[:5])

# All results are stored in model.results as a pandas DataFrame
model.results
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
