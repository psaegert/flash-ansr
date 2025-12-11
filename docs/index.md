# Flash-ANSR

Flash Amortized Neural Symbolic Regression combines a SetTransformer encoder, Transformer decoder, and constant refiner to map tabular data to symbolic expressions. These docs are structured for first-time users and contributors.

- **New here?** Start with [Getting Started](getting_started.md).
- **How does it work?** Read [Concepts & Architecture](concepts.md).
- **Train or finetune?** See [Training](training.md).
- **Benchmarks/baselines?** See [Evaluation](evaluation.md).
- **API details?** Browse [API Reference](api.md).
- **Contributing?** Check [Contributing](contributing.md) and [FAQ](faq.md).

## Quick inference
```bash
pip install flash-ansr
flash_ansr install psaegert/flash-ansr-v23.0-120M
```
```python
import numpy as np
from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path

X = np.random.randn(256, 2)
model = FlashANSR.load(
   directory=get_path('models', 'psaegert/flash-ansr-v23.0-120M'),
   generation_config=SoftmaxSamplingConfig(choices=512),
)
expr = model.fit(X, X[:, 0] + X[:, 1])
print(expr)
```

## Serving these docs locally
```bash
pip install mkdocs mkdocs-material mkdocs-autorefs
mkdocs serve
```
Visit http://127.0.0.1:8000.
