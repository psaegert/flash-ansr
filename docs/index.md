# Flash-ANSR

Welcome to the Flash Amortized Neural Symbolic Regression docs. Use the links below to jump to what you need quickly.

- **API reference**: `API-Reference` for the FlashANSR class, configs, and helpers.
- **Training**: how to train/finetune with repo configs and scripts.
- **Evaluation**: reproducible runs, benchmarks, and baselines (FlashANSR, PySR, NeSymReS, SkeletonPoolModel, BruteForceModel placeholder).
- **Contributing**: tests, linting, repo conventions.

## Quickstart (inference)
1) Install and fetch a checkpoint:
   ```bash
   pip install flash-ansr
   flash_ansr install psaegert/flash-ansr-v23.0-120M
   ```
2) Run inference:
   ```python
   from flash_ansr import FlashANSR, SoftmaxSamplingConfig, get_path
   model = FlashANSR.load(
       directory=get_path('models', 'psaegert/flash-ansr-v23.0-120M'),
       generation_config=SoftmaxSamplingConfig(choices=1024),
   )
   y_pred = model.predict(X)
   ```

## Quickstart (docs)
Serve these docs locally:
```bash
pip install mkdocs mkdocs-material mkdocs-autorefs
mkdocs serve
```
Visit http://127.0.0.1:8000.
