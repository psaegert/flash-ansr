# FAQ & Troubleshooting

### Where are models downloaded to?

`install_model(MODEL)` downloads to `./models/` in the package root by default, and `FlashANSR.load(directory=get_path('models', MODEL))` reads from there. Set the `FLASH_ANSR_ROOT` environment variable to point `get_path`/`get_root` at a different project root.

### How do I run on GPU vs CPU?

Flash-ANSR follows standard PyTorch device handling. Move the model with `.to(device)` after loading:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlashANSR.load(...).to(device)
```

If CUDA is unavailable, everything runs on CPU (slower, but functional).

### Constant refinement is using too many CPU cores

Constant refinement runs a parallel worker pool. Cap it explicitly at load time:

```python
FlashANSR.load(..., refiner_workers=N, persistent_refine_pool=True)
```

This is the right knob on shared machines where the default worker count would oversubscribe the CPUs.

### How do I reproduce v0.4.x inference behavior?

The v0.5 speed defaults (KV-cache, auto-batching, static decoding) are enabled by default and designed to be quality-neutral. To opt out:

```python
from flash_ansr import SoftmaxSamplingConfig
SoftmaxSamplingConfig(choices=1024, use_cache=False, batch_size=128, static_decode=False)
```

Also note the v0.5 rename: the candidate-selection penalty `parsimony` is now `length_penalty`.

### Should I evaluate with `flash_ansr evaluate-run` or with srbf?

Both work. The in-repo `flash_ansr evaluate-run` path is fully supported. The companion package [**srbf**](https://github.com/psaegert/srbf) packages the same evaluation engine, adapters, benchmarks, and metrics as a standalone framework; prefer it for systematic benchmarking or for evaluating models other than Flash-ANSR.

### Where do I report bugs or ask questions?

Open an issue at [github.com/psaegert/flash-ansr/issues](https://github.com/psaegert/flash-ansr/issues).
