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

Constant refinement runs a parallel worker pool. By default (`refiner_workers=None`) the pool uses every available CPU core, which oversubscribes shared machines. Cap it explicitly at load time:

```python
FlashANSR.load(..., refiner_workers=N, persistent_refine_pool=True)
```

Pass an explicit integer to cap the pool size (`0` disables multiprocessing). This is the right knob on shared machines where the default (all cores) would oversubscribe the CPUs.

### How do I turn off the inference-speed defaults?

KV-cache decoding, auto-batching and static decoding are enabled by default and designed to be quality-neutral. To opt out:

```python
from flash_ansr import SoftmaxSamplingConfig
SoftmaxSamplingConfig(choices=1024, use_cache=False, batch_size=128, static_decode=False)
```

Candidate ranking is configured with `ranking_mode` (`mdl`, `weighted` or `pareto`) and that mode's knobs; the pre-0.14 length penalty is `ranking_mode="weighted", ranking_weights={"n_nodes": 0.05}`.

### How do I evaluate a model and run benchmarks?

The evaluation engine, baseline adapters, benchmarks, and metrics live in the companion package [**srbf**](https://github.com/psaegert/srbf) (Symbolic Regression Benchmark Framework). Install it with `pip install srbf` and see the [srbf repository](https://github.com/psaegert/srbf) for usage. Flash-ANSR itself still provides the `FlashANSR` API (`.load`, `.fit`, `.predict`, `.compile_results`) and training, but systematic evaluation and benchmarking now live in srbf.

### Where do I report bugs or ask questions?

Open an issue at [github.com/psaegert/flash-ansr/issues](https://github.com/psaegert/flash-ansr/issues).
