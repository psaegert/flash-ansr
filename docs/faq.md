# FAQ & Troubleshooting

- **SimpliPy assets missing/download stalls**: set `HF_HOME`/`HF_HUB_CACHE` to a writable path; run once with internet so `SimpliPyEngine.load(..., install=True)` can fetch rule sets.
- **CUDA OOM**: lower `n_support`, reduce `beam_width`/`choices`, or move to CPU (`model.to('cpu')`). For training, reduce batch size or sequence length in configs.
- **Tokenizer lacks prompt tokens**: ensure tokenizer YAML includes special tokens before enabling prompt-related preprocessing; otherwise `serialize_prompt_prefix` will warn and skip prompts.
- **Dataset workers hanging**: call `dataset.shutdown()` on early exit or when breaking out of iteration.
- **Evaluation resumes unexpectedly**: `runner.resume: true` reloads existing pickle files; set `--no-resume` or point `--output-file` to a new path to avoid appending.
- **PySR/NeSymReS not found**: install extras in the active env and configure their paths in the evaluation YAML (see `run_pysr_*.yaml`, `run_nesymres.yaml`).
- **Expression looks wrong but fits data**: increase `generation_config` strength (beam width / choices) or enable completed-beam deduplication; consider refining constants again via `Refiner` if the structure is correct.
