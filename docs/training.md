# Training

## Prerequisites
- Install flash-ansr: `pip install flash-ansr`

## Data and configs
- Training consumes procedurally generated samples built from skeleton pools and support points.
- A typical config bundle lives under `configs/<name>/` and includes:
	- `train.yaml`: trainer, datasets, schedule, logging.
	- `dataset_*.yaml`: dataset/pool paths, padding, prompt settings.
	- `tokenizer.yaml`: operators and special prompt tokens.
	- `model.yaml`: architecture, precision, and simplifier config.
	- `skeleton_pool_*.yaml`: sampling and holdout settings.
- Keep paths relative; `load_config(..., resolve_paths=True)` normalizes `./` and `{{ROOT}}`.

## Minimal runnable example
```bash
./scripts/train.sh test   # uses tiny fixtures; finishes quickly
```
Produces checkpoints under `models/ansr-models/test/` with `model.yaml`, `tokenizer.yaml`, and `state_dict.pt`.

## Helper scripts
- `./scripts/generate_validation_set.sh <config>`: create a held-out validation skeleton pool matching your bundle (wraps `symbolic_data.SkeletonPool`).
- `./scripts/generate_test_set.sh <config>`: create a test-set skeleton pool from `configs/test_set/<config>/`.
- `./scripts/import_test_sets.sh`: **pending symbolic-data 0.2; currently prints a notice and exits 1.** Once the data-layer ingest CLI ships it will build a holdout pool from a raw benchmark spec (so training excludes evaluation skeletons); see step 1 below for the interim.
- `./scripts/train.sh <config>`: convenience wrapper to launch training with the bundle.

> Pool generation moved out of flash-ansr in 0.7: the model package consumes skeleton pools and no longer ships a data CLI (the removed `generate-skeleton-pool` / `import-data` / `filter-skeleton-pool` / `split-skeleton-pool` subcommands). The pool/sampling API lives in [symbolic-data](https://github.com/psaegert/symbolic-data) (a flash-ansr dependency); a first-class `symbolic-data` CLI is planned for symbolic-data 0.2. The helper scripts and `configs/` bundles referenced here live in the repository, not the PyPI wheel; clone the repo to use them.

## Full training workflow
1. **Import test sets** (training-time decontamination): training excludes the skeletons listed under `holdout_pools:` in your `skeleton_pool_*.yaml`, so the data-generating process never samples a benchmark test skeleton. Building that holdout pool from a raw benchmark spec (the old `import_test_sets.sh`) needs the data-ingest tooling that moved to [symbolic-data](https://github.com/psaegert/symbolic-data) and is **pending symbolic-data 0.2**: no interim command reproduces it (`generate_test_set.sh` samples *random* skeletons, not the benchmark equations). Until then: if you already have a benchmark holdout pool on disk, point `holdout_pools:` at it (this is what the shipped `configs/v23.*` bundles do); otherwise train with `holdout_pools: []`, which is fine for non-benchmark use, but note that benchmark eval numbers (e.g. FastSRB/Feynman) will be contaminated until you can decontaminate. (`symbolic_data.load_benchmark('fastsrb')` exposes the FastSRB benchmark for *evaluation* via srbf, not training-holdout exclusion.)
2. **Configure skeleton pools and datasets**: Adjust the `skeleton_pool_*.yaml` and `dataset_*.yaml` files inside your chosen config bundle to set operator priors, expression depths, and data sampling strategies.
3. **Prepare held out skeleton pools** (optional if reusing shipped ones): generate a pool with the `symbolic-data` Python API, or use `./scripts/generate_validation_set.sh <config>`. Run from the project root so the relative paths resolve:
    ```python
    from symbolic_data import SkeletonPool

    config = "./configs/my_model/skeleton_pool_val.yaml"
    pool = SkeletonPool.from_config(config)
    pool.create(size=1000, verbose=True)  # 1000 skeletons for validation
    pool.save("./data/ansr-data/my_model/skeleton_pool_val", config=config)
    ```
4. **Launch training**:
    ```bash
    # -ci/--checkpoint-interval: checkpoint every 250k steps
    # -vi/--validate-interval: validate every 10k steps
    flash_ansr train \
    -c "./configs/my_model/train.yaml" \
    -o "./models/ansr-models/my_model" \
    -v \
    -ci 250000 \
    -vi 10000
    ```
5. **Logging**: Enable/disable W&B logging via `wandb_mode` inside the config.
6. **Resume**: Continue from any checkpoint directory using `--resume-from` (optionally `--resume-step` when the step cannot be inferred).

### Resuming training
- Checkpoints are written under `<output-dir>/checkpoint_<step>/` when `-ci/--checkpoint-interval` is set. Each checkpoint contains `state_dict.pt` (model), `optimizer.pt`, `lr_scheduler.pt`, `scaler.pt`, and `training_state.pt` with the recorded `step`.
- Resume with the same config you trained with and point `--resume-from` at the checkpoint directory:
    ```bash
    flash_ansr train \
    -c "./configs/my_model/train.yaml" \
    -o "./models/ansr-models/my_model" \
    --resume-from "./models/ansr-models/my_model/checkpoint_250000" \
    -v
    ```
- The trainer infers the global step from `training_state.pt` or the folder name (`checkpoint_<step>`). If you renamed the folder or the metadata is missing, supply `--resume-step <step>` to realign the optimizer and LR schedule.
- The run keeps training until the target `steps` from the config are reached, so keep that value consistent when resuming.

## Data loader/runtime tips
- `FlashANSRDataset` uses multiprocessing; call `dataset.shutdown()` if you exit to minimize chances of hanging processes.
- Training uses [automatic mixed precision (AMP)](https://docs.pytorch.org/docs/stable/amp.html) and [torch.compile](https://huggingface.co/docs/transformers/en/perf_torch_compile)

## Adding a new config
- Copy an existing bundle (e.g., `configs/v23.0-20M/`).
- Update paths and tokenizer/operator choices; ensure special prompt tokens exist before enabling prompt features.
- Prefer cloning configs instead of mutating in-place to keep saved YAMLs portable.

## Validation during training
- Use `generate_validation_set.sh` to sample held-out skeletons matching the train pool schema.
- Point `dataset_val.yaml` to the generated pool and set validation frequency to track generalization.

## Exporting checkpoints
- Use `FlashANSRModel.save` outputs (`model.yaml`, `tokenizer.yaml`, `state_dict.pt`) together; consumers expect the trio.
