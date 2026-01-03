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
./scripts/train.sh configs/test   # uses tiny fixtures; finishes quickly
```
Produces checkpoints under `models/ansr-models/test/` with `model.yaml`, `tokenizer.yaml`, and `state_dict.pt`.

## Full training workflow
1. **Import test sets**: Ajdust and run `./scripts/import_test_sets.sh` to import test sets. The data generating processes during training will exclude these skeletons to ensure fair evaluation.
2. **Configure skeleton pools and datasets**: Adjust the `skeleton_pool_*.yaml` and `dataset_*.yaml` files inside your chosen config bundle to set operator priors, expression depths, and data sampling strategies.
3. **Prepare held out skeleton pools** (optional if reusing shipped ones):
    ```bash
    flash_ansr generate-skeleton-pool \
    -c "./configs/my_model/skeleton_pool_val.yaml" \
    -o "./data/ansr-data/my_model/skeleton_pool_val" \
    -s 1000 -v  # 1000 skeletons for validation
    ```
4. **Launch training**:
    ```bash
    flash_ansr train
    -c "./configs/my_model/train.yaml"
    -o "./models/ansr-models/my_model"
    -v
    -ci 250000  # Checkpoint every 250k steps
    -vi 10000  # Validate every 10k steps
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
- Copy an existing bundle (e.g., `configs/v23.0-60M/`).
- Update paths and tokenizer/operator choices; ensure special prompt tokens exist before enabling prompt features.
- Prefer cloning configs instead of mutating in-place to keep saved YAMLs portable.

## Validation during training
- Use `generate_validation_set.sh` to sample held-out skeletons matching the train pool schema.
- Point `dataset_val.yaml` to the generated pool and set validation frequency to track generalization.

## Exporting checkpoints
- Use `FlashANSRModel.save` outputs (`model.yaml`, `tokenizer.yaml`, `state_dict.pt`) together; consumers expect the trio.
