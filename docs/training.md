# Training

## Prerequisites
- Install flash-ansr: `pip install flash-ansr`

## Data and configs
- Training consumes procedurally generated problems (expression skeletons + their support points) streamed from a [symbolic-data](https://github.com/psaegert/symbolic-data) **catalog** via a **`ProblemSource`**. flash-ansr no longer ships a data CLI or its own data classes; the catalog/source API lives in `symbolic_data` (a flash-ansr dependency).
- A typical config bundle lives under `configs/<name>/` and includes:
	- `train.yaml`: trainer, datasets, schedule, logging.
	- `dataset_*.yaml`: the data source (a `source:` block), padding, and prompt settings.
	- `catalog_*.yaml`: the generative catalog recipe (operator priors, expression-length distribution, literal/support priors, decontamination).
	- `tokenizer.yaml`: operators and special prompt tokens.
	- `model.yaml`: architecture, precision, and simplifier config.
- Keep paths relative; `load_config(...)` normalizes `./` and `{{ROOT}}`.

### How a dataset config consumes a catalog
`FlashANSRDataset.from_config` reads a `source:` block and hands it to a `symbolic_data.ProblemSource`. The source samples a catalog into ready-to-use problems under a usage policy (`sampling:`):

```yaml
# dataset_train.yaml
source:
  catalog: './catalog_train.yaml'   # name | local path | saved-catalog dir | inline {type: ...}
  sampling:
    n_support: prior                # generative recipes: draw the per-sample support size from the catalog's prior
    n_validation: 0                 # n_support: prior implies no validation split
    noise: 0.0
tokenizer: './tokenizer.yaml'
padding: 'zero'
```

`source.catalog` may be:

- an **inline** generative-catalog dict (`{type: lample_charton, ...}`),
- a **local path** to a catalog config YAML (e.g. `./catalog_train.yaml`),
- a **directory** holding a saved/frozen catalog (a fixed validation set), or
- a **curated name** resolved from the symbolic-data Hugging Face manifest (e.g. `fastsrb`, `feynman`, `nguyen`).

`sampling:` keys: `n_support` (`prior` for generative recipes, or an integer), `n_validation`, `noise`, and `problems_per_expression`.

The `catalog_*.yaml` itself is a generative recipe; for the shipped v23 bundles its `type: lample_charton` recipe carries the operator weights, the expression-length distribution, and the literal/support priors. This `catalog_train.yaml` *is* the v23 training recipe (there is no shortcut "name" for it; it ships in the bundle).

## Minimal runnable example
```bash
./scripts/train.sh test   # uses tiny fixtures; finishes quickly
```
Produces checkpoints under `models/ansr-models/test/` with `model.yaml`, `tokenizer.yaml`, and `state_dict.pt`.

## Helper scripts
- `./scripts/train.sh <config>`: convenience wrapper to launch `flash_ansr train` with the bundle (`configs/<config>/train.yaml`).

> Data generation moved out of flash-ansr: the model package consumes catalogs and no longer ships a data CLI or any data-generation helper scripts. The catalog/source/sampling API lives in [symbolic-data](https://github.com/psaegert/symbolic-data); use its tooling to build, save, and publish catalogs. The helper script and `configs/` bundles referenced here live in the repository, not the PyPI wheel; clone the repo to use them.

## Full training workflow
1. **Decontaminate against benchmarks** (training-time exclusion): the generative catalog excludes every skeleton listed in its registered holdout catalogs, so the data-generating process never samples a held-out benchmark skeleton. Add a `holdout_pools:` list to the `catalog_*.yaml` recipe (this is the mechanism `FlashANSRDataset` actually honors, since it builds the catalog from this config):
    ```yaml
    # catalog_train.yaml
    type: lample_charton
    # ... operator weights, priors ...
    holdout_pools:
      - "v23-val"                            # a curated catalog NAME (resolved from the HF asset repo)
      - "./data/my_model/fastsrb_holdout"    # ...or a path to a saved catalog directory to exclude
    ```
    Each entry is a curated catalog **name** (`v23-val`, `fastsrb`, ..., or a `name@version` resolved from the HF asset repo — the shipped bundles use names), a **path** to a saved catalog directory, or a `LampleChartonCatalog` instance when building in Python; the catalog caches those skeletons and drops any generated expression that matches one. Matching is structural and constant-agnostic, so a generated `x1..` skeleton is excluded if it matches a held-out benchmark expression's structure. Use `holdout_pools: []` (or omit it) for non-benchmark use; if you skip it, benchmark eval numbers (e.g. FastSRB / Feynman) may be contaminated. Build the holdout catalog with the `symbolic_data` API (see step 3) and save it to the directory you reference here.
2. **Configure the catalog and dataset**: adjust `catalog_*.yaml` (operator priors, expression-length distribution, literal/support priors) and `dataset_*.yaml` (the `source:` block) inside your chosen bundle.
3. **Prepare a held-out validation set** (optional if reusing a shipped one): build a fixed catalog with the `symbolic_data` Python API and save it to disk, then point `dataset_val.yaml`'s `source.catalog` at that directory. Run from the project root so relative paths resolve:
    ```python
    from symbolic_data import LampleChartonCatalog

    config = "./configs/my_model/catalog_val.yaml"
    catalog = LampleChartonCatalog.from_config(config)
    catalog.create(size=1000, verbose=True)            # realize 1000 fixed validation skeletons
    catalog.save("./data/my_model/catalog_val", config=config)
    ```
    Then in `dataset_val.yaml`:
    ```yaml
    source:
      catalog: "./data/my_model/catalog_val"   # the saved (frozen) catalog directory
      sampling:
        n_support: prior
        n_validation: 0
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
5. **Logging**: enable/disable W&B logging via `--mode` (`online` / `offline` / `disabled`).
6. **Resume**: continue from any checkpoint directory using `--resume-from` (optionally `--resume-step` when the step cannot be inferred).

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
- `FlashANSRDataset` wraps a `ProblemSource` and uses multiprocessing; call `dataset.shutdown()` if you exit to minimize chances of hanging processes.
- Training uses [automatic mixed precision (AMP)](https://docs.pytorch.org/docs/stable/amp.html) and [torch.compile](https://huggingface.co/docs/transformers/en/perf_torch_compile)

## Adding a new config
- Copy an existing bundle (e.g., `configs/v23.0-3M-pma-k1/`).
- Update paths and tokenizer/operator choices; ensure special prompt tokens exist before enabling prompt features.
- Prefer cloning configs instead of mutating in-place to keep saved YAMLs portable.

## Validation during training
- Build a held-out validation catalog (see step 3) and point `dataset_val.yaml`'s `source.catalog` at the saved directory.
- Set the validation frequency (`-vi/--validate-interval`) to track generalization.

## Exporting checkpoints
- Use `FlashANSRModel.save` outputs (`model.yaml`, `tokenizer.yaml`, `state_dict.pt`) together; consumers expect the trio.
