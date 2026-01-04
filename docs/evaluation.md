# Evaluation

## Read this first
- Quick run: `flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml --experiment flash_ansr_fastsrb_choices_00032 -v`.
- Outputs: pickles with entries containing `expression`, `log_prob`, `fits` (per-dataset metrics), and optional `placeholder` entries when data generation fails but counts must stay aligned.
- Scope: shared engine covers FlashANSR, PySR, NeSymReS, SkeletonPool, BruteForce, and E2E baselines via a single YAML config.

## General workflow

- Use `flash_ansr evaluate-run -c <config>` as the single entrypoint; configs live under `configs/evaluation/` (with `scaling/`, `noise_sweep/`, and `support_sweep/` families).
- Each config wires a `data_source`, a `model_adapter`, and a `runner` (persistence/resume). The same structure covers FlashANSR, PySR, NeSymReS, and baselines.
- `runner.resume` allows checkpointed pickles to continue; placeholders are inserted when sample generation fails so counts stay consistent.
- `datasets_per_expression` controls how many deterministic datasets per skeleton/equation are generated; sampling mode is removed.

## Models and baselines

- **FlashANSR**: Default adapter; supports generation overrides (beam/softmax/MCTS) and prompt options. See the scaling configs under `configs/evaluation/scaling/v23.0-*/`.
- **[PySR](https://github.com/MilesCranmer/PySR)**: Adapter expects PySR installed; config fields mirror PySR runtime knobs (timeout, iterations, parsimony). Watchdog helper: `scripts/evaluate_PySR.py`.
- **[NeSymReS](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales)**: Adapter expects external checkout + checkpoint paths; exposes beam width/restarts. See `run_nesymres.yaml` and scaling configs.
- **SkeletonPoolModel (baseline)**: Transformer-free baseline that samples skeletons from a provided pool and only refines constants. Configure via `model_adapter.type: skeleton_pool` (or add a dedicated config entry) with pool path/config, `samples`, `unique`, `ignore_holdouts`, and `seed`. Useful for ablations and replication.
- **BruteForceModel (baseline)**: Exhaustive baseline over a provided skeleton pool. Configs live alongside the scaling files.
- **[E2E](https://github.com/facebookresearch/symbolicregression)**: External transformer baseline. Requires the authors' `model1.pt`, a working `symbolicregression` install, and the `e2e_fastsrb` scaling config.

### External model setup (one-time)

**PySR**
1. Install PySR into the same environment as flash-ansr: `pip install pysr`.
2. Trigger Julia precompilation (first import is slow): `python -c "from pysr import PySRRegressor"`.
3. Optional but recommended for long sweeps: use the watchdog wrapper `python scripts/evaluate_PySR.py -c <config> --experiment <name> -v` to auto-restart if PySR stalls.

**NeSymReS**
1. Clone their repo (see README) and install: `pip install -e nesymres/NeuralSymbolicRegressionThatScales/src`.
2. Install Lightning compatible with the checkpoint loader: `pip install pytorch-lightning==2.5.6`.
3. Patch Python 3.13 incompatibilities: `python scripts/patch_typing_io.py` then `python scripts/patch_nesymres.py nesymres/NeuralSymbolicRegressionThatScales`.
4. Place the checkpoint triplet under `models/nesymres/`: `eq_setting.json`, `config.yaml`, `100M.ckpt`.

**E2E (End-to-end symbolic regression)**
1. From `e2e/symbolicregression`, install dependencies (`pip install -r requirements.txt` or use the authors' `environment.yml`).
2. Patch for modern numpy + scaler guard + `tree_idx` compatibility (idempotent): `python scripts/patch_symbolicregression.py e2e/symbolicregression`.
3. Install the method with `pip install -e .`.
4. Install the required sympytorch fork: `pip install git+https://github.com/pakamienny/sympytorch.git`.
5. Download the pretrained checkpoint to `e2e/model1.pt` (mirror of https://dl.fbaipublicfiles.com/symbolicregression/model1.pt). Keep the filename as-is; the scaling config points there.

## Express

Use, copy or modify a config in `./configs`:

```
./configs
├── my_config
│   ├── dataset_train.yaml          # Link to skeleton pool and padding for training
│   ├── dataset_val.yaml            # Link to skeleton pool and padding for validation
│   ├── tokenizer.yaml              # Tokenizer settings
│   ├── model.yaml                  # Model settings and link to simplipy engine
│   ├── skeleton_pool_train.yaml    # Sampling and holdout settings for training
│   ├── skeleton_pool_val.yaml      # Sampling and holdout settings for validation
│   └── train.yaml                  # Data and schedule for training
```

Use the helper scripts to import data, build validation sets, and kick off training:

```sh
./scripts/import_test_sets.sh                     # optional, required only once per checkout
./scripts/generate_validation_set.sh my_config    # prepares validation skeletons
./scripts/train.sh my_config                      # trains using configs/my_config
```

For more information see below.

## Manual

### 0. Prerequisites

Test data structured as follows:

```sh
./data/ansr-data/test_set
├── fastsrb
│   └── expressions.yaml
```

The test data can be cloned from the Hugging Face data repository:

```sh
git clone https://huggingface.co/psaegert/ansr-data data/ansr-data
```

### 1. Import test data

External datasets must be imported into the supported format:

```sh
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/fastsrb/expressions.yaml" -p "fastsrb" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/fastsrb/skeleton_pool" -v
```

with

- `-i` the input file
- `-p` the name of the parser implemented in `./src/flash_ansr/compat/convert_data.py`
- `-e` the SimpliPy engine version to use for simplification
- `-b` the config of a base skeleton pool to add the data to
- `-o` the output directory for the resulting skeleton pool
- `-v` verbose output

This will create and save a skeleton pool with the parsed imported skeletons in the specified directory:

```sh
./data/ansr-data/test_set/<test_set>
└── skeleton_pool
    ├── skeleton_pool.yaml
    └── skeletons.pkl
```

### 2. Generate validation data

Validation data is generated by randomly sampling according to the settings in the skeleton pool config:

```sh
flash_ansr generate-skeleton-pool -c {{ROOT}}/configs/${CONFIG}/skeleton_pool_val.yaml -o {{ROOT}}/data/ansr-data/${CONFIG}/skeleton_pool_val -s 5000 -v
```

with

- `-c` the skeleton pool config
- `-o` the output directory to save the skeleton pool
- `-s` the number of unique skeletons to sample
- `-v` verbose output

### 3. Train the model

```sh
flash_ansr train -c {{ROOT}}/configs/${CONFIG}/train.yaml -o {{ROOT}}/models/ansr-models/${CONFIG} -v -ci 100000 -vi 10000
```

with

- `-c` the training config
- `-o` the output directory to save the model and checkpoints
- `-v` verbose output
- `-ci` the interval to save checkpoints
- `-vi` the interval for validation

### 4. Evaluate the model

⚡ANSR, PySR, NeSymReS, E2E, skeleton-pool, brute-force, and the FastSRB benchmark run through a shared evaluation engine.
Each run is configured in a single YAML that wires a **data source**, a **model adapter**, and runtime **runner** settings.
The common CLI entry point is:

```sh
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml --experiment flash_ansr_fastsrb_choices_00032 -v
```

Use `-n/--limit`, `--save-every`, `-o/--output-file`, `--experiment <name>`, or `--no-resume` to temporarily override the config without editing the file. When a config defines multiple experiments (see `configs/evaluation/scaling/`), omitting `--experiment` now runs **all** of them sequentially; pass an explicit name if you only want a single sweep entry.

#### 4.1 Config-driven workflow

Every run config (see `configs/evaluation/*.yaml`) follows the same structure:

```yaml
run:
  data_source:  # how to create evaluation samples
    ...
  model_adapter:  # which model/baseline to call
    ...
  runner:        # bookkeeping + persistence
    limit: 5000
    save_every: 250
    output: "{{ROOT}}/results/evaluation/v23.0-20M/fastsrb.pkl"
    resume: true
```

- **`data_source`** selects where problems come from. `type: skeleton_dataset` streams from a `FlashANSRDataset`, while `type: fastsrb` reads the FastSRB YAML benchmark. Common knobs include `n_support`, `noise_level`, and target sizes. Provide `datasets_per_expression` to iterate each skeleton or FastSRB equation deterministically with a fixed number of generated datasets (handy for reproducible evaluation sweeps).
- **`model_adapter`** declares the solver. Supported values today are `flash_ansr`, `pysr`, `nesymres`, `skeleton_pool`, `brute_force`, and `e2e`, each with their own required fields (model paths, timeout/beam/samples knobs, etc.).
- **`runner`** controls persistence: `limit` caps the number of processed samples, `save_every` checkpoints incremental progress to `output`, and `resume` decides whether to load previous results from that file.

When `resume` is enabled the engine simply reloads the existing pickle, skips that many deterministic samples, and keeps writing to the same file. If a dataset cannot be generated within `max_trials`, the runner now appends a placeholder entry (`placeholder=True`, `placeholder_reason=...`) so the results length still reflects every attempted expression/dataset pair. Downstream analysis can filter those placeholders, but their presence keeps pause/resume logic trivial and avoids juggling extra state files. Skeleton dataset evaluations remain sequential—`datasets_per_expression` (default `1`) controls how many deterministic datasets are emitted per skeleton, and the previous random sampling mode has been removed.

Running `flash_ansr evaluate-run ...` loads the config, resumes any previously saved pickle, instantiates the requested data/model pair, and streams results back into the same output file.

#### 4.2 Example run configs

Ready-to-use configs live under `configs/evaluation/scaling/` (with matching `noise_sweep/` and `support_sweep/` variants). All shipped experiments target FastSRB; the `*_v23_val.yaml` siblings swap in the v23 validation skeleton pool.

##### 4.2.1 FlashANSR

`configs/evaluation/scaling/v23.0-20M_fastsrb.yaml` (plus the 3M and 120M variants) sweep SoftmaxSampling `choices`. Example:

```sh
flash_ansr evaluate-run \
  -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml \
  --experiment flash_ansr_fastsrb_choices_00032 -v
```

##### 4.2.2 PySR

`configs/evaluation/scaling/pysr_fastsrb.yaml` mirrors the same sweep over `niterations`. Run a single point with:

```sh
flash_ansr evaluate-run \
  -c configs/evaluation/scaling/pysr_fastsrb.yaml \
  --experiment pysr_fastsrb_iter_00032 -v
```

For long sweeps, `python scripts/evaluate_PySR.py -c <config> --experiment <name> -v` restarts jobs if PySR stalls.

##### 4.2.3 NeSymReS

`configs/evaluation/scaling/nesymres_fastsrb.yaml` varies `beam_width` for the 100M checkpoint tracked under `models/nesymres/`. Example:

```sh
flash_ansr evaluate-run \
  -c configs/evaluation/scaling/nesymres_fastsrb.yaml \
  --experiment nesymres_fastsrb_beam_width_00008 -v
```

##### 4.2.4 Skeleton pool baseline

`configs/evaluation/scaling/skeleton_pool_fastsrb.yaml` samples skeletons directly from `data/ansr-data/test_set/fastsrb/skeleton_pool_max8`. Example:

```sh
flash_ansr evaluate-run \
  -c configs/evaluation/scaling/skeleton_pool_fastsrb.yaml \
  --experiment skeleton_pool_fastsrb_samples_00032 -v
```

##### 4.2.5 Brute force baseline

`configs/evaluation/scaling/brute_force_fastsrb.yaml` exhaustively enumerates skeletons up to `max_expressions`. Example:

```sh
flash_ansr evaluate-run \
  -c configs/evaluation/scaling/brute_force_fastsrb.yaml \
  --experiment brute_force_fastsrb_max_expressions_00064 -v
```

##### 4.2.6 E2E baseline

`configs/evaluation/scaling/e2e_fastsrb.yaml` sweeps `model_adapter.candidates_per_bag` (the beam size). Example:

```sh
flash_ansr evaluate-run \
  -c configs/evaluation/scaling/e2e_fastsrb.yaml \
  --experiment e2e_fastsrb_candidates_00016 -v
```

##### 4.2.7 Compute-scaling sweeps

All scaling configs are multi-experiment. Omit `--experiment` to run the full sweep; the primary knobs are:

- **FlashANSR**: `generation_overrides.kwargs.choices`
- **PySR**: `niterations`
- **NeSymReS**: `beam_width`
- **SkeletonPool**: `samples`
- **BruteForce**: `max_expressions`
- **E2E**: `candidates_per_bag`

Outputs are namespaced under `results/evaluation/scaling/<model>/<dataset>/...` so sweeps can run back-to-back.
