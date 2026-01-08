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

## External model setup (one-time)

**PySR**
1. Install PySR into the same environment as flash-ansr: `pip install pysr`.
2. Trigger Julia precompilation (first import is slow): `python -c "from pysr import PySRRegressor"`.
3. Optional but recommended for long sweeps: use the watchdog wrapper `python scripts/evaluate_PySR.py -c <config> --experiment <name> -v` to auto-restart if PySR stalls.

**NeSymReS**
1. Clone their repo and install: `pip install -e nesymres/NeuralSymbolicRegressionThatScales/src`.
2. Install Lightning compatible with the checkpoint loader: `pip install pytorch-lightning==2.5.6`.
3. Patch Python 3.13 incompatibilities: `python scripts/patch_typing_io.py` then `python scripts/patch_nesymres.py nesymres/NeuralSymbolicRegressionThatScales`.
4. Place the checkpoint triplet under `models/nesymres/`: `eq_setting.json`, `config.yaml`, `100M.ckpt`.

**E2E (End-to-end symbolic regression)**
1. Clone the repo: `git clone https://github.com/facebookresearch/symbolicregression.git e2e/symbolicregression`
2. Patch for modern numpy + scaler guard + `tree_idx` compatibility (idempotent): `python scripts/patch_symbolicregression.py e2e/symbolicregression`.
3. Editable install (deps auto-resolved, includes sympytorch): `pip install -e e2e/symbolicregression`.
4. Download the pretrained checkpoint to `/models/e2e/model1.pt` (mirror of https://dl.fbaipublicfiles.com/symbolicregression/model1.pt). Keep the filename as-is; the scaling config points there.
```
mkdir -p models/e2e
wget -O models/e2e/model1.pt https://dl.fbaipublicfiles.com/symbolicregression/model1.pt
```

## Configs at a glance

- Evaluation configs live under `configs/evaluation/` (families: `scaling/`, `noise_sweep/`, `support_sweep/`).
- Each file is a single run definition: `data_source`, `model_adapter`, and `runner` blocks.
- Multi-experiment configs run **all** experiments when `--experiment` is omitted; pass a name to isolate one.
- Outputs default to `results/evaluation/...` as specified in the config; override with `-o/--output-file`.

## Step-by-step run guide

### 0. Benchmark data

Fetch the FastSRB benchmark once (if you do not already have `data/ansr-data/test_set/fastsrb/expressions.yaml`):

```sh
mkdir -p "{{ROOT}}/data/ansr-data/test_set/fastsrb"
wget -O "{{ROOT}}/data/ansr-data/test_set/fastsrb/expressions.yaml" \
  "https://raw.githubusercontent.com/viktmar/FastSRB/refs/heads/main/src/expressions.yaml"
```

This writes `skeleton_pool.yaml` and `skeletons.pkl` under the specified output directory.

### 1. Run evaluation

```sh
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml --experiment flash_ansr_fastsrb_choices_00032 -v
```
or

```sh
flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml -v
```
to run all experiments in the config.

- Adjust `-c` to any file under `configs/evaluation/` and optionally set `--experiment`.
- Override on the fly: `-n/--limit`, `--save-every`, `-o/--output-file`, `--no-resume`.
- The runner loads existing partial pickles, skips processed items, and appends new results. If sample generation fails within `max_trials`, a placeholder entry is written to preserve counts.

### 2. Example configs

- FlashANSR v23.0-20M scaling: `configs/evaluation/scaling/v23.0-20M_fastsrb.yaml`
- PySR scaling: `configs/evaluation/scaling/pysr_fastsrb.yaml`
- NeSymReS scaling: `configs/evaluation/scaling/nesymres_fastsrb.yaml`
- E2E baseline: `configs/evaluation/scaling/e2e_fastsrb.yaml`
