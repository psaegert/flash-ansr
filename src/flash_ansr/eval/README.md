# Evaluation Playbook

This document is the companion to the main README’s [Evaluation Quickstart](../../README.md#evaluation-quickstart). It enumerates the exact setup and commands for every evaluation path used in the paper:

- ⚡ANSR (main model) on curated sets, FastSRB, and compute-scaling sweeps
- PySR baseline
- NeSymReS FastSRB baseline

Each section is a checklist: install dependencies, download/patch model assets, run the config, interpret warnings.

## 0. Shared prerequisites

Start from a fresh Python 3.13 environment and a clean checkout of `flash-ansr`.

```bash
git clone https://github.com/psaegert/flash-ansr
cd flash-ansr
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install -e .
```

All commands below assume you stay in the repo root and that datasets live under `data/ansr-data/**` per the main README.

## 1. FlashANSR evaluations

1. **Install the desired checkpoint.** Example: `flash_ansr install-model psaegert/flash-ansr-v23.0-120M` (stores under `models/psaegert/flash-ansr-v23.0-120M/`).
2. **Pick a config under `configs/evaluation/`.**
	- Curated Nguyen/Feynman/Soose sets: `run_flash_ansr_<dataset>.yaml`
	- FastSRB benchmark: `run_fastsrb.yaml`
	- Compute scaling: `scaling/flash_ansr_*.yaml`
3. **Run the CLI.**

	```bash
	flash_ansr evaluate-run \
	  -c configs/evaluation/run_flash_ansr_nguyen.yaml \
	  --limit 1000 --save-every 100 -v
	```

4. **Outputs** land under `results/evaluation/.../*.pkl`. Re-run without `--limit` for the full experiment. Use `--experiment <name>` when a config defines multiple sweeps.

## 2. PySR baseline

1. **Install PySR and Julia dependencies** inside the same environment (PySR pulls JuliaCall automatically).

	```bash
	pip install pysr
	python -c "from pysr import PySRRegressor"
	```

	The import triggers Julia’s precompilation. If Julia isn’t available system-wide, follow the [PySR docs](https://astroautomata.com/PySR/install/) first.

2. **Select a config** (e.g., `configs/evaluation/run_pysr_nguyen.yaml` or the scaling sweep under `scaling/pysr_*.yaml`).
3. **Optional watchdog.** For multi-hour sweeps, `python scripts/evaluate_PySR.py -c <config> --experiment <name> -v` restarts the run if PySR stalls.
4. **Run directly or via watchdog.**

	```bash
	flash_ansr evaluate-run -c configs/evaluation/run_pysr_nguyen.yaml -v
	```

5. **Artifacts** mirror the FlashANSR layout inside `results/evaluation/.../pysr_*.pkl`.

## 3. NeSymReS FastSRB baseline

The goal of this note is to make reproducing the NeSymReS FastSRB evaluation as frictionless as possible for reviewers. Follow the checklist below starting from a fresh Python 3.13 environment with `flash-ansr` cloned locally.

## 1. Environment & installs

```bash
# 1. Create/activate your Python 3.13 environment (conda, venv, etc.)
conda activate flash-ansr-compat  # or your preferred env name

# 2. Install Flash-ANSR in editable mode
pip install -e .

# 3. Install NeSymReS with its modern dependency set
pip install -e nesymres/NeuralSymbolicRegressionThatScales/src

# 4. Install PyTorch Lightning so the NeSymReS checkpoint loader works
pip install pytorch-lightning==2.5.6

# 5. Patch the Python 3.13-incompatible files (rerun after reinstalls)
python scripts/patch_typing_io.py  # fixes typing.io imports + Hydra dataclasses
python scripts/patch_nesymres.py nesymres/NeuralSymbolicRegressionThatScales \
	# pass the path to wherever you cloned NeSymReS
```

What this does:

- Upgrades Hydra to `hydra-core==1.3.2`, OmegaConf to `omegaconf==2.3.0`, and ANTLR to
	`antlr4-python3-runtime==4.9.3`. Hydra/OmegaConf still ship `typing.io` imports and
	Hydra's config dataclasses need `default_factory` tweaks on Python 3.13, so run
	`scripts/patch_typing_io.py` any time those packages are reinstalled.
- Installs the NeSymReS Python package from the submodule so its utilities are on the
	`PYTHONPATH`, then runs `scripts/patch_nesymres.py` to keep its dataclasses and
	packaging metadata compatible with Python 3.13.
- Pulls in Lightning 2.5.x so we can call `Model.load_from_checkpoint`.

> Lightning will warn that the NeSymReS checkpoint was saved with an older release.
> You can optionally run `python -m pytorch_lightning.utilities.upgrade_checkpoint models/nesymres/100M.ckpt`
> to silence the message. Inference works either way.

## 2. Wiring the evaluation config

`configs/evaluation/scaling/nesymres_fastsrb.yaml` now points straight at the tracked
model assets, so no extra copying is needed:

```yaml
model_adapter:
	type: nesymres
	eq_setting_path: "{{ROOT}}/models/nesymres/eq_setting.json"
	config_path: "{{ROOT}}/models/nesymres/config.yaml"
	weights_path: "{{ROOT}}/models/nesymres/100M.ckpt"
	simplipy_engine: "dev_7-3"
	n_restarts: 4
	device: cuda
```

Make sure `models/nesymres/` contains the 100M checkpoint triplet before running.

## 3. Running a smoke test

```bash
flash_ansr evaluate-run \
	-c configs/evaluation/scaling/nesymres_fastsrb.yaml \
	--experiment nesymres_fastsrb_beam_00001 \
	--limit 2 -v
```

Expected output: `results/evaluation/scaling/nesymres/fastsrb/beam_00001.pkl` with two
records. Drop `--limit` and iterate over the remaining experiments for the full sweep.

## 4. What changed in this repo

We made the following source tweaks so the above process “just works” on 3.13:

1. **NeSymReS packaging** (`nesymres/…/setup.py`): bumped dependency ranges to
	 `hydra-core>=1.3.2,<1.4` and `omegaconf>=2.3.0,<2.4`. Hydra now brings in a compatible
	 ANTLR runtime, so no manual edits are necessary.
2. **Dataclass fix** (`nesymres/dclasses.py`): set `bfgs: BFGSParams = field(default_factory=BFGSParams)`
	 to avoid mutable-default errors on Python 3.13.
3. **Config references** (`configs/evaluation/scaling/nesymres_fastsrb.yaml`): updated the
	 `eq_setting_path`, `config_path`, and `weights_path` entries to `models/nesymres/**` so
	 reviewers can use the tracked assets directly.

With these committed changes, no site-packages surgery is required—installing from the
repo is enough.

## 5. Known warnings

- `torch was imported before juliacall`: emitted because the environment also has PySR
	(which depends on JuliaCall). It’s harmless for NeSymReS runs.
	- Lightning checkpoint migration: see §1 if you want to upgrade the checkpoint in place.

---

Need another evaluation scenario? Mirror one of the configs in `configs/evaluation/`, adjust the `model_adapter` block, and keep the workflow above. Contributions that add new baselines should update this playbook so every reviewer can reproduce them without guesswork.
