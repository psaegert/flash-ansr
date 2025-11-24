# Notes on evaluating NeSymReS models

The goal of this note is to make reproducing the NeSymReS FastSRB evaluation as
frictionless as possible for reviewers. Follow the checklist below starting from a
fresh Python 3.13 environment with `flash-ansr` cloned locally.

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
