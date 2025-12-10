# Contributing

## Repo layout
- Code: `src/flash_ansr/`
- Tests: `tests/`
- Configs: `configs/`
- Scripts: `scripts/` (training/eval helpers)

## Developer setup
- Use the project conda env and install extras in `requirements.txt`.
- Large assets (models/data) are pulled via the provided scripts or `flash_ansr install` commands.

## Quality checks
- Tests: `./scripts/pytest.sh` (full suite). For a quick pass on a module: `pytest tests/test_baselines/test_skeleton_pool_model.py`.
- Lint: `./scripts/pylint.sh` (matches repo config).
- Formatting: follow existing style; avoid non-ASCII unless already present.

## Pull request guidelines
- Prefer small, focused changes; include or update tests when altering logic.
- Keep configs stable; when adding new ones, mirror the relative path patterns used elsewhere and document them.
- For decoding/refinement changes, check `tests/test_inference.py` and related decoding tests.
- When adding dependencies, place them in `requirements.txt` and keep optional/experimental imports gated.

## Documentation
- Keep `README.md` lean; detailed usage belongs in `docs/` (API, training, evaluation, baselines).
- If you add a new baseline or CLI, update `docs/evaluation.md` (and `docs/api.md` if it is user-facing).
