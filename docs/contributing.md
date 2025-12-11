# Contributing

## Repo layout
- Code: `src/flash_ansr/`
- Tests: `tests/`
- Configs: `configs/`
- Scripts: `scripts/` (training/eval helpers)

## Environment
- Use the `flash-ansr` conda env; install extras from `requirements.txt`.
- For GPU work, confirm CUDA is available before running tests.
- Large assets (models/data) are pulled via scripts or `flash_ansr install`.

## Quality checks
- Tests: `./scripts/pytest.sh` (full suite). For a fast spot check: `pytest tests/test_baselines/test_skeleton_pool_model.py`.
- Lint: `./scripts/pylint.sh` (respects repo config).
- Formatting: follow existing style; stick to ASCII unless a file already uses Unicode.

## PR guidelines
- Prefer small, focused changes with tests when logic changes.
- Keep configs stable; when adding new ones, mirror relative paths used elsewhere and document them.
- For decoding/refinement changes, run `tests/test_inference.py` and related decoding tests.
- New dependencies go to `requirements.txt`; gate experimental imports.
- Update docs when adding baselines, CLIs, or config fields.

## Style and docs
- Add succinct comments only for non-obvious logic; keep tensors float32 unless a precision module requires otherwise.
- When adding dataloader fields, update `FlashANSRDataset.collate` to pad/stack consistently and document it.
- Keep `README.md` lean; detailed usage belongs in `docs/` (API, training, evaluation, baselines).

## Releasing
- Ensure checkpoints include `model.yaml`, `tokenizer.yaml`, and `state_dict.pt`.
- Sanity-check bundles by loading them via `flash_ansr install <repo>` in a clean environment.
