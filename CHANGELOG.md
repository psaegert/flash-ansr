# Changelog

All notable changes to Flash-ANSR are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-07-01

Post-release audit round (deferred tiers C + D): fixes the two misspelled public class names, clearer
config errors, and a `convert_data` de-duplication. Re-pinned `symbolic-data>=0.10`.

### Changed
- **BREAKING: fixed two misspelled public names, no alias.** `convert_data.TestSetParaser` ->
  `TestSetParser` and `preprocessing.FlashASNRPreprocessorConfig` -> `FlashANSRPreprocessorConfig`.
  Update imports; there is no deprecated alias.
- **BREAKING (transitive): `symbolic-data>=0.10`.** `LampleChartonCatalog.load` now returns the
  catalog object only; the flash-ansr data path is updated to match.
- **`FlashANSRModel.from_config` validates required keys up front** and raises one clear `KeyError`
  naming the missing key(s) and listing the keys present, instead of an opaque bare `KeyError` from
  deep in the constructor on config drift.

### Fixed
- **`FastSRBParser`: a missing / `None` / empty `prepared` cell is counted as missing and skipped**
  instead of raising `AttributeError` (the `^`->`**` replace previously ran before the None check).

### Internal
- De-duplicated the four `convert_data` test-set parsers (SOOSE / Feynman / Nguyen / FastSRB) onto a
  shared `TestSetParser._process_expression` / `_finalize` pipeline (~130 lines removed).
  **Behavior note:** SOOSE / Feynman / Nguyen now treat an *unparseable* expression as invalid and
  skip it (counted in the invalid tally), matching FastSRB, rather than raising and aborting the whole
  conversion (fail-loud -> fail-skip). Inert on the clean curated sets; a hardening for malformed input.

## [0.9.5] - 2026-07-01

### Added
- `Refiner.all_constants_values` public read-only property (the `(constants, covariance, loss)` fit
  attempts, best-first), so downstream consumers (e.g. srbf baselines) no longer reach into the
  private `_all_constants_values`.

### Fixed
- The `convert_data` benchmark parsers guard their percentage prints against an empty test set
  (`/ max(len(test_set_df), 1)`), so converting an empty input no longer raises `ZeroDivisionError`.

## [0.9.4] - 2026-07-01

Post-release audit cleanup + two robustness fixes.

### Fixed
- **`Refiner.fit` accepts a 1-D `y`** `(n,)` (coerced to `(n, 1)`) instead of raising an opaque
  `IndexError`, matching `FlashANSR.fit`.
- **Constant-pruning variant generation is bounded.** Above a threshold of constants the exhaustive
  `2**constant_count` powerset (one tree-prune per mask) is replaced by a deterministic bounded set
  (remove-none/all, each single removal/keep), so a high-constant beam can no longer blow up the
  pruning path (reachable via `prune_constant_budget > 0`). Small expressions are unchanged.
- Clear errors for missing dataset-config keys and for `get_expression()` on an unfitted model;
  `FlashANSRDataset.save()` positional-arg forwarding fixed; `ModelFactory` no longer imports a
  non-existent module; the Feynman parser now counts `n_invalid_expressions` and a stray debug
  `print` was removed; `main(argv)` typed `Sequence[str]`; `to_dataframe` keeps `expression_infix`;
  `FlashANSRPreprocessor.format` guards empty input; docstring fixes.

## [0.9.3] - 2026-07-01

### Fixed
- **`infer()` no longer raises `ConvergenceError` when no beam converges.** It now returns an
  `InferenceResult` with empty `candidates` and the FULL candidate ledger (every generated beam
  classified `FIT_FAILED` / `INVALID`) -- honoring its documented contract exactly when the ledger is
  most useful (total-failure diagnosis). `fit()` still raises on all-fail (its read-back contract is
  unchanged); the behavior is threaded via a new internal `allow_empty` flag on `_compile_results_pure`.

## [0.9.2] - 2026-07-01

Post-release audit fixes (no API change).

### Fixed
- `FlashANSRDataset.compile()` now clones each batch out of the worker pool's shared memory
  (`persistent=True`) before the pool shuts down, fixing a use-after-free: the materialized dataset
  previously held tensors aliasing shared memory that was freed when the generator's `finally`
  triggered shutdown.
- `Refiner._fit` copies `p0` at entry, so per-restart refinement noise is no longer accumulated in
  place across restarts (each restart starts from a fresh copy of the initial guess) and a
  caller-supplied `p0` array is never mutated.

## [0.9.1] - 2026-06-30

Terminology cleanup + the training data layer fully on `symbolic_data` catalogs by name.

### Changed
- **Training data is referenced by catalog name.** Dataset/holdout configs point `source.catalog` /
  `holdout_pools` at the HF catalogs `v23-val` / `fastsrb` (not local saved skeleton-pool dirs);
  `FlashANSRDataset.from_config` resolves a name/path/inline ref via `ProblemSource` (a saved directory
  is still loaded as before). Requires `symbolic-data>=0.9` (declarative-holdout support).
- Purged the term "skeleton pool" from source + docs (`convert_data` runtime warnings/docstrings reworded
  to "catalog"); fixed the `_MOVED_TO_SRBF` redirect (→ `srbf` / `Benchmark` / `LampleChartonModel`) and
  stale module-path comments; documented the `infer()` → `InferenceResult`/`CandidateLedger` API.

### Removed
- Obsolete/broken data-generation scripts (`generate_test_set.sh`, `generate_validation_set.sh`,
  `import_test_sets.sh`) that built the now-superseded saved skeleton-pool dirs.

## [0.9.0] - 2026-06-30

This release completes the data-layer handover to `symbolic_data` and adds a first-class inference
API that returns every candidate (and the fields a benchmark records) directly, so downstream
consumers never scrape model internals.

### Added
- `FlashANSR.infer(X, y, ...) -> InferenceResult`: run symbolic regression on one problem and get the
  results back directly, without `fit()`'s instance-state mutation. An `InferenceResult` carries the
  score-sorted refined `Candidate`s plus a lean, columnar `CandidateLedger` (the FULL generation pool
  joined with the refined survivors, each classified `FIT_OK` / `FIT_FAILED` / `INVALID`). New public
  module `flash_ansr.inference` (`InferenceResult`, `Candidate`, `CandidateLedger`,
  `build_candidate_ledger`, `FIT_OK`/`FIT_FAILED`/`INVALID`).
- Each `Candidate` exposes everything a consumer records per prediction: `raw_beam`, `expression`
  (skeleton tokens), `expression_prefix` (raw substituted prefix), `expression_infix` (the
  variable-mapped infix string, identical to `get_expression(map_variables=True)`), `skeleton_prefix`,
  `constants`, `log_prob`, `score`, `fvu`, `complexity`, `constant_count`, `pruned_variant`, and
  opt-in `y_pred` / `y_pred_val` (computed for the top-k only, default best-only, to avoid OOM at high
  candidate counts).

### Changed (breaking)
- The data/training path now consumes a `symbolic_data.ProblemSource` instead of a `SkeletonPool`.
  `FlashANSRDataset(source=...)`; dataset configs use a `source:` block
  (`source: {catalog: <path|dict>, sampling: {...}}`) in place of the old `skeleton_pool:` key.
  The streaming worker builds a per-worker `ProblemSource` post-fork (each seeded from fresh entropy),
  replacing the previous `os.getpid()`-derived global seeding.
- Pool/catalog config files renamed `skeleton_pool*.yaml` -> `catalog*.yaml` (generative catalogs gain
  a `type: lample_charton` line); the bundled `configs/v23.*` dataset configs are migrated accordingly.
  Saved validation/benchmark pool directories continue to load (a saved catalog directory is read as a
  fixed-skeleton source).
- Preprocessing/conversion parameters renamed off the old term: `FlashANSRPreprocessor(catalog=...)`
  (was `skeleton_pool=`), `convert_data` `base_catalog` (was `base_skeleton_pool`).
- The prompt feature extractor takes an injected `numpy.random.Generator` (no module-global RNG).
- Requires `symbolic-data>=0.7.2`.

### Removed (breaking)
- The top-level `flash_ansr.SkeletonPool` re-export is removed. The procedural generator now lives in
  `symbolic_data` as `LampleChartonCatalog` (a `GenerativeCatalog`); import it from there
  (`from symbolic_data import LampleChartonCatalog`). `flash_ansr.NoValidSampleFoundError` is still
  re-exported from `symbolic_data`.
- The bundled `configs/evaluation/` tree is removed; evaluation/benchmarking lives in the `srbf`
  package, which consumes `FlashANSR.infer()` directly.

## [0.8.0] - 2026-06-29

### Removed (breaking)
- The deprecated `flash_ansr.expressions` shim package (introduced in 0.7.0) is removed. Import the
  expression/data layer from `symbolic_data` instead (installed via `flash-ansr[train]`, or directly).
  The top-level `flash_ansr.SkeletonPool` / `flash_ansr.NoValidSampleFoundError` re-exports are
  unchanged. No `flash_ansr` code imported the shim; it existed only for external back-compat.

## [0.7.0] - 2026-06-28

### Changed (breaking)
- The expression/data layer was carved out into the new `symbolic_data` package; `flash_ansr.expressions.*`
  became deprecation shims (removed in 0.8.0) re-exporting from `symbolic_data`. `symbolic-data` is now a
  runtime dependency. The standalone data CLI (`import-data` / pool create/split) was dropped; model
  commands are unchanged.

## [0.6.1] - 2026-06-27

A small maintenance release.

### Changed
- Default Weights & Biases logging mode for training is now `disabled` (CLI `--mode` and
  `Trainer.train(..., wandb_mode=...)`). Training works out of the box without a W&B account or
  network; pass `--mode online` (or `wandb_mode="online"`) to enable logging.

### Internal
- Moved the `simplify="sympy"` timeout helper to a dependency-light leaf module
  (`flash_ansr.utils.sympy_timeout`); the model and skeleton pool now import it from there. No
  behaviour change; this decouples the helper from the data/sampling module ahead of a future
  package split.
- The `simplify="sympy"` path now raises a clear, actionable `ImportError` (pointing at
  `pip install flash-ansr[sympy]`) if `sympy` is ever unavailable. In practice `sympy` ships as a
  transitive dependency of `torch`, so this is defensive only.

## [0.6.0] - 2026-06-26

A scope-focused release: the evaluation framework, comparison baselines, and benchmarks are split
out into a standalone companion package, [**srbf**](https://github.com/psaegert/srbf) (Symbolic
Regression Benchmark Framework). `flash-ansr` is now the lean product: load a pretrained model,
`fit(X, y)`, get an expression, or train your own. Use `srbf` for systematic benchmarking and to
evaluate models beyond Flash-ANSR.

### Breaking Changes
- **Evaluation and baselines moved to `srbf`.** Install with `pip install srbf`. The following are no
  longer importable from `flash_ansr` (a helpful redirect error points to srbf):
  - `flash_ansr.Evaluation` -> `from srbf.eval import Evaluation`
  - `flash_ansr.SkeletonPoolModel`, `flash_ansr.BruteForceModel` -> `from srbf.baselines import ...`
  - the `flash_ansr.eval`, `flash_ansr.baselines`, and `flash_ansr.benchmarks` modules, and the
    NeSymReS adapter `flash_ansr.compat.nesymres`.
- **CLI:** the `flash_ansr evaluate-run` subcommand moved to `srbf`. All other subcommands stay
  (`train`, `install`, `remove`, `generate-/filter-/split-skeleton-pool`, `import-data`,
  `find-simplifications`, `benchmark`, `wandb-stats`).

### Removed
- Eval-only dependencies `editdistance` and `zss` are no longer required by the core package (they
  move with `srbf`).
- `THIRD_PARTY_LICENSES` (NeSymReS/FastSRB notices) moves to `srbf`; flash-ansr core vendors no
  third-party code.

### Added
- Optional classifier-free guidance for optional-condition models: `guidance_weight` on
  `FlashANSRModel.sample_top_kp` (`uncond + w * (cond - uncond)`). Inert by default
  (`guidance_weight=None`/`1.0` is byte-identical to the standard decode path).
- A public-API contract test (`tests/test_public_api_contract.py`) freezes the surface `srbf`
  consumes, so a contract break cannot merge unnoticed.

### Changed
- The optional `[baselines]` extra (sympy, for the moved baseline adapters) is replaced by a
  `[sympy]` extra that enables only the optional `simplify="sympy"` simplification backend. The
  product default simplifies via `simplipy` and needs no sympy.

## [0.5.0] - 2026-06-24

A performance-focused release: substantial inference-time speedups with quality-neutral defaults,
plus new architecture and refinement options.

### Highlights
- Significant inference-time speedups from KV-cache decoding, static (position-indexed) decoding,
  candidate-budget-adaptive batching, and an optional overlapped evaluation pipeline; all designed
  to be quality-neutral.
- New defaults trade a small amount of compute for better recovery out of the box (larger candidate
  budget, automatic batch sizing); see Changed Defaults to reproduce 0.4.x behavior.

### Breaking Changes
- Renamed the `parsimony` keyword to `length_penalty` across the public API
  (`FlashANSR(...)`, `.load(...)`, `.fit(...)`, `.compile_results(...)`). There is no compatibility
  alias: update `parsimony=` call sites to `length_penalty=`.

### Changed Defaults
*(Upgrading without code changes can produce different predictions/scores than 0.4.5. To reproduce
0.4.x behavior, set the values in parentheses.)*
- KV cache enabled by default during decoding (`use_cache=False`).
- Automatic batch sizing for candidate generation (`batch_size=128`).
- Static decoding auto-enabled where applicable (`static_decode=False`).
- Increased default candidate budget in softmax sampling (previous smaller `choices`).
- Experimental constant pruning available in the inference path (`prune_constant_budget=0`).

### Added
- Exclusive Self-Attention (XSA) architecture option, supported under static decoding (bit-identity verified).
- KV-cache and position-indexed static-decode forward path (bit-exact to the dynamic path).
- `OverlappedEvaluationEngine`: an opt-in, quality-neutral cross-problem pipeline that overlaps
  simplification and refinement, with a persistent pre-CUDA refine/simplify pool.
- Constant-refinement initialization options: `cauchy` and `magspan` `p0_noise` inits.
- B1/B2/B4 ablation switches and accompanying configs/tests.
- Option to compute Fisher and Hessian matrices during data generation.
- `length_penalty`, `constants_penalty`, and `likelihood_penalty` recorded in evaluation metadata.
- Provenance metadata and atomic writes for evaluation results.

### Performance
- KV-cache decoding, c-adaptive batching, parallel post-generation simplification, and the overlapped
  evaluation engine reduce end-to-end inference time, validated to be quality-neutral.

### Fixed
- Beam search: correct EOS handling, prevent score mixing on the active-beam fallback, robust to any
  `max_len`, and accurate completion flags with bulk GPU→CPU transfer.
- Guard the FVU computation against finite-divergent overflow (false perfect-fit over-count).
- Improved out-of-vocabulary handling for sympy-based encoding/inference.
- Apply evaluation settings that were previously not propagated to evaluation runs.
- Constant-pruning log-probability rescoring comparability fix.

### Dependencies
- Require `simplipy>=0.3.0` (Rust rewrite; prefix serialization now groups chained `+` left-associatively).
- Declare previously-implicit runtime dependencies: `huggingface_hub`, `sympy` (and lower-bound floors on
  `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`).
- Drop unused dependencies from the core install (`absl-py`, `einops`, `schedulefree`); `drawdata` and
  `matplotlib` are now demo-only (installed from within the demo notebook).

[0.5.0]: https://github.com/psaegert/flash-ansr/releases/tag/v0.5.0
