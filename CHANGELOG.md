# Changelog

All notable changes to Flash-ANSR are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
