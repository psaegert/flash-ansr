# Changelog

All notable changes to Flash-ANSR are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`refiner_scope`: which literals the refiner may move.** The predict-vs-refine doctrine
  (owner ruling 2026-09-02): the model PREDICTS the typed literals -- `pow` exponents and
  `rootn` indices, whose value fixes the expression's domain rather than its magnitude -- and
  the refiner fits the rest. `Refiner.fit(..., refine_scope=)`, `FlashANSR(refiner_scope=)` /
  `FlashANSR.load(refiner_scope=)` accept `'fittable'` (default: every `<constant>` slot plus
  every spelled literal simplipy's `mask_fittable` policy would abstract; typed literals stay
  verbatim), `'placeholders'` (only the slots; every spelled literal is compiled in) and
  `'all'` (every literal, typed ones included). `flash_ansr.refine.refinement_slots` is the
  one slot definition the refiner and the T11 verbatim seeding share, so `p0` stays aligned
  by construction and now seeds every freed spelling (`-2`, `2.5`, `3/2`), not only digit-only
  tokens.

### Changed
- **The default `emission` is `'fittable'`** (`FlashANSR.fit` / `infer` / `predict`; was
  `'constants'`): the application mode -- the model spells the typed literals and leaves every
  fittable constant as a placeholder, which the refiner fits from random inits. Pass
  `emission='constants'` for the unflagged training format.
- **The refiner no longer frees `pow` exponents by default.** It used simplipy's deprecated
  digit-only conversion, which turned `pow x1 2` into `pow x1 C_0` and then fitted a real
  exponent from a random init -- `nan` on half the axis, the mechanism behind the measured
  pow-family failures (oracle recovery on fastsrb 24% with exponents freed vs 77% with the
  literals fixed). Pass `refine_scope='all'` for the old behaviour.

### Changed
- **Streaming workers are spawned, not forked.** A trainer has CUDA initialised in the
  parent by the time it opens a stream, and forking a CUDA process is undefined — the
  child inherits driver state it never initialised. The pool now uses an explicit spawn
  context. Everything a worker needs is passed explicitly and picklably: shared memory is
  attached by name, and the source config is rebuilt into the worker's own `ProblemSource`.

  Two consequences for callers. A **script** that opens a pool now needs an
  `if __name__ == "__main__":` guard, because the child re-imports the main module (the
  `flash_ansr` console entry point already has one). And pool creation costs a fresh
  interpreter per worker: measured at 8 workers, start-up plus first batch goes from
  0.87 s to 7.7 s. Steady-state throughput is unchanged — median ms/batch by quarter over
  480 batches is 7.6 / 7.9 / 7.8 / 6.6 forked against 7.6 / 7.3 / 8.0 / 5.6 spawned — so
  the cost is one-off per pool, which is also what `iterate(keep_alive=True)` exists to
  amortise.

- **Placeholding is naive and positional.** Masking a constant no longer asks the engine to
  re-derive the expression and compare: each placeheld literal site simply becomes a
  `<constant>` the model predicts, and that site's value is the block's ground truth. A
  structurally spelled rational contributes one slot per literal, so `3 / 2` masks to
  `<constant> / <constant>` and trains as two predictions. The collection-stability check and
  its `n_collection_restructured` counter are gone, along with `mask_selected_sites` and
  `nonspecial_site_positions`. `<predict_constants>` now follows the configured priors.
  Measured over 15,360 instances of the v25 prior: 76.6% of flagged instances carry at
  least one placeholder (the rest have no eligible slot -- a literal-free expression under
  `mask_all`, no fittable slot under `mask_fittable`), and those emit a block 43.6% of the
  time against a configured 0.5, the difference being the 10% of instances `condition_dropout`
  leaves unconditioned, where the block is skipped by design.
- **The worker pool can outlive one `iterate()`.** `iterate(keep_alive=True)` leaves a fully
  drained stream's pool running so the next identical call reuses it; the caller then owns
  the pool and must `shutdown()`. Validation uses it, which is where it matters — every pass
  used to cold-start and make each worker re-parse the holdout catalogs. An abandoned
  generator still shuts down (its jobs are in flight), and a live pool now raises rather than
  silently serving a request built for different settings.
- **Outlier AUROC/AUPRC are logged on an interval**, `outlier_metrics_interval` (default 50).
  They are rank statistics over every micro-batch's scores, so producing them every step cost
  a device sync per micro-step plus two sorts. The outlier loss is still logged every step.

### Fixed
- The `expression/anchor` split excluded complexity, `predict_y` and mask circumstances but
  not `predict_residual`, so prefix-placed residual rows entered a baseline the docstring
  requires be shaped like the base task.
- `ce_split` had no entry for task segment 4: `predict_residual` tokens were supervised but
  never reported. It now carries the same conditional/unconditional and masked-context
  curves as `predict_y`.

- **Constants are IEEE-754 binary64, spelled as 8 byte tokens.** A serialized constant is
  `<ieee754>` + 8 tokens from the 256-symbol `<b00>`..`<bff>` alphabet + `</ieee754>` — 10
  tokens, the same span width as before, over a wider alphabet and at double precision. The
  serializer no longer narrows a fitted constant, and no longer refuses a finite value for
  exceeding a narrower format's range. Measured on generated data, 46.5% of constants are
  values binary32 could not represent.

  `tokenizer.yaml` declares its `constants_format`, and `Tokenizer.from_config` refuses a
  vocabulary built for a different one, naming it. A vocabulary that declares nothing is
  checked by its alphabet, so a configuration stored alongside an older checkpoint is
  recognised rather than failing one out-of-vocabulary token at a time.

- **The numeric width has one name.** Every numeric surface — encoder input, numeric channel,
  constants — takes its dtype from `flash_ansr.utils.numeric.NUMERIC_DTYPE`.

- **A number's spelling follows its producer.** What the model predicts is spelled in IEEE-754
  bytes: expression constants, `<predict_y>`'s target, `<predict_constants>`' values. What the
  caller supplies is a compact `<float>` carrying its value on the numeric channel:
  `<predict_y>`'s coordinates, a stated complexity. `<float>` is forbidden at every generation
  position and never appears inside an expression.

- **`<hypothesize>` marks the boundary between what is given and what is generated.** Properties
  before it are stated by the caller, compact, and may not be restated after it; everything
  after it is the model's own and spelled in bytes. At inference the prompt ends at
  `<hypothesize>` when present and at `<expression>` otherwise, so generation always begins at
  the last prompt token. `mu`, `hypothesize` and `mask` compose in `complexity_prefix`.
  Query/answer blocks are exempt: inside them the loss mask decides, so caller-supplied
  coordinates stay compact wherever the block sits.

### Added
- **`<predict_residual>`, the displacement block.** Given a point, the model reports how far
  the observation there sits off the law: `<predict_residual> <point> <float>*dims </point>
  <ieee754> 8 bytes </ieee754> </predict_residual>`. The coordinates are caller-supplied and
  compact; the displacement is predicted and therefore a byte span. Enabled by
  `residual_block` in the dataset configuration, which requires a source with a noise
  mixture — without one the observed targets are the clean ones and every answer is zero.

  It differs from `<predict_y>` in three ways that matter. The queried point stays IN the
  encoder's support, because the observation reaches the model only through the encoder.
  The block is dropped on an unconditioned instance rather than moved, because a nulled
  memory puts the observation out of reach in either placement. And it draws its point from
  what `<predict_y>` left, so the two never query the same one.

### Removed
- **Beam search and MCTS.** `SoftmaxSamplingConfig` is the generation configuration;
  `create_generation_config` rejects any other method by name.
- **In-decode span compaction**, along with the per-row decode position it required.
- **The per-point residual head** and `predict_residuals()`.


### Added
- **Model weights are safetensors.** `model.safetensors` replaces `state_dict.pt` everywhere weights
  are written or read — `FlashANSRModel.save`/`load`, the set-encoder, `FlashANSR.load`, and the
  trainer's resume path (which delegates its write to `model.save`). Weights are the artifact people
  download, so they are stored in the format the ecosystem reads: a flat tensor map with a JSON
  header, no pickle, memory-mappable, framework-agnostic. Optimiser, scaler, scheduler and
  `training_state` stay on `torch.save`: they are not all tensors (param groups, step counters,
  Python scalars) and they never leave the machine that wrote them.
- **`flash_ansr convert-weights <dir>...`** writes `model.safetensors` beside an existing
  `state_dict.pt`, leaving the pickle exactly where it is. Legacy pickles are no longer READ — one
  format, one code path — so a checkpoint that predates this must be converted once. The
  `FileNotFoundError` names the command when it finds an unconverted `state_dict.pt`. The converter
  refuses a state dict carrying non-tensor entries rather than dropping them silently.
- **`conditioned=` on every decoder verb, and `predict_y(expression=...)`: the trained
  circumstances that had no entry point.** `condition_dropout: 0.1` routes one training instance in
  ten to the learned `null_memory`, and `predict_y_block.p_conditional: 0.5` writes the block in two
  placements — before `<expression>` (data alone) or after it (data AND expression). Neither knob
  was reachable: `fit`/`infer`/`predict_*` always conditioned, and `predict_y` only ever emitted the
  prefix placement. Now `fit`, `infer`, `predict_y`, `predict_constants` and `predict_complexity`
  all take `conditioned=` (default `True`), and `predict_y` takes `expression=`. `X`/`y` may be
  `None` whenever `conditioned=False`. On a checkpoint without `optional_condition` the knob raises
  `CapabilityUnavailable` rather than silently conditioning. `score_outliers` deliberately has no
  such knob: the head reads the encoder directly and has no null path.
  `fit(conditioned=False)` is the "propose from the prior, fit to the data" arm — the data still
  selects the winner, it just does not shape the proposals.
- **Forbidden non-finite token guard on the simplification path.** `float("inf")` / `float("-inf")` /
  `float("nan")` are encodable vocabulary tokens (ids 25/26/27), and SimpliPy folds a degenerate
  sub-expression to one instead of failing (`['/', 'x1', '-', 'x2', 'x2']` -> `['*', 'float("inf")',
  'x1']`, `is_valid` True), so such skeletons re-entered the candidate stream as valid predictions.
  `flash_ansr.utils.skeleton.simplify_and_mask` — the one seam every candidate producer shares —
  now raises `NonFiniteExpressionError`, and each producer (beam search, softmax post-processing,
  MCTS canonicalization, the constant-pruning lane, the forked simplify pool, dataset conversion)
  DROPS the candidate and counts it: `flash_ansr.utils.non_finite_drops()` /
  `reset_non_finite_drops()` expose the tally. The token set matches symbolic-data's generator-side
  forbidden list (`float("inf")`, `float("-inf")`, `float("nan")`, `zoo`, `nan`, `oo`). On the
  training-data INGEST direction (`mask_literals_positional`) it does NOT drop but propagates:
  symbolic-data rejects these before yielding, so one arriving means a broken producer contract,
  and skipping it would silently reshape the training distribution.
- **`encoder_mask_query_norms` model flag (default `False`)**: threads the support-set padding mask
  into the ISAB self-refinement blocks' query/residual-stream SetNorms (`norm_q`/`norm_ffn`) and zeroes
  sub-layer outputs on padded query rows. Legacy (default) behavior computes those shared set statistics
  over the zero-padding too, which understates the set RMS by `sqrt(n_valid/set_len)` — inflating valid
  rows by up to 32x for a 1-point support set padded to 1024 during training — and, for small sets, lets
  projection-bias "garbage" rows dominate the `norm_ffn` statistic. With the flag enabled, a sample's
  encoding is invariant to padding length (tested). Existing checkpoints were trained with the legacy
  semantics and load/run bit-identically under the default.
- **`sanitize_input_num` model flag (default `False`)**: zeroes the numeric-token bit encodings at
  positions whose `input_num` is NaN (no numeric payload). The previous guard checked `isnan` on the
  IEEE-754 *bit encodings* — which are ±1 for any input, including NaN — and therefore never fired
  (dead code, removed); non-constant positions received a learned NaN-bit-pattern embedding instead of
  zero. Checkpoints trained before this flag (v23- and v24-era alike) keep the legacy behavior
  under the default. Under the mixed constants representation (0.13.0), `input_num` is NaN at
  every non-payload position by design, so enabling this flag is recommended for models trained
  from now on.

### Fixed
- **Unconditioned instances no longer lose their `<predict_y>` block.** The gate excluded the block
  from every condition-dropout instance, on the reasoning that "predicting y* with a nulled memory
  is a nonsense task". That holds for the PREFIX placement — nulled memory and no expression is
  nothing to condition on — but not for the suffix: with the expression in scope it is FUNCTION
  EVALUATION, a well-posed task that grounds the expression tokens semantically. The block is now
  pinned to the suffix on unconditioned instances instead of being dropped (owner ruling
  2026-08-26). Takes effect from the next training run; T16 was trained under the old gate, so
  `predict_y(expression=..., conditioned=False)` on a T16 checkpoint queries an untrained
  circumstance.
- **Raw batches held past their turn were a use-after-free, and it segfaulted.** The tensors
  `FlashANSRDataset.iterate` yields VIEW the streaming pool's shared-memory ring and are valid only
  until the pool refills that block. `Trainer` kept two of them by reference — `first_raw_batch` in
  `_validate_step` and `_last_raw_train_batch` in `_train_step` — for the T12 paired constant-span
  eval, then read them after the loop. That dereferences unmapped memory: a hard `SIGSEGV`,
  reproducible in twenty lines (keep batch 0, iterate twice, read `batch["x_tensors"][0]`), not an
  exception anything could catch. The paired eval now takes a copy at capture (`_detach_raw_batch`),
  and `tests/test_data/test_raw_batch_lifetime.py` pins the contract. The crash was invisible
  because the eval is gated on `constant_representation == 'ieee754_mixed'` and that was not the
  default; making it the default made the segfault deterministic.
- **Encoder padding masks are coerced to `bool` (with a warning) at the `SetTransformer` entry.**
  `scaled_dot_product_attention` interprets float masks as *additive logit biases*, so a float 0/1
  padding mask silently masked nothing. The standard pipeline (`FlashANSRDataset.collate`) already
  passes bool masks and is unaffected; direct callers passing float masks were silently unprotected.
- **Trainer raises on gradient-accumulation remainder.** `_train_step` split micro-batches with an
  integer division that silently dropped `batch_size % gradient_accumulation_steps` samples from every
  step; it now fails loudly. No shipped config was affected (all use `gradient_accumulation_steps=1`).

### Removed
- **v23 is gone from this line entirely (owner ruling, 2026-08-26): one generation, one code path.**
  `constant_representation` had `'v23'` as its DEFAULT in the data layer, so a training config that
  omitted the key silently produced v23 data — the ieee754 format the whole v24 line rests on was
  opt-in. `'ieee754_mixed'` is now the only legal value and the default; a tokenizer without the span
  tokens is refused at dataset construction. The dead conditionals that guarded it are gone with it:
  the byte-identical passthrough in `serialize_constant_tokens`, the four `!= 'ieee754_mixed'`
  preconditions in `FlashANSRDataset.__init__`, the `mixed_constants` branches in the streaming
  worker, and the unreachable `if IEEE754_START_TOKEN in self.tokenizer` span-mapping branch in
  `_fit_refine` (`_validate_checkpoint` has refused such a vocabulary at load since it was added).
- **The v23 model bundles and their register are deleted**: `configs/v23.0-*`, `configs/v23.2-120M`
  and `configs/VERSIONS.md` — 177 files. They pinned the retired generation-1 `dev_7-3` engine and
  produced checkpoints this line cannot load; `git` keeps them for anyone who needs the recipes under
  a pinned `flash-ansr<0.13`. `configs/v24-template/tokenizer.yaml` drops the eight tokens no v24 run
  ever trained on (`<prompt>`, `</prompt>`, and the six `<allowed_term>` / `<include_term>` /
  `<exclude_term>` delimiters). Trained-run tokenizers (`v24.0-T13`..`T16`) are untouched: their
  vocabularies are pinned by their checkpoints.
- **The `simplify='sympy'` simplification path is gone (owner ruling, 2026-08-18).** SymPy
  simplification was an *ablation* of the product simplifier (SimpliPy), and the standing rule is that
  production code stays clean and minimal: experiments and ablations branch off it or patch it, they
  never live inside it. Removed: the `simplify == 'sympy'` branch in `FlashANSRModel._postprocess_sampled`,
  the `flash_ansr.utils.sympy_timeout` helper module, and the `[sympy]` optional-dependency extra.
  `simplify` is now a two-state `bool` (`True` = SimpliPy, the product default; `False` = no
  simplification) everywhere it is accepted.
- **A config that asks for the removed path FAILS LOUDLY; it never falls back.** `simplify='sympy'` is
  refused by `SoftmaxSamplingConfig` / `create_generation_config` and by `sample_top_kp` /
  `_sample_top_kp_static` / `_postprocess_sampled` with a `ValueError` naming the removal. Silently
  re-serving such a config with SimpliPy would swap the canonicalizer a config explicitly named — a
  behaviour change wearing a removal's clothes — so the request is refused instead.
- **Catalog configs requesting the SymPy skeleton path are refused at the flash-ansr boundary.**
  `simplify` in a *catalog* config is symbolic-data's parameter, which flash-ansr only passes through;
  symbolic-data keeps its own `simplify='sympy'` path and is unchanged. `FlashANSRDataset.from_config`
  now raises rather than build a data source that routes into it.

## [0.13.0] - 2026-08-18

Compatibility release for the simplipy 0.13 line, plus the numeric-constants foundations for
the next model generation. All new decoding features default off; existing configs produce
byte-identical behavior.

### Changed
- **simplipy 0.13 lockstep.** Requirements pin `simplipy>=0.13.0,<0.14`; test engine bundles use
  generation-2 simplipy assets; mask handling ported to the simplipy 0.13 API; symbolic-data
  `>=0.14` contract fix.

### Added
- **Per-constant mixed serialization** (`constant_representation` config gate): numeric constants
  can serialize as a `<float>` summary token or an expanded `<ieee754>` hex-nibble span, mixed
  50/50 per constant; universal loss mask for `<float>`-target positions.
- **Constrained decoding** (`constrain_ieee754`, default off): a decode-time grammar mask at the
  sampling, beam, and static logit sites guaranteeing every opened `<ieee754>` span emits exactly
  8 hex nibbles and closes within the length budget.
- **v24.0 target format** (owner ruling 2026-08-18). Expanded constants are HEX NIBBLE spans:
  `<ieee754>` + 8 tokens over the 16-symbol `<h0>`..`<hf>` alphabet + `</ieee754>` = **10 tokens**
  (was 34 with the now-retired `<b0>`/`<b1>` bit tokens). Same float32 value semantics, 4x fewer
  autoregressive steps per constant; nibble order is big-endian (most-significant first). A new
  `configs/v24-template/tokenizer.yaml` pins the v24 target vocabulary: simplipy's tagged canonical
  dialect (`<add> </add> <mul> </mul> <sub> <div>`), the generation-2 23-operator set with no
  generation-1 sugar, the constants format, and **no explicit number tokens at all** (the integers
  -10..10 are retired; `np.pi`/`np.e` stay as symbolic constants). v23 configs are untouched and
  v23-era behavior is byte-identical.
- **KV-span compaction**: closed `<ieee754>` spans compact out of the dynamic KV cache with
  verified equivalence to the fresh forward (atol 1e-5), re-encoding the collapsed `<float>`
  position.
- **`condition_dropout` config key** (default 0) for unconditional-prediction training instances.

### Removed
- **v23-era model support.** flash-ansr 0.12.x remains the supported pairing for v23 models
  (`pip install "flash-ansr<0.13"`); this line targets the next model generation. Tests requiring
  a published model are skipped until new checkpoints exist.

## [0.12.1] - 2026-08-17

- Cap `simplipy>=0.10,<0.12`: simplipy 0.12.0 deletes `SimpliPyEngine.mask` (six call sites here), refuses the generation-1 `dev_7-3` engine the v23 models pin, and silently flips the `explicit_constant_placeholders` default -- 0.12.x of flash-ansr only ever worked against simplipy 0.10/0.11.

## [0.12.0] - 2026-07-26

Compatibility release for simplipy 0.10 (the certificate-algebra engine) and symbolic-data 0.13,
plus a fit-boundary correctness fix. Skeleton canonicalization changes with the new engine
semantics, so candidate keys and selection outcomes can shift slightly vs 0.11.0 -- do not pool
0.11-era result pickles with 0.12-era ones.

### Changed
- **simplipy >= 0.10 lockstep.** Masking (numeric literals -> `<constant>`) is a separate,
  terminal `engine.mask()` step since simplipy 0.9 -- all six deployed canonicalization sites now
  call `mask(simplify(...))` explicitly, and the removed `max_pattern_length` keyword is gone
  from every call (rule application is always unrestricted). Requirements pin `simplipy>=0.10`
  and `symbolic-data>=0.13` (older flash-ansr versions break against simplipy >= 0.10: the
  removed keyword raises `TypeError` in the decode path).

### Fixed
- **Refiner scores on the fitted domain (R1).** Candidate scoring masked non-finite rows on `y`
  only; a non-finite `X` row is outside the domain the simplification rules are certified on and
  now excludes the row at the fit boundary as well (no-op on finite data).

## [0.11.0] - 2026-07-10

Research-to-production upstream: corrected candidate scoring (scale-invariant FVU + ranking fix) and the
rewritten Monte-Carlo tree-search decoder (batched + asynchronous overlapped refinement). Softmax sampling
remains the recommended decode method; benchmarked at deployed scale, MCTS matches its recovery but costs
more wall-clock and refiner calls -- it ships as a correct, fully-supported alternative.

### Changed
- **BREAKING (scoring semantics): scale-invariant `compute_fvu`.** The absolute `FLOAT64_EPS` variance
  floor is gone: FVU = `loss / variance` with explicit edge cases (`sample_count <= 1` -> raw loss;
  non-finite loss/variance -> `+inf`; zero variance -> `0.0` iff the residual is exactly zero else
  `+inf`). The old floor spuriously rated ANY candidate near-perfect on tiny-magnitude targets (the
  constant-candidate mis-selection bug). `normalize_variance` is DEPRECATED (retained for imports only).
- **BREAKING (ranking): `score_from_fvu` treats a non-finite or negative FVU as the WORST score
  (`+inf`), not the best.** Previously a diverged/invalid candidate mapped to the floor (best finite
  score) and could out-rank real fits. A genuine perfect fit (`fvu == 0.0`) still gets the best finite
  score via the floor.
- **Selection variance now uses `ddof=0`** (`y.var(unbiased=False)`), matching the evaluation-side FVU
  definition so selection and evaluation agree.
- **MCTS decoder rewritten** (`decoding/mcts.py`, `generation/mcts.py`): value-guided best-first search
  (max-backup PUCT) whose value function is a full constant-refinement per distinct canonical candidate;
  refine-budget-driven stopping; canonical (simplify+constantify) dedup shared with the refiner cache so
  each candidate is refined exactly once (search fits are reused at deploy time). `invalid_penalty`
  default corrected `1e6 -> 1.0` (the old magnitude poisoned mean-backup).
- **MCTS batched + asynchronous execution:** leaf-parallel batched policy/rollout forwards
  (`batch_width`), and an opt-in overlapped event loop (`async_search`, `inflight`, `gpu_batch`) that
  runs GPU generation concurrently with pool refinement via non-blocking futures.

### Added
- `RecoverableForkPool.submit()`: non-blocking single-job submission returning a `Future` (pure
  pass-through; never re-forks -- the async decode path's transport).
- `MCTSGenerationConfig`: `refine_budget`, `max_rollouts`, `batch_width`, `async_search`, `inflight`,
  `gpu_batch`, `backup`, `value_objective`, `fpu_reduction`, `renormalize_prior`,
  `reward_log_fvu_hi/lo`, `rollout_resample_retries` (validated; full `to_kwargs` round-trip).
- High-constant beams: candidate pruning falls back to a bounded deterministic mask set above
  12 constants instead of the exhaustive `2**n` powerset (from 0.10.x-era production hardening).
- Tests: `test_mcts_async.py` (byte-identity of the async loop vs the synchronous search at
  `inflight=1`, budget exactness, virtual-loss accounting, order-invariance); scoring regression tests
  for the scale-invariance + ranking fixes; MCTS config-surface contract tests.

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
  **Behavior note:** all four parsers are now **fail-loud on an unparseable expression by default** --
  a malformed input raises and aborts the import (a data problem in a curated set is worth surfacing).
  Pass `parse_data(..., skip_unparseable=True)` for the lenient mode that counts + skips malformed rows
  (e.g. a known-noisy external benchmark file). This unifies the previously-divergent behavior
  (FastSRB used to skip parse errors silently; it now also fails loud by default). Engine-invalid
  (parsed but not representable) and too-many-variable expressions remain the designed, *reported*
  count + skip filters -- they are not errors.

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
  expression/data layer from `symbolic_data` instead (installed automatically as a core dependency of
  flash-ansr, or directly).
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
  (`flash_ansr.utils.sympy_timeout`); the model and the data/sampling module now import it from there. No
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
  - `flash_ansr.Evaluation` -> `from srbf import Benchmark` (the evaluation engine was subsequently
    renamed to the top-level `srbf.Benchmark`)
  - `flash_ansr.SkeletonPoolModel`, `flash_ansr.BruteForceModel` -> `from srbf.baselines import ...`
  - the `flash_ansr.eval`, `flash_ansr.baselines`, and `flash_ansr.benchmarks` modules, and the
    NeSymReS adapter `flash_ansr.compat.nesymres`.
- **CLI:** the `flash_ansr evaluate-run` subcommand moved to `srbf`. All other subcommands stay
  (`train`, `install`, `remove`, `generate-/filter-/split-skeleton-pool`, `import-data`,
  `find-simplifications`, `benchmark`, `wandb-stats`). *(The standalone data CLI, including the
  `generate-`/`filter-`/`split-skeleton-pool` and `import-data` commands, was later removed in 0.7.0;
  the current CLI exposes only `train`, `install`, `remove`, `benchmark`, and `wandb-stats`.)*

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
