# Changelog

All notable changes to Flash-ANSR are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **AdaMuon** (`optimizer: {name: AdaMuon}` in train.yaml; Si, Zhang, Shen 2025, arXiv:2507.11005).
  The hidden weight matrices take the sign-stabilized, orthogonalized, variance-normalized and
  RMS-aligned update; embeddings, the heads' final projections and every bias or norm gain stay
  on AdamW, all inside one optimizer so the schedule, checkpoints and resume are unchanged. The
  model declares its parameter roles (`FlashANSRModel.parameter_roles`); an unplaceable
  parameter is an error. The RMS alignment makes the AdamW learning-rate schedule the right
  schedule for the Muon groups too. `kwargs`: `lr`, `weight_decay` (Muon groups, 0.1),
  `adam_weight_decay` (embeddings and projections, 0.01), `momentum`, `nesterov`, `ns_steps`,
  `eps`, `betas`, `adam_eps`.
- **z-loss is back as a documented loss term** (`z_loss_weight`, PaLM's log² Z in fp32; 0.0 off).
  The v25.0-T7-20M run showed that `head_pre_logits_norm` moves the loss-flat logit-offset drift
  into the norm's own gain and bias instead of removing it (mean |logit| 34 → 1,881, the bf16 head
  then costs up to 74 % extra loss); the z-loss gives that direction a restoring gradient.
- **Logit-scale metrics** every step: `train_log_z`, `train_logit_absmean`, `head_row_norm_max`
  and their validation twins, the early alarm for the drift above (it is visible hundreds of
  thousands of steps before any loss moves).
- **`configs/v25.0-T8-20M`**: the T7-20M recipe with AdaMuon, `z_loss_weight: 1e-4` and no
  pre-logits LayerNorm.

## [0.14.0] - 2026-09-05

The first release of the third model generation. Constants are binary64 values spelled as byte
tokens, weights are safetensors, candidate ranking has three modes with MDL as the default, and
every circumstance a model is trained on has a public verb. The reference checkpoint is
[`psaegert/flash-ansr-v25.0-T7-3M`](https://huggingface.co/psaegert/flash-ansr-v25.0-T7-3M).
Generation-1 (v23) checkpoints and configs are not served by this line; `pip install "flash-ansr<0.13"`
remains their pairing. Requires `simplipy>=0.14.6,<0.15` and `symbolic-data>=0.18,<0.19`.

### Added
- **One verb per trained circumstance.** `fit` and `infer` are the fitting verbs. `predict_y`
  interpolates a point set (or, with `expression=`, evaluates the given expression at the query
  points), `predict_complexity` asks the model for its own complexity hypothesis under the
  `<hypothesize>` licence and records whether it would have opened the block unprompted,
  `predict_constants` samples the constants of an expression from the model's posterior (the
  `<predict_constants>` infilling block, restricted to the byte alphabet as in training, with
  per-byte log-probabilities and a count of steps whose unrestricted argmax would have left it),
  and `score_outliers` returns raw per-point outlier probabilities from the encoder head. Every
  verb refuses at call time, before the encoder runs, when the checkpoint lacks the block it needs,
  and `v1..vn` variable aliases resolve once at the boundary.
- **`conditioned=` on every decoder verb, and `predict_y(expression=...)`.** `condition_dropout`
  routes a share of training instances to the learned `null_memory`, and the `<predict_y>` block
  is written in two placements, before `<expression>` (data alone) or after it (data and
  expression). `fit`, `infer`, `predict_y`, `predict_constants` and `predict_complexity` take
  `conditioned=` (default `True`); `predict_y` takes `expression=`; `X`/`y` may be `None`
  whenever `conditioned=False`. On a checkpoint without `optional_condition` the knob raises
  `CapabilityUnavailable` rather than silently conditioning. `score_outliers` has no such knob:
  the head reads the encoder directly and has no null path. `fit(conditioned=False)` proposes
  from the prior and fits to the data; the data still selects the winner, it just does not shape
  the proposals.
- **`emission=` on `fit` / `infer` / `predict`**: `'fittable'` (default) has the model spell the
  typed literals and leave every fittable constant as a placeholder for the refiner;
  `'constants'` is the unflagged training format in which the model predicts every constant.
- **`complexity=` conditioning speaks the trained grammar.** The `<complexity> <float>
  </complexity>` block is emitted bare as a prefix element with mu on the numeric channel, exactly
  as training wrote it. `Candidate.mu` reports simplipy's complexity of the skeleton (the unit the
  prompt consumes), computed over the refined survivors and `None` when the engine will not price
  a dialect.
- **`refiner_scope`: which literals the refiner may move.** The model predicts the typed
  literals, `pow` exponents and `rootn` indices, whose value fixes the expression's domain rather
  than its magnitude, and the refiner fits the rest. `Refiner.fit(..., refine_scope=)`,
  `FlashANSR(refiner_scope=)` / `FlashANSR.load(refiner_scope=)` accept `'fittable'` (default:
  every `<constant>` slot plus every spelled literal simplipy's `mask_fittable` policy would
  abstract; typed literals stay verbatim), `'placeholders'` (only the slots; every spelled literal
  is compiled in) and `'all'` (every literal, typed ones included).
  `flash_ansr.refine.refinement_slots` is the one slot definition the refiner and the verbatim
  seeding share, so `p0` stays aligned by construction and seeds every freed spelling (`-2`,
  `2.5`, `3/2`), not only digit-only tokens.
- **`infer(top_k='all')`** predicts every refined candidate on the support and validation sets;
  `CandidateLedger` carries `n_nodes`, `n_constants`, `mdl`, `score`, `pareto_rank`, `rank` and
  `result_index` (copied from the refined rows, never re-derived from beam ids) so a persisted
  ledger can be re-ranked offline and checked against the live rank 0. `Candidate` gains
  `constants_emitted`, the constants as the model predicted them before refinement, next to the
  refined `constants`.
- **Model weights are safetensors.** `model.safetensors` replaces `state_dict.pt` everywhere
  weights are written or read: `FlashANSRModel.save`/`load`, the set encoder, `FlashANSR.load`,
  and the trainer's resume path. Weights are the artifact people download, so they are stored in
  the format the ecosystem reads: a flat tensor map with a JSON header, no pickle, memory-mappable,
  framework-agnostic. Optimiser, scaler, scheduler and `training_state` stay on `torch.save`.
  `flash_ansr convert-weights <dir>...` writes `model.safetensors` beside an existing
  `state_dict.pt`; legacy pickles are no longer read, and the `FileNotFoundError` names the
  command. The converter refuses a state dict carrying non-tensor entries rather than dropping
  them silently.
- **Training tasks around the expression.** All of them are config-gated, absent from the batch
  surface when unconfigured, and pin their priors explicitly:
  - a **noise mixture** streams noisy targets to the encoder, keeps the clean targets for the
    tasks that must not learn the noise, and labels every contaminated support point;
  - a **per-point outlier head** on the set encoder trains against those labels with a BCE
    auxiliary loss (`outlier_loss_weight`, `outlier_pos_weight`), parameter-free when off so
    existing checkpoints load unchanged;
  - the **`<complexity>` block** conditions on or predicts the complexity of the masked target,
    as a full-precision byte span or as a one-token `<float>` summary; under the
    **`<hypothesize>`** flag, inserted by the harness and never supervised, the model opens the
    block itself and everything after the flag carries loss;
  - the **`<predict_y>` block** holds out one accepted support point and has the model spell its
    clean target, placed before the expression (interpolation) or after it (evaluation);
  - the **`<mask_all>` / `<mask_fittable>` flags** select simplipy's masking policy for the emitted
    expression, so a caller chooses per call whether constants are predicted or placeheld for a
    downstream fitter;
  - the **`<predict_constants>` block** carries one byte span per placeholder after the
    expression, positionally bound, so which constants to predict is stated by the expression
    itself; a partial circumstance masks a random subset of slots on a share of unflagged
    instances so infilling stays in-distribution without a flag;
  - the **`<predict_residual>` block** reports how far the observation at a queried point sits off
    the law (`residual_block`, requires a noise mixture): the coordinates are caller-supplied and
    compact, the displacement is predicted and therefore a byte span; the point stays in the
    encoder's support and is drawn from what `<predict_y>` left, so the two never query the same
    one, and the block is dropped on an unconditioned instance rather than moved.
  Commutative prompt elements are permuted per instance so none welds to a position; the
  hypothesis element is pinned last.
- **Training metrics.** Per-task cross-entropy curves, the `ce_split/{train,val}/` family (every
  task crossed with its conditioning circumstance: expression by data conditioning, by complexity
  presence, by mask policy; `predict_y` by placement and by masked context; constants by flagged
  or partial), the `expression/anchor` split shared by every arm, the composite `val_loss`, and
  every model-quality metric on both sides: `train_outlier_auroc` next to `val_outlier_auroc`,
  with AUPRC alongside. Outlier AUROC/AUPRC are logged on an interval (`outlier_metrics_interval`,
  default 50) because they are rank statistics over every micro-batch's scores; the outlier loss
  is logged every step.
- **Model flags.** `pre_encoder_bits` (16, 32 or 64: one `IEEE754PreEncoder` over every width,
  with the encoder input dimension following the format; the named subclasses stay as aliases),
  `head_pre_logits_norm` (a LayerNorm between the head MLP and the logit projection, bounding the
  norm growth of the byte-token logit rows that cross-entropy cannot see), `outlier_head`,
  `encoder_mask_query_norms` (threads the support-set padding mask into the ISAB self-refinement
  blocks' query and residual-stream SetNorms and zeroes sub-layer outputs on padded query rows,
  so a sample's encoding is invariant to padding length) and `sanitize_input_num` (zeroes the
  numeric-token bit encodings at positions without a numeric payload; the previous guard checked
  `isnan` on the bit encodings, which are never NaN, and so never fired). All default to the
  legacy behaviour so existing checkpoints load and run bit-identically.
- **Trainer.** `validate_num_workers` (config key, `-vw` on the CLI,
  `run()` override): the validation pool coexists with the training pool for the whole run, so it
  gets its own worker count instead of doubling the fleet.
- **Streaming worker pools can outlive one `iterate()`.** `iterate(keep_alive=True)` leaves a
  fully drained stream's pool running so the next identical call reuses it; the caller then owns
  the pool and must `shutdown()`. Validation uses it. An abandoned generator still shuts down,
  and a live pool raises rather than silently serving a request built for different settings.
- **Forbidden non-finite token guard on the simplification path.** `float("inf")` /
  `float("-inf")` / `float("nan")` are encodable vocabulary tokens, and SimpliPy folds a
  degenerate sub-expression to one instead of failing, so such skeletons re-entered the candidate
  stream as valid predictions. `flash_ansr.utils.skeleton.simplify_and_mask` now raises
  `NonFiniteExpressionError`, every candidate producer drops the candidate and counts it
  (`flash_ansr.utils.non_finite_drops()` / `reset_non_finite_drops()`), and the training-data
  ingest direction propagates instead, since symbolic-data rejects these before yielding and one
  arriving means a broken producer contract.
- **`tokenizer.yaml` declares its `constants_format`**, and `Tokenizer.from_config` refuses a
  vocabulary built for a different one, naming it. A vocabulary that declares nothing is
  recognised by its alphabet: one that opens `<ieee754>` spans over the retired 16-symbol nibble
  alphabet is refused with a message naming the `compat/v24-nibbles` tag that still serves it.
- **Run configurations** for the third generation under `configs/v25.0-*` (the 3M `v25.0-T7`
  recipe the reference checkpoint was trained with, and its 20M shape `v25.0-T7-20M`), with the
  `v24.0-T13`..`T16` pilot and full-run configurations of the nibble lane kept for provenance.
  The training catalogs hold out every catalog `srbf` evaluates on, 6,660 expressions across 29
  catalogs, not only FastSRB.

### Changed
- **Constants are IEEE-754 binary64, spelled as 8 byte tokens.** A serialized constant is
  `<ieee754>` + 8 tokens from the 256-symbol `<b00>`..`<bff>` alphabet + `</ieee754>`: 10
  tokens, the same span width as the retired hex-nibble format, over a wider alphabet and at
  double precision. The serializer no longer narrows a fitted constant and no longer refuses a
  finite value for exceeding a narrower format's range. Measured on generated data, 46.5% of
  constants are values binary32 could not represent.
- **The numeric width has one name.** Every numeric surface, encoder input, numeric channel,
  constants, takes its dtype from `flash_ansr.utils.numeric.NUMERIC_DTYPE`. A width mismatch is
  refused where it arises instead of surfacing as a silent cast.
- **A number's spelling follows its producer.** What the model predicts is spelled in IEEE-754
  bytes: expression constants, `<predict_y>`'s target, `<predict_constants>`' values. What the
  caller supplies is a compact `<float>` carrying its value on the numeric channel: `<predict_y>`'s
  coordinates, a stated complexity. `<float>` is forbidden at every generation position and never
  appears inside an expression.
- **`<hypothesize>` marks the boundary between what is given and what is generated.** Properties
  before it are stated by the caller, compact, and may not be restated after it; everything after
  it is the model's own and spelled in bytes. At inference the prompt ends at `<hypothesize>` when
  present and at `<expression>` otherwise, so generation always begins at the last prompt token.
  `mu`, `hypothesize` and `mask` compose in `complexity_prefix`. Query/answer blocks are exempt:
  inside them the loss mask decides, so caller-supplied coordinates stay compact wherever the
  block sits.
- **Candidate ranking is one of three modes, and the default is MDL.** `FlashANSR` /
  `FlashANSR.load` take `ranking_mode` (`'mdl'` | `'weighted'` | `'pareto'`) and that mode's
  knobs: `mdl_strength` (decades of FVU per bit; default 1e-2, one decade per 100 bits),
  `ranking_weights` (a dict over `n_nodes`, `n_constants`, `n_constant_placeholders`,
  `n_typed_literals`, `mdl` (per bit), `neg_log_prob`), `ranking_metrics` + `ranking_tie_break`
  (the non-dominated front's axes and its within-front order; the tie-break may name a metric
  outside the set). A knob given with another mode raises instead of lying dormant.
  `ranking_config()` returns the resolved values. `compile_results` takes the same knobs,
  call-scoped, and never writes them back. The `mdl` metric is simplipy's `complexity()` of the
  realized expression (refined constants substituted), priced in the refine worker; a candidate
  that cannot be priced sorts below every priced one while `mdl` is weighted, and a pool with
  nothing priceable raises `RankingError`. **Breaking:** the loose `node_penalty` /
  `constants_penalty` / `likelihood_penalty` / `mdl_penalty` estimator arguments are gone; the
  pre-0.14 ranking is `ranking_mode='weighted', ranking_weights={'n_nodes': 0.05}`.
- **Results format 2.** `save_results` writes the resolved `ranking` record in place of the four
  loose penalties; `load_results` re-orders the restored table under the file's ranking (warning
  when it differs from the estimator's, never adopting it) and refuses a payload without one.
- **`Candidate.complexity` is `Candidate.n_nodes`**; `Candidate` gains `mdl` (milli-bits of the
  realized expression), `pareto_rank` (-1 under a scalar ranking) and `rank` (position in the
  sorted list). `InferenceResult.to_dataframe()` carries them plus `mu`.
- **The default `emission` is `'fittable'`** (was `'constants'`): the application mode, in which
  the model spells the typed literals and leaves every fittable constant as a placeholder that
  the refiner fits from random inits. Pass `emission='constants'` for the unflagged format.
- **The refiner no longer frees `pow` exponents by default.** It used simplipy's deprecated
  digit-only conversion, which turned `pow x1 2` into `pow x1 C_0` and then fitted a real
  exponent from a random init, `nan` on half the axis and the mechanism behind the measured
  pow-family failures (oracle recovery on FastSRB 24% with exponents freed against 77% with the
  literals fixed). Pass `refine_scope='all'` for the old behaviour.
- **Published constants are round-trip exact.** Every returned expression used to round its
  constants to two decimals, so a converged `6.674e-11` printed as the zero function while the
  reported FVU stayed near zero. Substitution is exact by default; `precision=` on
  `get_expression` is a display-only opt-in.
- **The decode boundary keeps what the model emitted.** `np.pi` and `np.e` sit in the tokenizer's
  special-token section and were deleted from every candidate at decode, leaving an arity-short
  token list the validity gate rejected; `Tokenizer.EXPRESSION_SPECIAL_TOKENS` names the specials
  that are expression content and `decode_expression` keeps them. Sampled candidates are
  de-duplicated on the skeleton with its constant values, not the value-erased skeleton, so three
  draws of the same skeleton with different constants are three candidates and the refiner's
  inits. `initial_tokens` without `input_num` defaults to the numeric channel the checkpoint was
  trained with instead of skipping the numeric embedding.
- **The tagged canonicalization runs in the catalog's configured target canon.** The final
  target spelling follows the same `simplify_mode` as the prefix target; a catalog without the
  knob keeps the historical call byte-identically.
- **Streaming workers are spawned, not forked.** A trainer has CUDA initialised in the parent by
  the time it opens a stream, and forking a CUDA process is undefined. The pool uses an explicit
  spawn context; everything a worker needs is passed explicitly and picklably, shared memory is
  attached by name, and the source config is rebuilt into the worker's own `ProblemSource`. A
  script that opens a pool needs an `if __name__ == "__main__":` guard, because the child
  re-imports the main module (the `flash_ansr` console entry point has one). Pool creation costs
  a fresh interpreter per worker (measured at 8 workers, start-up plus first batch goes from
  0.87 s to 7.7 s); steady-state throughput is unchanged, which is what `iterate(keep_alive=True)`
  amortises.
- **The streaming pool no longer serialises through a manager process.** Per-batch metadata
  rode a `SyncManager` list proxy, one global serialization point that capped the pool at about
  300 instances per second regardless of worker count; it now rides the result queue with the
  batch. Each spawned worker runs with one torch thread, as torch's own `DataLoader` workers do,
  instead of inheriting the full intra-op pool and spin-waiting on it.
- **Placeholding is naive and positional.** Masking a constant no longer asks the engine to
  re-derive the expression and compare: each placeheld literal site simply becomes a `<constant>`
  the model predicts, and that site's value is the block's ground truth. A structurally spelled
  rational contributes one slot per literal, so `3 / 2` masks to `<constant> / <constant>` and
  trains as two predictions. The collection-stability check and its `n_collection_restructured`
  counter are gone, along with `mask_selected_sites` and `nonspecial_site_positions`.
- **`batch_size='auto'` sizes against the device that can report its memory.** A device that
  cannot (CPU, and MPS before its budget query) gets the conservative cap instead of the caps
  measured on a 24 GiB card; MPS reports through `torch.mps.recommended_max_memory()`.
- **Requirements.** `simplipy>=0.14.6,<0.15`, `symbolic-data>=0.18,<0.19`, `safetensors`.

### Fixed
- **Seven public-API surfaces that said one thing and did another.** `Refiner.transform` wrote
  fitted values into the wrong slots whenever the expression carried a numeric literal (on
  `['+','*','2','x1','<constant>']` fitted to `7*x1 + 100` it published `2*x1 + 7.0`); it now walks
  the same token list the fit was built from. `pad_input_set` returned a DataFrame unchanged, so
  `infer` on a DataFrame raised a positional-argument error from the compiled lambda; it converts
  and refuses what it cannot. `compile_results` wrote its overrides onto the estimator, so a
  sweep left every later `fit` ranked under the last value swept; it is call-scoped. Pruned-
  candidate rescoring skipped the numeric channel and so scored under a different model than the
  one that generated the candidate. An unencodable candidate vanished silently; it is a counted
  drop. `constrain_ieee754` on an unsupported checkpoint fails at load with the reason instead of
  after a full encoder forward. `generate()`'s documented `memory=` is forwarded.
- **Selection can no longer be won by a fit that did not happen.** A `NaN` restart loss poisoned
  the fit sort (NaN compares false against everything, so a divergent restart was published while
  an exact fit sat two slots down); `fit_sort_key` sinks non-finite losses and `valid_fit`
  describes index 0. A verbatim constant init was treated as terminal on finiteness alone, so a
  converged-but-catastrophic local optimum cancelled the restarts; the fallback is skipped only
  when the verbatim fit is already perfect, and the two fit sets are merged so the better one
  wins. One non-finite `y` collapsed the ranking to alphabetical order while `fit` reported
  success; the variance is taken over the finite rows and the dropped rows are warned about. The
  finiteness gate re-checks after the float cast, which can manufacture `inf`. A failed `fit`
  no longer leaves the previous problem's results in place.
- **Third-generation checkpoints serve through `FlashANSR.fit`.** The model emits the engine's
  tagged canonical dialect, which the decode boundary now normalises to explicit prefix; a
  sequence carrying tagged delimiters that does not parse is an invalid candidate, not an error.
  A constant-free candidate is a completed fit judged by whether its loss is finite; the zero-
  constant early path used to leave `valid_fit` false, so every constant-free candidate was
  silently discarded.
- **An unrealizable candidate is dropped, not fatal.** A candidate naming an operator the loaded
  engine does not define killed the whole problem's refinement batch; it is dropped like any
  invalid fit, surfaced under `converge_error='print'` and re-raised under `'raise'`.
- **Unconditioned instances no longer lose their `<predict_y>` block.** The gate excluded the
  block from every condition-dropout instance; with the expression in scope the suffix placement
  is function evaluation, a well-posed task, so the block is pinned to the suffix on unconditioned
  instances instead of being dropped.
- **Raw batches held past their turn were a use-after-free, and it segfaulted.** The tensors
  `FlashANSRDataset.iterate` yields view the streaming pool's shared-memory ring and are valid
  only until the pool refills that block. The trainer kept two of them by reference for the
  paired constant-span eval and read them after the loop, a hard `SIGSEGV`. The eval takes a
  copy at capture (`_detach_raw_batch`), and `tests/test_data/test_raw_batch_lifetime.py` pins
  the contract.
- **The `expression/anchor` split** excluded complexity, `predict_y` and mask circumstances but
  not `predict_residual`, so prefix-placed residual rows entered a baseline the docstring
  requires be shaped like the base task. **`ce_split`** had no entry for the residual segment:
  `predict_residual` tokens were supervised but never reported.
- **Encoder padding masks are coerced to `bool` (with a warning) at the `SetTransformer` entry.**
  `scaled_dot_product_attention` interprets float masks as additive logit biases, so a float 0/1
  padding mask silently masked nothing.
- **Cross-attention K/V batches are normalised before the attention call.** Cross-attention
  decodes `choices` rows against a single encoder memory, and leaving the batch dimension to
  broadcast inside `scaled_dot_product_attention` is outside its documented contract: correct on
  CPU and CUDA, silently wrong on MPS once `head_dim >= 64`. The static cross-attention path keeps
  its expanded K/V as a stride-0 view instead of materialising a per-row copy of the encoder
  memory in every layer.
- **The per-candidate refine seed no longer leaks into the caller's global RNG.** The worker
  seeds `np.random` per candidate so a fit is reproducible from its expression hash; run
  in-process, that seeding landed on the caller's global state. The state is saved and restored
  around the fit.
- **Gradient accumulation.** `do_optimizer_step=False` now accumulates: `zero_grad`,
  `unscale_` and gradient clipping moved inside the stepping guard. The trainer raises on a
  `batch_size % gradient_accumulation_steps` remainder instead of silently dropping those samples
  from every step.

### Removed
- **Generation-1 (v23) support, entirely: one generation, one code path.**
  `constant_representation` had `'v23'` as its default in the data layer, so a training config
  that omitted the key silently produced v23 data. `'ieee754_mixed'` is the only legal value and
  the default; a tokenizer without the span tokens is refused at dataset construction. The dead
  conditionals that guarded it are gone with it. The v23 model bundles and their register
  (`configs/v23.0-*`, `configs/v23.2-120M`, `configs/VERSIONS.md`) are deleted; `git` keeps them
  for anyone who needs the recipes under a pinned `flash-ansr<0.13`.
  `configs/v24-template/tokenizer.yaml` drops the eight tokens no run of this generation trained
  on (`<prompt>`, `</prompt>`, and the six term-constraint delimiters).
- **Beam search and MCTS.** `SoftmaxSamplingConfig` is the generation configuration;
  `create_generation_config` rejects any other method by name. In-decode span compaction, the
  per-row decode position it required, `tail_zero_bits`, the per-point residual head and
  `predict_residuals()` go with them.
- **The `<prompt>` wrapper lane.** `PromptFeatureExtractor` and its module, the legacy
  `serialize_prompt` half of `prompt_serialization`, the `PromptFeatures` schema and the metadata
  threading through the estimator, pipeline and trainer. Its successor, bare prefix elements the
  harness force-feeds and loss-masks, is what the models are trained on;
  `serialize_prompt_prefix` stays. `allowed_terms` / `include_terms` / `exclude_terms` are gone
  from `fit()`: they were documented as constraining generation, emitted tokens no checkpoint had
  seen, and were enforced nowhere.
- **The `simplify='sympy'` simplification path.** SymPy simplification was an ablation of the
  product simplifier, and production code carries no ablations. `simplify` is a two-state `bool`
  (`True` = SimpliPy, the default; `False` = no simplification) everywhere it is accepted, and a
  config that asks for the removed path fails loudly at `SoftmaxSamplingConfig` /
  `create_generation_config` / `sample_top_kp` and at `FlashANSRDataset.from_config` for a
  catalog requesting the SymPy skeleton path; it never falls back.
- `flash_ansr.results.compile_results_table`: a second scoring/sort implementation with a subtly
  different guard. `FlashANSR._compile_results_pure` is the one sort.
- Reading `state_dict.pt` checkpoints (see `flash_ansr convert-weights`).

Thanks to Kianté Fernandez (@kiante-fernandez) for the cross-attention, gradient-accumulation,
refine-RNG and batch-sizing fixes (#52, #53, #54, #55, #58).

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
