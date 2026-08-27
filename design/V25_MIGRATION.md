# v25: float64 numerics + byte constant tokens — migration plan

**Status: owner-approved 2026-08-27.** Mapped by a 7-way parallel audit of the four repos,
then synthesized; every line number was verified against the working tree. Steps **S1, S2 and
S3 are LANDED** (flash-ansr 3e2cca1 / 60ec46a / the cast sweep, srbf 4f97164). Next: S4.

**Do not launch a training run between S3 and S4.** Between them the tree carries
float32-representable values in float64 containers -- self-consistent, but not the target
format, and a checkpoint trained there would be neither v24 nor v25.

### Q2 — RULED 2026-08-27: bytes, and the residual head is RETIRED

Two owner rulings, the second of which deletes the question rather than answering it.

1. **f64 + bytes, not nibbles.** 8 positions x 256 symbols, one numeric format everywhere.
2. **The per-point encoder residual head is retired.** Residual prediction moves to the
   DECODER as a `<predict_residual>` block, built exactly like `<predict_y>`: **one point per
   instance, never eight, never all**, with the same suffix/prefix conditioning structure.

So Q2 (head width, 61,760 vs 432,320 params) is moot -- there is no head. So is the whole
`residual_scale` ruler question (MAD vs |y|+MAD), so is the `residual_loss_weight`
re-derivation, and so is the non-finite masking guard: the residual becomes prompt payload
spelled in IEEE bytes, supervised by the ordinary next-token CE, gated by the codec's own
finiteness check exactly as `y*` already is.

**Two places it must NOT copy predict_y** (owner rulings 2026-08-27):

* **The point is NOT held out.** `predict_y` does `np.delete` on the chosen row
  (`streaming.py:743-748`) so the target is genuinely unseen. The residual must not: it is
  `y_observed(x*) - f(x*)`, and `y_observed` reaches the model only through the encoder. Delete
  the row and the target becomes a single unobserved noise draw whose irreducible loss is the
  noise entropy forever. Keeping it in makes the prefix arm the capability that was wanted --
  *infer f from the data, report the displacement at a point you can observe* -- and it is not
  a lookup: retrieving `y*` at a specific `x*` out of pooled set-encoder memory is real work.
* **On an unconditioned (nulled-memory) instance the block is DROPPED, not suffix-pinned.**
  `predict_y` pins to the suffix there because with the expression it degenerates to function
  evaluation, which is well-posed. The residual has no such fallback: with nulled memory
  `y_observed` is unreachable in *both* placements. This follows the `predict_constants`
  doctrine ("supervising values against a NULLED memory is a nonsense task"), not predict_y's.

**Also gate on `noise_spec is not None`.** Without a noise mixture `y_encoder IS y_clean`, the
residual is identically 0.0, and the block would teach nothing but "emit eight zero bytes".
Same T0 contract the `outlier_mask` key already follows.

**Ordering against predict_y.** predict_y deletes its row first; the residual block must pick
its point from what remains, or the two race for the same row.

Block shape, identical in cost to predict_y -- `4 + n_dims + IEEE754_SPAN_LENGTH`:

    <predict_residual> <point> c ... c </point> <ieee754> b0 ... b7 </ieee754> </predict_residual>

**What this deletes** (a net simplification, and it lands with S4/S5 since it needs the byte
codec and the two new block tokens):

| where | what goes |
|---|---|
| `model/flash_ansr_model.py` | `residual_head`, and the `residual_head is not None` arm of the point-representation capture (the outlier head still needs it) |
| `train/train.py` | `_RESIDUAL_SCALES`, `_masked_median`, `_scaled_residual`, `_residual_scores`, `_residual_loss`, `_float32_to_nibbles_torch`, the `residual_*` metrics and the val-composite term |
| `data/streaming.py` | the `residual` shm buffer and its fill -- the worker holds `y_encoder` and `y_support` locally, so one point's residual is computed inline (ring 320 -> 304 MiB) |
| `data/data.py`, `data/collate.py` | the `residual` batch key and its clone-not-cast hazard |
| `tasks.py` | `predict_residuals` re-implemented on the decoder path, mirroring the predict_y verb |
| configs | T18's `residual_head`, `residual_loss_weight`, `residual_scale` |

**What this adds:** `<predict_residual>`/`</predict_residual>` (free -- S5 regenerates the
tokenizer anyway), `TASK_SEGMENT_PREDICT_RESIDUAL = 4`, a `predict_residual` block config
(`p_present`, `min_n_support`, `p_conditional`), the streaming block, and the inference verb.

### What S3 decided that the plan left open

* **One name for the width.** `flash_ansr.utils.numeric.NUMERIC_DTYPE` / `NUMERIC_DTYPE_NP`.
  Every site the sweep touched dereferences it instead of spelling a literal, because the
  failure mode has no signature: a missed site produces a correctly-shaped tensor of
  scrambled bits, not an error. The next width change is one line plus a test.
* **Q7 (`streaming.py:929-934`) -- intent preserved, not flipped.** The comment's stated
  reason was "difference in the dtype the ENCODER reads". That dtype is now binary64, so the
  differencing follows it. Cancellation is still whatever the data does, which was the point.
* **`data.py:583/599/601` (the `include_metrics` diagnostics) -- widened.** No v24 config
  turns them on and `estimate_fisher_metric` is dtype-agnostic (`jacrev`/`vmap`), so the only
  live question was whether a diagnostic should describe the data in a narrower width than the
  data. It should not. `collate.py:264/266` moved with them, or the producer and the consumer
  would have disagreed.
* **`collate.py:256` (`complexity`) -- widened, though nothing reads it.** mu is exact in
  binary32 below 2**24 and the measured band tops out at 790k, so this is uniformity, not a
  fix.
* **Masks stay float32.** `outlier_mask` and `data_attn_mask` are boolean data; widening them
  would double 32 MiB of shared ring for nothing. S10 narrows them to `np.bool_` properly,
  with the `.clone()` that a bool buffer requires.
* **`train.py:296/297/311` NOT touched.** Pre-existing float64 in the AUROC/AP rank
  statistics, unrelated to the numeric channel. A blanket sweep caught them; they were put
  back as literals so the constant keeps meaning one thing.
* **The residual label and its mask now read the same narrowed value.** `_residual_scores`
  computes `encoded = residual.to(torch.float32)` and both the nibble label and the
  `isfinite` mask read *that*. Without it the f64 lane admits residuals the f32 codec cannot
  spell, and they encode as inf's bit pattern -- a well-formed target for a value the head can
  never be right about. When S4 widens the codec, deleting that one line is the whole change.
* **`configs/{test,v24.0-T17,v24.0-T18}/model.yaml` pin `pre_encoder_bits: 64`.** The code
  default stays 32 (C4) and is now itself under test: the frozen v24 checkpoint yamls omit the
  key, and flipping the default would rebuild them silently at the wrong width instead of
  failing at `load_weights(strict=True)`.
* **`fit()` normalizes the caller's dtype in BOTH branches.** It previously passed a
  caller-supplied `torch.Tensor` through with only a device move -- which, against a
  reinterpreting pre-encoder, is the scrambled-bits path with a public entry point.

Owner rulings folded in:
* **effort=4 everywhere** for the simplipy engine.
* **Train at FULL precision.** Precision reduction is an inference-time study only, if at all.
* **Q1** — unify the pre-encoder over 16/32/64 rather than deleting the 16-bit arm. DONE.
* **Q2** — resolved without a trade: re-factor the residual head to a shared output projection
  plus a position embedding. The 7x parameter objection was an artifact of the
  `Linear(d, positions * symbols)` shape, not of bytes; under `Linear(d, symbols) + pos_emb`
  f64 bytes cost 88k against today's 61.8k. Bytes everywhere, one numeric format.
* **Q3** — gate the constants format on the ALPHABET (`<b00>` vs `<h0>`), not on `<ieee754>`,
  which both formats share.

Still to re-derive by measurement, not carried over: `residual_loss_weight` (the irreducible
CE floor moves 1.73 -> 4.16 nats), and everything listed under S8.

---

{
  "summary": "Map every site that assumes float32 or 8-nibble constants across the four repos, then synthesize an ordered migration plan",
  "agentCount": 8,
  "logs": [
    "mapped 7/7 subsystems, 296 sites"
  ],
  "result": {
    "subsystems": 7,
    "plan": "# f64 + byte-token migration — one ordered plan

Repos: `FA` = /home/psaegert/Projects/flash-ansr, `SD` = /home/psaegert/Projects/symbolic-data, `SR` = /home/psaegert/Projects/srbf. Line numbers verified against the working tree on `feat/ac-core` unless marked \"per map\".

---

## 0. Corrections — things in the maps that are wrong or that two maps contradict

**C1. `p_nibbles` is not the expanded/compact mixing probability.** Map 7 calls `configs/v24.0-T17/dataset_train.yaml:37 p_nibbles: 0.5` \"the expanded/compact mixing probability … consumed by the symbolic-data serializer, so the rename must land on both sides together.\" Verified false on both counts. It sits inside `complexity_block:` (`p_present`/`p_nibbles`/`p_hypothesize`, validated at `FA/src/flash_ansr/data/data.py:192`) and is read at `FA/src/flash_ansr/data/streaming.py:695` to pick the complexity block's *nibbles vs `<float>` summary* variant. The expanded/compact coin flip is `expanded_probability` in `serialize_constant_tokens`. And `symbolic-data` contains **zero** ieee754/nibble references outside one comment in `generative.py` — the rename is flash-ansr-only (streaming.py:695, data.py:192, and 12 config files).

**C2. srbf is not out of scope.** Map 2: \"srbf has zero references to ieee754/nibbles/span tokens — nothing to migrate there.\" The first clause is literally true (`grep -rlin 'nibble|ieee754' srbf/src` → empty), the conclusion is wrong. srbf couples through `len(self.model.tokenizer)` at `SR/src/srbf/model_adapters.py:171` into `CandidateStoreWriter(vocab_size=…)`, which raises at `SR/src/srbf/candidate_store.py:43` for `vocab_size > 256`. Verified: that constructor call sits inside `_capture_ledger`'s `except Exception as exc: warnings.warn(…)` (candidate_store.py caller, adapters.py:180), so at vocab 335 the THOROUGH tier silently produces an **empty candidate store for a whole campaign** while the eval reports success. Map 7 is right; treat this as a hard prerequisite, not srbf cleanup.

**C3. The encoder input must go f64 — maps 1/3/4 are right, map 5 is wrong.** Map 5 says `paired_eval.py:119-121` x/y tensors are \"model compute and should stay f32\", warning that `torch.autocast` refuses to cast float64. The autocast fact is true but misapplied here: I traced every consumer of `x_tensors`/`y_tensors` (`train.py:1100`, `train.py:1294`, `paired_eval.py:119-122`) — all three concatenate them into `data_tensor` and pass it to `forward` → `_create_memory` (flash_ansr_model.py:495), whose **only** use is `self.pre_encoder(data)`. The pre-encoder does `view(int)` + shift + `(int8 - 0.5) * 2`, which promotes to the torch default dtype, so f64 stops dead at the pre-encoder output and never reaches an `nn.Linear` or an autocast-listed op. Keeping this tensor f32 would re-impose the range wall on the support points — where map 6 measured 0.381% of rows overflowing to ±inf and 0.059% silently flushing to 0.0. Encoder input goes f64; `amp_dtype`, TF32 and the GradScaler stay exactly as they are (map 5 is right about those).

**C4. `pre_encoder_bits` default — map 1 says flip to 64, map 7 says do not.** Map 7 wins on evidence: `runs/v24.0-T16/checkpoint_1500000/model.yaml` omits the key and relies on `config_.get(\"pre_encoder_bits\", 32)` (flash_ansr_model.py:482). Resolution that satisfies both: **keep the default at 32**, add 64 as a third accepted value, pin `pre_encoder_bits: 64` explicitly in every new config (memory rule: pin artifacts and versions), and make the mismatch impossible to miss by adding the dtype assert in the pre-encoder (S1). The assert, not the default, is what closes the hazard.

**C5. The \"12 of 983 FastSRB literals\" mechanism in the brief does not match the code.** Map 6 could not reproduce it: the `prepared` expressions symbolic-data actually evaluates carry 888 literals, all inside f32 range; the 12 out-of-range literals (3 over, 9 under) live in the `accept` field, routed to `entry.meta` at `SD/src/symbolic_data/catalog.py:74` and never evaluated. The *conclusion* (the extreme regime is absent from training) stands and is independently measured — 0.489% of otherwise-valid draws killed, 0.381% of rows → ±inf, 0.059% → 0.0 — but the regime arrives through **evaluated y and support points**, not through substituted literals. That changes which sites are load-bearing: `SD/generative.py:1121/1266/1302` and `SD/catalog.py:398-399` matter more than the literal casts. Worth confirming the 983 denominator with whoever wrote the brief.

**C6. The silent-underflow half of the boundary has no guard anywhere.** Overflow (`-3.63e+87 → -inf`) is caught by `np.isfinite` in three places. Underflow (`1.19e-52 → 0.0`) is finite, passes every gate, and ships as a corrupted target: 54/91,648 rows in generation, 32/512 and 67/512 y-values in FastSRB I.6.2/I.6.2b, and — inside flash-ansr — `compaction.py:172` compacting a beam onto a `<float>` carrying 0.0 that the model never emitted. No map treats this as the headline; it is the more dangerous half.

**C7. Residual-head parameter counts, reconciled.** Map 1 says 24,576, map 7 quotes the T18 config's 61.8k. Both right at different scopes; computed: the head is `Linear(192,192)+GELU+Linear(192,W)`. Totals — nibbles (W=128): **61,760**; f64 bytes (W=2048): **432,320** (7.0×, ~14% of a 3M model, for a loss weighted 0.05); f64 nibbles (W=256): **86,464**. Other tensor deltas: `encoder.embedding` 110,592 → 221,184; vocab tensors (untied embedding + head + bias) 36,575 → 128,975.

**C8. Map 2's `IEEE754_SPAN_LENGTH` headline is correct and is the plan's load-bearing fact.** Verified in `utils/ieee754.py:50`: `IEEE754_N_NIBBLES + 2` = 10 before and after (8 nibbles of f32 = 8 bytes of f64). Nothing that budgets slots moves: `constrained.py:109`, `beam_compaction.py:230`, `streaming.py:723/787`, `paired_eval.py:76-77`, `generation.py:90/174` max_len defaults, `decoder_max_seq_len: 256`. Say this in the commit message or someone will double a max_len.

---

## 1. Rulings needed from the owner (and which step each blocks)

| # | Question | Blocks | My recommendation |
|---|---|---|---|
| **Q1** | `IEEE75416PreEncoder` / `float16_to_ieee754_bits` (pre_encoder.py:23-39, 63-82) is a reduced-precision **training-input** arm. The full-precision ruling appears to forbid it, and it can never be re-framed as an inference-time study: changing `encoding_size` changes `output_size` → the encoder embedding shape → from-scratch training only. Maps 1 and 2 disagree (open question vs. \"leave it alone\"). | S3 | **Retire it** — delete the class, the `pre_encoder_bits == 16` branch and `test_b2_pre_encoder_16bit`. It cannot serve the sanctioned study. Reviving costs one from-scratch arm. |
| **Q2** | Residual-head width. (a) 8 bytes × 256 = 2048 outputs, head 61,760 → **432,320** params, honours \"one numeric format everywhere\" (the ruling quoted verbatim in `flash_ansr_model.py:270-275`). (b) 16 nibbles × 16 = 256 outputs, head 86,464, same f64 value, 5× smaller, breaks the one-format invariant. (c) leave the head at f32/nibbles — rejected: with `residual_scale: 'none'` the raw residual is exactly the quantity that can leave f32 range, so (c) re-imposes the gate being removed. Either way, the noise share moves from 5/8 positions × 4 bits to **6/8 positions × 8 bits** (byte0 = sign + 7 exp; byte1 = 4 exp + top 4 mantissa; bytes 2-7 = low 48 mantissa), so the head's irreducible CE floor goes ~1.73 → ~4.16 nats of the position mean and `residual_loss_weight: 0.05` is wrong by construction. | S4, and **T18 cannot be configured until this is settled** | (a) if the one-format ruling stands; re-derive the loss weight from a fresh measurement either way. |
| **Q3** | Format discriminator. `_validate_checkpoint` (flash_ansr.py:1070) gates v24-ness on `<ieee754>` presence; `IEEE754GrammarConstraint.__init__` (constrained.py:43-50) checks presence only. Both would pass on a nibble-era vocabulary. | S4 | Respell the alphabet `<b00>`..`<bff>` (two-digit — note `<b0>`/`<b1>` were the **retired bit tokens**, and `tests/test_ieee754.py:125-127` still rejects `[\"<b0>\"]*8`, so never write `f\"<b{v:x}>\"`), **and** add `constants_format: 'ieee754_f64_bytes'` to tokenizer.yaml, required by `_validate_checkpoint`. The respelling alone makes both gates fail loudly; the key makes it explicit. |
| **Q4** | Rebuild the six frozen `SD/assets/catalogs/*.npz` at f64? Verified not range-clipped (max abs nonzero 1.747e30); only mantissa is gained. Rebuilding moves every stored value by ≤1 f32 ulp and every published number measured against them. | S6 (optional) | **Do not rebuild.** Add a `storage_dtype` marker to the `_meta` blob (catalog.py:241-254 / :304) so a mixed f32/f64 corpus is detectable. |
| **Q5** | Beam expansion. `expansion_per_beam = max(1, min(vocab_size, beam_width * expansion_factor))` = 64 at beam_width 32 (flash_ansr_model.py:1005, beam_compaction.py:319). Today all 16 in-span tokens always make top-64 → **in-span search is exhaustive**; at 256 symbols only 64 of 256 are expanded. This moves search quality independently of the format. | S4; must be settled **before** any recovery number from a byte checkpoint is compared to a v24 baseline | Make expansion grammar-state-aware (expand `IEEE754_N_BYTE_SYMBOLS` when the constraint reports \"inside a span\") in **both** loops, so the comparison is apples-to-apples. |
| **Q6** | New `constant_representation` string? `CONSTANT_REPRESENTATIONS` is a 1-tuple (serialization.py:78-79), validated at data.py:152 and serialization.py:154. The required-token gate at data.py:157 already refuses a stale tokenizer loudly, so a new string is defence-in-depth. **But** `train.py:1354` gates the entire T12 paired eval on `== \"ieee754_mixed\"` — a new string silently deletes `val_ce_constant_*`/`train_ce_constant_*` from the run with no error. | S4 | Either keep the string (the token gate is the real lock), or add `ieee754_mixed_f64` **and** change train.py:1354 to a membership test in the same commit. Do not do the first half only. |
| **Q7** | `streaming.py:929-934` differences the residual in f32 with an explicit \"not float64\" rationale (\"exactly the displacement present in the data, cancellation and all\"). The stated intent is \"difference in the dtype the encoder reads\", which under S3 becomes f64. | S3 | Re-rule, don't flip mechanically. If the intent is preserved, it becomes `.astype(np.float64)`. |
| **Q8** | `tail_zero_bits` bound 0..23 → 0..52 (two copies: serialization.py:162, data.py:183) silently redefines `configs/v24.0-T13/dataset_train_tail16.yaml`'s `tail_zero_bits: 16` from \"keep 7 of 23 mantissa bits\" to \"keep 36 of 52\". Under bytes the knob only has token-level meaning at multiples of 8 (was 4); 16 satisfies both, so nothing fails loudly. | S4 | Re-pin as a fraction or as \"keep N mantissa bits\"; declare the T13 D-arm result not comparable across the cut. |

---

## 2. Blast radius

**Total loss, no compat shim (matching the v23→v24 precedent).** Four tensors in every v24 checkpoint change shape and none is reshapeable:

| tensor | now | after | why |
|---|---|---|---|
| `encoder.embedding.weight` | [192, 576] | [192, 1152] | 18 features × 32 bits → × 64 |
| `numeric_embedding.weight` | [192, 32] | [192, 64] | compact `<float>` channel |
| `decoder.tok_embeddings.weight` | [95, 192] | [335, 192] | vocab 95 − 16 + 256 |
| `next_token_head.3.weight`/`.bias` | [95, 192] / [95] | [335, 192] / [335] | untied, so both move |

Plus `residual_head.3` 128 → 2048 outputs under Q2(a) (no local artifact carries a residual head; T16 has `residual_head: false`).

- **Checkpoints dead:** `runs/v24.0-T16/checkpoint_650000` and `checkpoint_1500000` (the only 1.5M-step run in the tree). `runs/hf-v23.0-3M` is already refused as pre-v24, but note it also relies on the `pre_encoder_bits` default — a further reason not to flip it (C4). `load_weights(strict=True)` (utils/weights.py:46-47) turns every one of these into a loud RuntimeError, not a partial load. **Keep it strict.**
- **Tokenizers dead:** all ten `configs/*/tokenizer.yaml` (T13, T14, T14-base, T15, T15-base, T16, T17, T18, v24-template, test). Verified vocab counts: T16/T17/T18 = 95 → 335; v24-template 87 → 327; test 104 → 344. Also the two resolved copies inside `runs/v24.0-T16/checkpoint_*/tokenizer.yaml` — **never rewrite those in place**; the vocabulary and the embedding rows are one artifact.
- **Evidence dead:** everything under `runs/v24.0-T16/evals/` (cap1..cap9, matrix_summary.json, the t16-vs-v23 FastSRB rows). Constant-infilling precision is a per-nibble measurement whose unit changes. That directory already contains `matrix_summary.SUPERSEDED-float32-contaminated.json` — this exact contamination has bitten once.
- **Nothing published is at risk.** Verified: `PUBLISHED_MODEL = None` in both `tests/test_inference.py:25` and `tests/test_results_serialization.py:112`, and `model/manage.py` hard-codes no repo id. No HF asset encoding the nibble alphabet was ever released.
- **Timing is favourable.** T17 and T18 are unspent (no `runs/` directories, no live training). Their configs cost nothing to regenerate now.
- **Compat path — recommended:** carry **no code**. Tag the pre-migration commit `compat/v24-nibbles` and pin the venv; `pip install flash-ansr@<tag>` reproduces T16 exactly. A live compat lane would mean maintaining a second copy of the codec, the pre-encoder, the grammar constraint, the compaction id maps and the head width — the entire format layer — to serve one frozen checkpoint. Freeze `configs/v24.0-T13..T16` as belonging to that tag; regenerate `v24-template`, `T17`, `T18`, `test`.
- **srbf on-disk:** every existing `problem_*.npz` is uint8-token/float32-scalar. Readers must key off the stored dtype, never assume. New files become uint16/float64; the ~13-25 B/candidate figure in the module docstring must be re-measured.
- **Metrics that change unit without changing code** (nothing errors; every pre/post curve is simply non-comparable): residual CE floor ln16 = 2.773 → ln256 = 5.545; `residual_nibble_acc/*` chance 1/16 = 0.0625 → 1/256 = 0.0039; the composite `val_loss` that drives checkpoint selection shifts ≈ +0.14 at weight 0.05 (train.py:1409-1415); `ce_constant_expanded`/`_compacted` roughly double (paired_eval.py:152-157, whose comment carries an explicit numeric bridge to the pilot's +0.032 gap that stops being valid). The `ce_constant_gap ≈ 0` acceptance bar is scale-free and survives intact.

**Explicitly not changing** (three maps independently flag these; a repo-wide float32→float64 sweep would eat all of them):
- `flash_ansr_model.py:1932` `* 4` and `utils/generation.py:467` `dtype_bytes: int = 4` — fp32 KV-cache bytes, not the numeric format. Changing to 8 halves every decode batch for nothing.
- `SR/metrics/numeric.py:181` `is_perfect_fit` (float32 eps) and `SR/result_processing.py:252` (the reference-relative floor, published in `SR/CHANGELOG.md:40` as \"max(reference FVU, float32 eps)\"). These are **decision bars** that borrow f32 eps as a constant. Moving them moves every published verdict. Their locks (`tests/test_fvu_correctness.py:24,236`, `tests/test_eval/test_reference_metrics.py:52`) stay.
- `FA/flash_ansr.py:156` `_T11_ACCEPT_FVU = float(np.finfo(np.float32).eps)` — deliberately tied to srbf's bar. Do not touch without srbf agreeing; otherwise the two desynchronize and the benchmark moves for a reason unrelated to this migration.
- `SR/metrics/token_prediction.py:251,270` (`torch.float32` precision/recall ratios) — `result_processing.py:345` explicitly reproduces \"the SAME torch formula (and float32 dtype + NaN→0)\"; changing one side re-opens the hand-rolled-metric divergence.
- `SD/_generate/holdout.py` — already f64 end to end, zero f32. Its 4-dp standardized image keys and `_DEFAULT_HOLDOUT_GRID_SEED = 20240617` are the decontamination artifact; the migration leaves them **bit-stable**. Say so in the CHANGELOG so nobody \"harmonizes\" it later.
- `SD/tools/build_ai_descartes.py:272-285` (cancellation-stable relativistic rewrite) and `build_first_principles.py:130-133` (ln(force) target). These are what the benchmark *asks*, not how it stores. Changing them is a benchmark change.
- `serialization.py:230-262` `find_ieee754_spans` / `truncation_cuts_ieee754_span` — tag-driven, width-agnostic, correct for any span width.
- `data.py:823` `.clone()` on the residual handoff (the comment at 818-821 explains it; `.to(same_dtype)` returns a live view onto the recycled ring).
- All `torch.bool` masks and `torch.long` ids in collate (lines 170, 175, 218, 223, 233, 238, 246, 251, 274, 275, 287) and `input_ids` `np.int64` (streaming.py:200).
- `CHANGELOG.md` history in all three repos — add entries, never rewrite (policy D41/D42).

---

## 3. The ordered plan

The tree is runnable and self-consistent after every step. The key structural decision is **splitting the flash-ansr work along two independent axes**: S3 changes *tensor widths* (f32→f64 dtypes, 32→64-bit pre-encoder) while the codec still snaps every constant to f32, so spans and the compact form still agree; S4 then changes *what a constant's value is* (the codec, the alphabet, the token spellings). Each is separately reviewable and separately testable, and splitting costs nothing in artifacts because neither survives a checkpoint. **Do not launch a training run between S3 and S4.**

---

### S1 — Guards and duplicate removal (FA). Mechanical. No behaviour change.
Land these on the still-f32 tree so they are already in place when the widening happens.

- `model/pre_encoder.py:12` — assert `x.dtype is torch.float32` before `x.view(torch.int32)`. This is the single most important line in the plan: `Tensor.view(dtype)` does not validate, it resizes the last dimension. 18 float32 values reinterpreted as 9 int64 and expanded to 64 bits each = 576 — **exactly** the width the old encoder expects, so a 64-bit pre-encoder fed f32 data produces a correctly-shaped tensor of scrambled bits and raises nothing.
- `model/flash_ansr_model.py:499` — after `B, M, D, E = data_pre_encodings.size()`, assert `D == self.encoder_max_n_variables and E == self.pre_encoder.encoding_size`. `_create_memory` currently reads D and E off the tensor instead of checking them, which is why it absorbs the above silently.
- `tasks.py:53` — delete `NIBBLES_PER_SPAN = 8` (a second, hard-coded span width, ten lines after the module imports `IEEE754_N_NIBBLES` at :43; its comment \"a float32 significand is 8 hex nibbles\" is already wrong) and use the shared constant.
- `scoring.py:113` — `NIBBLE_TOKENS` membership → module-level frozenset (the tuple scan goes 16 → 256 comparisons per token on `count_constants`' hot path).
- `data/streaming.py` `_producer_worker` (~414-969) — the body is `try: while True: … finally: shm.close()` with **no except clause**. A raise from `serialize_constant_tokens` kills the worker and the pool then blocks forever at `data.py:783 get_completed_slot()` with no error message. Add an `except Exception` that counts-and-continues in the spirit of the existing `n_dropped_*` counters. This is cheap insurance that becomes load-bearing in S6.

**Size:** ~45 lines, 5 files. **Depends on:** nothing. **Risk:** none.

---

### S2 — srbf candidate store widening (SR). Mechanical. Can land in parallel with S1.
- `candidate_store.py:43` — bound 256 → 65536; `:89-91` tokens `uint8` → `uint16`; `:105-107` `const_vals` → `float64`; `:96` `fvu` → `float64` (a genuine FVU of ~1e-50 currently flushes to 0.0 on disk and reads back as a perfect recovery — exactly what `fvu_exact()` was written to settle). `log_prob` may stay f32; making both f64 keeps one rule.
- `:9,19-23` docstring layout contract; `:204-207` `_self_test` at vocab 335 / uint16 / f64, re-run and re-state the bytes/candidate figure.
- `model_adapters.py` — add an explicit vocab/dtype check in `prepare()` that is **not** inside `_capture_ledger`'s `except Exception` swallow.
- `tests/test_eval/test_candidate_store.py:21,28,34,40,49` — add a `vocab_size=335` case that must succeed (today it raises), round-trip dtype uint16.

**Size:** ~50 lines, 3 files + 1 test. **Depends on:** nothing. **Must land before S4** or the vocab crossing silently empties the THOROUGH tier.

---

### S3 — Phase A: f64 tensors and the numeric channel (FA). One commit. Judgement in three places.
Everything here moves *tensor dtypes*. The codec still snaps to f32 (`serialization.py:200`), so the expanded and compact forms continue to agree on every value — the tree stays self-consistent, just carrying f32-representable values in f64 containers.

**Pre-encoder / model**
- `pre_encoder.py` — add `float64_to_ieee754_bits` (`view(torch.int64)`, `arange(63,-1,-1, dtype=torch.int64)`, MSB-first) and `IEEE75464PreEncoder(encoding_size=64)`, each with the S1 dtype assert. Per **Q1**, delete `float16_to_ieee754_bits` / `IEEE75416PreEncoder`.
- `flash_ansr_model.py:194-200` — accept 64, dispatch, update the error text. **Keep the `= 32` default at :170 and :482** (C4).
- Automatic consequences: `encoder.embedding` [192,576]→[192,1152], `numeric_embedding` [192,32]→[192,64]. No literal to edit — `output_size = encoding_size * input_size`.

**Every f32 cast that supplies `data` or `input_num`** — these must land together or the paths disagree about what a constant is (the failure already documented at flash_ansr.py:514-519):
- `flash_ansr.py:1877,1879,1885,1887` (the `fit()` boundary), `:1894-1901` (keep the finiteness check, drop the float32 framing and the 3.4e38 bar), `:522-526` (rescoring `input_num`), `:1943-1946` (**pin `null_memory` to the model's parameter dtype, not `data_tensor.dtype`** — otherwise an f64 memory meets an f32 decoder's cross-attention projection).
- `tasks.py:183-184` (`_encoder_batch`), `:203`, `:215` (`_start_state`/`_append`). Note `:661-663` already parses `x_query` as `np.float64` and then narrows it at `:203` — a half-migrated path already in the tree.
- `collate.py:164` (x/y), `:177` (residual), `:195` (constants), `:203`, `:211` (numeric channel).
- `streaming.py:173,177` (x/y ring), `:192` (residual ring). Ring cost at the live T16 shape (pool 32, batch 128, 512 support, 17 vars): 176 MiB → 344 MiB shared. **Leave `:183`/`:196` (`outlier_mask`, `data_attn_mask`) at f32** — they are boolean data and a blanket sweep doubles 32 MiB for nothing (see S10).
- `compaction.py:167` (`torch.isfinite(torch.tensor(decoded))` — **no dtype argument**, so it judges finiteness in f32 and raises `ValueError` on exactly the constants the migration exists to admit), `:172` (worse: silent, `1.19e-52 → 0.0` passes :167 and gets compacted onto the channel), `:182-190`.
- `beam_compaction.py:169,171`. Note `:286` gates with `math.isfinite` on a Python float and is already f64-correct — under f64 it *admits* a row that compaction.py:167 then *rejects*, and the beam loop crashes. Fixing :167 is what makes the two authorities agree.
- `flash_ansr_model.py:903-904`, `:1672-1673`, `:1905-1906` — three copies of the `numeric_template` idiom (beam, sampling, static-KV); `flash_ansr.py:522-526` is the fourth. All four or none.
- `paired_eval.py:119-121` (encoder input, per C3), `:133,136` (numeric channel).
- `train.py:987` (`_scaled_residual`), `:990` (the MAD ruler). If the buffer goes f64 and :987 does not, an out-of-range residual becomes ±inf and is **silently dropped** by the `isfinite` mask at :1025 — the head keeps missing the extreme regime with no error.
- `data.py:583,599,601` — diagnostics only, reached only under `include_metrics` (no v24 config sets it). Widen for uniformity or leave; make it a decision, not an oversight.

**Deliberately deferred to S4** (moving them here would make the expanded and compact forms disagree): `serialization.py:200` (`value32` snap), `streaming.py:730` (`y_star` pre-round — it exists so predict_y's target is exactly spellable by the codec), `streaming.py:484` (literal array dtype).

**Judgement calls inside S3:** Q1 (fp16 encoder), Q7 (residual differencing dtype, streaming.py:929-934), and the `data.py:583` diagnostics.

**Tests:** `tests/test_models/test_pre_encoder.py:159-183` — add the 64-bit case feeding `torch.randn(..., dtype=torch.float64)`, **plus a negative test that the f64 pre-encoder refuses a float32 input** (the current suite would pass on a silently reinterpreted tensor: `torch.randn(2,5,4).view(torch.int64)` gives shape (2,5,2,64), 128 values, not (2,5,4,64)). `test_norm_position.py:96-97` keep asserting 32 as the default, add an explicit-64 case. `test_noise_streaming.py:106` residual dtype.

**Size:** ~300 lines, ~20 files. **Depends on:** S1, Q1, Q7. **Breaks:** all v24 checkpoints (encoder + numeric embeddings).

---

### S4 — Phase B: nibbles → bytes, f32 → f64 in the codec (FA). One commit, unavoidably large.
`utils/ieee754.py` is the single source of truth; everything below dereferences its constants, so a split commit leaves a green-looking tree with two disagreeing encoders.

- **`utils/ieee754.py`** — rewrite the docstring for binary64 / 8 bytes / 256 symbols with a new worked example (`-2.0` → `0xc000000000000000` → `<bc0>` + 7 zero bytes). `IEEE754_N_NIBBLES` → `IEEE754_N_BYTES` (**value stays 8**, comment \"64 bits / 8\" — a missed rename here is invisible, which makes it the constant most likely to be left half-migrated). `IEEE754_N_NIBBLE_SYMBOLS` 16 → `IEEE754_N_BYTE_SYMBOLS` 256. `NIBBLE_TOKENS` → `BYTE_TOKENS = tuple(f\"<b{v:02x}>\" for v in range(256))` per **Q3**. `IEEE754_SPAN_LENGTH` **unchanged, formula and value**. Codec bodies: `:112-114` `float64`/`view(np.uint64)`, `shifts = arange(8*(N-1), -1, -8, dtype=np.uint64)`, `& np.uint64(0xFF)`; `:117-124` same plus `sum(dtype=np.uint64).view(np.float64)`; `:150-159` `pattern = (pattern << 8) | byte`, `struct.unpack(\">d\", struct.pack(\">Q\", pattern))`; `:75-86` drop the `np.float32` round-trip and the overflow `ValueError` (**this is the deletion the migration exists for**), keep `math.isfinite`. Rename `wrap_float32` → `wrap_float64`.
- **`data/serialization.py`** — drop `:200` `value32`; `:162` bound 0..52 (**Q8**); `:212-214` `\">d\"` / `to_bytes(8,\"big\")`; `:215` `wrap_float64`; `:219-223` drop the f32 overflow arm, keep a finite assertion; `:297` alphabet check follows the constant; `:7,9,135-137,144` docstrings. Index arithmetic at `:310,312,318,320` is **unchanged** (8 inner, span 10).
- **`data/data.py:183-184`** bound 0..52. `:157` needs no edit — it is the desired fail-loud gate.
- **`data/streaming.py`** — codec calls at `:690,:697,:757,:795`; `:730` drop the `np.float32` pre-round; `:484` → `np.float64`; `:681` comment (`2**24` → `2**53`); `:700,:759,:760,:798` mask lengths **unchanged** (8 is 8).
- **`train/train.py`** — `_float32_to_nibbles_torch` (`:279-289`) → `_float64_to_bytes_torch`: `.to(torch.float64).view(torch.int64)`, `arange(8*(N-1), -1, -8, dtype=torch.int64)`, `& 0xFF`. **Shifts must be int64**, and this must land in the same commit as the numpy sibling — `tests/test_residual_head.py:65-67` asserts they are bit-identical and they are separately hand-written. Then `:1020,:1022,:1036`; metric key renames `:1214-1222,:1394-1402` (`residual_nibble_acc` → `residual_byte_acc` — renaming orphans dashboards, but *keeping* the name is worse: the same key would silently change baseline from 1/16 to 1/256); `:187` `(\"nibbles\",\"float\")` must move with `streaming.py:701` or `ce_split/*/complexity/prompted` silently becomes an empty curve; `:1354` per **Q6**; `:423-432,259,996-999,1007-1009,1139-1148,1557` prose.
- **`model/flash_ansr_model.py`** — `:281` residual head per **Q2**; `:257`/`:232-233` follow `len(tokenizer)` automatically; `:2070-2089` span-id cache; `:764-767` re-word (the −5.45/−15.64 measurement is T16-specific and must be re-measured, not re-asserted).
- **`decoding/`** — `constrained.py:43,54-55,106,115` follow `BYTE_TOKENS` (measure `torch.isin` over 256 ids on the beam path before assuming it is free); `:109,113,118` arithmetic **unchanged**, docstring `:1-23` restated. `compaction.py:144-150,162-171` id maps + byte decoder, arithmetic unchanged. `beam_compaction.py:160` `wrap_float64`; `:102,132-133,230,284` unchanged. Per **Q5**, grammar-aware expansion in `flash_ansr_model.py:1002-1010` and `beam_compaction.py:316-324`.
- **`tasks.py`** — `:275,305` loop stays 8; `:289-294` restricted softmax becomes 256-way (re-check any published `temperature` default — the same temperature now acts over 16× more mass); `:454,462,464,466-467`; `:545,659,761` hoist the three `[int(tokenizer[t]) for t in …]` copies to a cached helper (256 lookups per call now); `:9-14` strengthen the leading-token warning — one flipped first token now spans ~1e±308 instead of ~1e±38.
- **`flash_ansr.py`** — `:2016-2021` handshake; `:183-185` comment; `:1070` discriminator per **Q3**.
- **Tests** — `test_ieee754.py:42-127` rewritten against binary64 (**`:103-105` inverts: 1e39 now serializes**; keep the `:125-127` spirit and add `<h0>` to the rejected set). `test_residual_head.py` corpus f64, `max() < 256`, head shape, `test_loss_is_ln16…` → `ln256`; `:238-268` **the premise of `test_masking_follows_the_scaled_value` is false in f64** (1e20 / (1.4826·1e-30) ≈ 6.7e49, finite) — rebuild with f64-scale magnitudes or the regression goes untested. `test_constant_decoding.py`, `test_constant_beams.py`, `test_paired_constant_eval.py:83-94`, `test_constant_representation.py:116-123`, `test_task_blocks.py`, `test_mask_modes.py`, `test_tagged_target_streaming.py`, `test_postprocess_spans.py`, `test_condition_dropout.py`.
- **New test, write it first:** a full round-trip over content values **> 15**. Three sites encode \"4 bits per position\" as bare literals (`ieee754.py:113,123` `& 0xF`; `:156` `<< 4`); two of them *corrupt* rather than raise if missed, and the consumer that would surface it (`tasks.py:466`) casts to `np.uint8`, which holds 0..255 happily. **Nothing in the current suite covers a content value above 15.**

**Size:** ~800 lines, ~35 files. **Depends on:** S2, S3, Q2, Q3, Q5, Q6, Q8.

---

### S5 — Regenerate configs (FA). Mechanical, but generate the token list, never hand-type it.
Regenerate `configs/v24-template`, `configs/v24.0-T17`, `configs/v24.0-T18`, `configs/test` tokenizer.yaml (256 `<b00>`..`<bff>` entries in byte order; header point 4 restated as \"8 byte tokens over the 256-symbol alphabet … = 10 tokens\", now carrying float64; add `constants_format:` per Q3). Add `pre_encoder_bits: 64` explicitly to every new model.yaml. Update `T18/model.yaml:49-54` and `T18/train.yaml:46-65` per Q2 — every quoted number there (32.5× CE-term count, 1/16 chance line, \"low 20 mantissa bits\", \"raw float32 residual\") is an f32/16-symbol fact. Rename `p_nibbles` → `p_expanded`/`p_byte_block` if desired: flash-ansr only, and it must move with `streaming.py:695` **and** `data.py:192`'s `probability_keys` tuple (C1). **Freeze** `configs/v24.0-T13..T16` and both `runs/v24.0-T16/checkpoint_*/tokenizer.yaml` as belonging to the `compat/v24-nibbles` tag.

**Size:** ~10 files, mostly generated. **Depends on:** S4.

---

### S6 — symbolic-data: f64 storage, then lift the boundary rejection (SD).
**Hard ordering: this comes after S4.** If the rejection is lifted while flash-ansr's codec still refuses out-of-f32-range values, the first such literal reaches `streaming.py:484` → `serialize_constant_tokens` raises → worker dies → the pool hangs at `data.py:783` with no error message. And the underflow case does not even raise: it serializes the bit pattern of exact zero, a silent target corruption (C6). S1's worker except-handler is the backstop.

Internal ordering within S6 (violating any of these produces a silent no-op or a silent re-narrowing):
1. **Generation before storage.** `source.py:50-53` is the single choke point where realized arrays become f32, but the *rejection* lives upstream. Change source.py alone and you ship f64 arrays holding f32-narrow content — the migration looks done and admits nothing.
2. **Probe and full path together.** `generative.py:1266` (32-row probe) and `:1121` (batched probe, the default v24 hot path) gate the draw *before* `:1302` runs. Leave either at f32 and `:1302` is unreachable.
3. **`support_sampling.py:418` and `:423` together.** `:423`'s empty-rest sentinel is `dtype=np.float32` and flows into `np.concatenate` at `generative.py:1161` — widening only `:418` lets numpy quietly downcast the whole box back.
4. **`catalog.py:398-399` and `:423-424` together** — first batch and adaptive top-up must agree on validity.
5. **`problem.py:195-206` with catalog.py**, or real-data catalogs re-narrow at `Problem` construction after surviving the sampler.

Full site list: `generative.py:1103,1121,1133,1154,1156,1236,1266,1302` + the comment block `:1273-1281`; `support_sampling.py:58,106,119,130,384,418,423,425`; `catalog.py:398-399,423-424`; `source.py:50-53,64-65,260-265`; `noise.py:363-365` (**keep the guard, move it to the f64 boundary** — multiplicative noise and the outlier shove can still overflow f64) and `:8-10,54,302-306`; `problem.py:146-147,195-206,226,233`. Tests: `test_per_point_rejection.py:88-96` (rewrite — `exp(60x)` now stays finite across all of [0,10]), `test_noise.py:101-107,131`, `test_gt_kind.py:65,104-116` (1e50 is finite in f64; use 1e400/inf), `test_source.py:36,108,166`, `test_registry.py:5`, `test_support_oversampling.py:24` (re-capture all baselines in one commit so the diff is attributable — removing the casts adds no rng draws, so values and stream are preserved, but **acceptance widens, so the accepted sequence diverges**).

Two independent wins to call out in the CHANGELOG: (a) the literal f32 snap at `:1103/:1236` is undone by the f64 widening at `:82`, so `substitute_constants` records the 17-digit f32 round-trip — 49.05% of literals change spelling, mean literal string 7.11 → 10.61 chars, defeating the `rounded` prior's entire purpose (description-length control); (b) `tools/audit_finite_fraction.py` measures `finite_fraction` in f64 while `catalog.py:398` rejects in f32, so the shipped metadata over-estimates the realizable fraction and undersizes the oversampling budget for exactly the extreme entries — the migration closes this for free.

**Size:** ~130 lines across 7 src files, ~8 test files. **Depends on:** S4, Q4.

---

### S7 — srbf constants dtype (SR). Mechanical, ~5 lines.
`data_sources.py:191` `constants` → `float64` — the only coercion between symbolic-data's `Problem` and the eval record, and the srbf-side twin of the generation gate. Test fixtures: `test_eval/test_catalog_source.py:18-21`, `test_eval/test_benchmark_config.py:368-369,437-438`, `test_benchmark.py:48`. **Do not** touch `test_eval/test_reference_metrics.py:52`'s f32-eps assertion. **Depends on:** S2; land with or after S6.

---

### S8 — Re-baseline and re-measure. Judgement.
Nothing here is code. Re-measure, do not carry forward: the T18 residual-loss weight (Q2 changes the irreducible floor); the `flash_ansr_model.py:764-767` numeric-channel log-prob evidence; `paired_eval.py:152-157`'s historical bridge to the pilot's +0.032 gap; srbf's bytes/candidate figure; symbolic-data's \"5e-5 of instance-draws\" noise-rejection rate and the reachable fraction of the `ScaleTransform` scale prior (nobody has measured how much of it f32 was truncating — worth doing before claiming the X distribution changed); the whole of `runs/v24.0-T16/evals/`. Per the memory rule, none of these go to the owner unmeasured, and per Q5 no recovery number from a byte checkpoint is compared to a v24 baseline until the expansion question is settled.

### S9 — Docs and CHANGELOGs. Mechanical.
`FA/docs/concepts.md:18` (D' = 64 × D), `FA/design/INFERENCE_API.md`, `FA/docs/getting_started.md:18` (stays true if the tags keep their spelling — revisit only under Q3), `SD/docs/sampling.md:148`. New CHANGELOG entries in all three repos; **rewrite nothing** (`FA/CHANGELOG.md:139-153`, `SD/CHANGELOG.md:125-130,145,171-172`, `SR/CHANGELOG.md:40` are accurate history). State explicitly that the holdout decontamination keys are bit-stable.

### S10 — Deliberately deferred, each its own change
- **Mask narrowing** `outlier_mask`/`data_attn_mask` f32 → `np.bool_` (saves 12 MiB, is the right shape). **Not free:** `data.py:806/817` rely on `.to(torch.bool)` *allocating* (the comment at :818-821 says so). Against a bool buffer that cast is a no-op returning a live view onto the recycled ring — the SIGSEGV documented at `train.py:315-323`. Requires an explicit `.clone()` at both sites, separately tested.
- **`_validate_step` drops `zero_tail_bits`** (train.py:1352-1364) so paired views are always built with a 0-bit tail even on an arm training at `tail_zero_bits: 16`. Pre-existing; amplified by a 52-bit mantissa.
- **Validation autocast asymmetry**: the train step guards `enabled=False` when `amp_dtype == torch.float32` (train.py:1102-1105); `_validate_step` (train.py:1296) enters autocast unconditionally. Becomes load-bearing the moment anyone runs the `FLASH_ANSR_AMP_DTYPE=fp32` arm.
- **Frozen `.npz` rebuild** (Q4) and the `storage_dtype` marker.

---

## 4. The three steps most likely to hide a bug

1. **S3, the pre-encoder + every f32 cast that feeds it.** `Tensor.view(dtype)` resizes the last dimension without validating: a 64-bit pre-encoder fed float32 data emits a **correctly-shaped 576-wide tensor of scrambled bits** and raises nothing, anywhere — and `_create_memory` reads D and E off the tensor rather than checking them, so it cannot catch it. The reverse direction (f64 data, 32-bit pre-encoder) gives 36×32 = 1152 vs an expected 576 and *does* raise, which is precisely why the dangerous direction is the one you are moving toward. There are eleven independent cast sites feeding two channels across five files; missing any one reproduces this. Mitigation is the two asserts in S1 plus the negative pre-encoder test — do not ship S3 without all three.

2. **S4's bit arithmetic — the `& 0xF` / `<< 4` family.** Three sites encode 4-bits-per-position as bare literals (`ieee754.py:113`, `:123`, `:156`). Two of them silently truncate rather than raise if missed (only `:156` overflows `'>I'` and faults), and the one consumer that would surface it, `tasks.py:466-467`, casts draws to `np.uint8`, which holds 0..255 perfectly happily and keeps returning wrong values forever. Compounding it: `_float64_to_bytes_torch` and `float64_to_byte_values` are separately hand-written in different files with a bit-identity test between them (`test_residual_head.py:65-67`), so a split commit leaves a green tree with two disagreeing encoders. Write the >15 round-trip test **before** touching the codec.

3. **S4's token spelling and the two presence-only gates.** `NIBBLE_TOKENS` is `f\"<h{v:x}>\"`; the same expression over `range(256)` re-emits the literal strings `<h0>`..`<hf>` for values 0..15 — same spellings, different meanings (byte 0x0f vs nibble 0xf). A stale tokenizer.yaml would then satisfy 16 of the 256 required tokens, and both gates that could catch it check presence only: `IEEE754GrammarConstraint.__init__` (constrained.py:43-50) and `_validate_checkpoint` (flash_ansr.py:1070, which asks only whether `<ieee754>` exists). The result is an 8-byte grammar enforced over nibble tokens, confidently, with no error. Today a shape mismatch on `tok_embeddings` would fault first — that is luck, not a check. Two-digit `<b00>`..`<bff>` (never `f\"<b{v:x}>\"`, and note `<b0>`/`<b1>` were the retired *bit* tokens) turns both gates into loud load-time failures.

**Runner-up:** S6's ordering. Lifting symbolic-data's rejection before flash-ansr can represent the result kills a worker inside a `try/finally` with no except clause, and the pool then hangs at `get_completed_slot()` forever with no error message — a hang that looks exactly like slow generation. S1's worker handler exists for this."
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Discover"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Synthesize"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "map:pre-encoder + model",
      "phaseIndex": 1,
      "phaseTitle": "Discover",
      "agentId": "aa47fb326ee6c387e",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787841823720,
      "queuedAt": 1787841820169,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "flash-ansr pre-encoder + model (src/flash_ansr/model/pre_en…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787842222509,
      "tokens": 113999,
      "toolCalls": 41,
      "durationMs": 398789,
      "resultPreview": "{\"subsystem\":\"flash-ansr pre-encoder + model (src/flash_ansr/model/pre_encoder.py, flash_ansr_model.py, encoders/set_transformer.py, and their immediate couplings)\",\"sites\":[{\"file\":\"/home/psaegert/Projects/flash-ansr/src/flash_ansr/model/pre_encoder.py:6-20\",\"what\":\"`float32_to_ieee754_bits`: `i = x.view(torch.int32)` (line 12) and `bit_idx = torch.arange(31, -1, -1, device=..., dtype=torch.int32…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "map:ieee754 codec + serialization",
      "phaseIndex": 1,
      "phaseTitle": "Discover",
      "agentId": "a06fd60b66ccdc20f",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787841823547,
      "queuedAt": 1787841820169,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "ieee754 codec + serialization (flash-ansr numeric token for…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787842321875,
      "tokens": 140362,
      "toolCalls": 59,
      "durationMs": 498328,
      "resultPreview": "{\"subsystem\":\"ieee754 codec + serialization (flash-ansr numeric token format)\",\"sites\":[{\"file\":\"src/flash_ansr/utils/ieee754.py:1-25\",\"what\":\"Module docstring is the format spec: \\\"binary32\\\", \\\"8 HEX NIBBLES\\\", \\\"32-bit pattern\\\", tokens[0]=bits 31..28, the -2.0 -> 0xc0000000 -> <hc><h0>x7 worked example, \\\"16-symbol alphabet instead of 2\\\", and \\\"float64 magnitudes that overflow float32 refuse …"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "map:decoding + grammar",
      "phaseIndex": 1,
      "phaseTitle": "Discover",
      "agentId": "a26a65d662180aa9b",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787841823637,
      "queuedAt": 1787841820169,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "flash-ansr inference: constrained decoding, grammar mask, b…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787842346973,
      "tokens": 138604,
      "toolCalls": 46,
      "durationMs": 523335,
      "resultPreview": "{\"subsystem\":\"flash-ansr inference: constrained decoding, grammar mask, beam search, KV compaction, and the auxiliary task verbs (src/flash_ansr/decoding/*, utils/ieee754.py, tasks.py, plus the beam/sampling drivers in model/flash_ansr_model.py and the estimator seams in flash_ansr.py)\",\"sites\":[{\"file\":\"/home/psaegert/Projects/flash-ansr/src/flash_ansr/utils/ieee754.py:50\",\"what\":\"IEEE754_SPAN_LE…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "map:data path + ring",
      "phaseIndex": 1,
      "phaseTitle": "Discover",
      "agentId": "a752babceee4d6a98",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787841823541,
      "queuedAt": 1787841820169,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "flash-ansr data path + ring (src/flash_ansr/data/{streaming…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787842328916,
      "tokens": 139225,
      "toolCalls": 51,
      "durationMs": 505375,
      "resultPreview": "{\"subsystem\":\"flash-ansr data path + ring (src/flash_ansr/data/{streaming,data,collate,serialization}.py, plus the two utils they pull in)\",\"sites\":[{\"file\":\"src/flash_ansr/data/streaming.py:484\",\"what\":\"`literals = np.asarray(literal_values, dtype=np.float32)`. `mask_literals_positional` returns full float64 Python floats (utils/skeleton.py:171-174 does `float(token)` / `float(num)/float(den)`); …"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "map:train + metrics",
      "phaseIndex": 1,
      "phaseTitle": "Discover",
      "agentId": "a37d2bc42042320b3",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787841823596,
      "queuedAt": 1787841820169,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "flash-ansr train + metrics (src/flash_ansr/train/*, plus th…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787842181339,
      "tokens": 103762,
      "toolCalls": 40,
      "durationMs": 357743,
      "resultPreview": "{\"subsystem\":\"flash-ansr train + metrics (src/flash_ansr/train/*, plus the training-side tests and the T18 trainer config)\",\"sites\":[{\"file\":\"/home/psaegert/Projects/flash-ansr/src/flash_ansr/train/train.py:15\",\"what\":\"`from flash_ansr.utils.ieee754 import IEEE754_N_NIBBLES, IEEE754_N_NIBBLE_SYMBOLS` — the whole residual head and its metrics are parameterised off these two names (8 and 16).\",\"chan…"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "map:symbolic-data",
      "phaseIndex": 1,
      "phaseTitle": "Discover",
      "agentId": "a9bbae72c395c17b5",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787841823354,
      "queuedAt": 1787841820169,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "symbolic-data (/home/psaegert/Projects/symbolic-data, v0.15…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787842512417,
      "tokens": 164028,
      "toolCalls": 63,
      "durationMs": 689062,
      "resultPreview": "{\"subsystem\":\"symbolic-data (/home/psaegert/Projects/symbolic-data, v0.15.0, branch feat/ac-core)\",\"sites\":[{\"file\":\"src/symbolic_data/generative.py:1302\",\"what\":\"THE primary f32 boundary. `y_support = np.asarray(y_evaluated, dtype=np.float64).astype(np.float32)` — the terminal cast in `sample_data`'s sequential path. Expression is evaluated in f64, then snapped; a finite f64 |y| > 3.4028e38 becom…"
    },
    {
      "type": "workflow_agent",
      "index": 7,
      "label": "map:srbf + configs/assets",
      "phaseIndex": 1,
      "phaseTitle": "Discover",
      "agentId": "a8999cf60073eabee",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787841823952,
      "queuedAt": 1787841820169,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "srbf + flash-ansr configs/assets (v24.0-T16/T17/T18 tokeniz…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787842277086,
      "tokens": 130752,
      "toolCalls": 60,
      "durationMs": 453134,
      "resultPreview": "{\"subsystem\":\"srbf + flash-ansr configs/assets (v24.0-T16/T17/T18 tokenizer, model, checkpoints, published HF assets)\",\"sites\":[{\"file\":\"/home/psaegert/Projects/srbf/src/srbf/candidate_store.py:43\",\"what\":\"`if vocab_size > 256: raise ValueError(f\\\"vocab_size {vocab_size} > 256 does not fit uint8; bump the token dtype\\\")`. Measured: T16/T17/T18 tokenizers are vocab=95 (49 special + 29 ops + 17 vars…"
    },
    {
      "type": "workflow_agent",
      "index": 8,
      "label": "synthesize-plan",
      "phaseIndex": 2,
      "phaseTitle": "Synthesize",
      "agentId": "a3b663e6b2cead816",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1787842516738,
      "queuedAt": 1787842514100,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /home/psaegert/Projects && grep -rn \"p_nibbles\" flash-an…",
      "promptPreview": "CONTEXT. We are migrating flash-ansr end-to-end from float32 to float64 numerics, and from
4-bit NIBBLE constant tokens to 8-bit BYTE constant tokens. Decided facts, do not re-litigate:

  * f64 is needed for RANGE, not precision: 12 of 983 FastSRB literals fall outside float32
    (-3.63e+87 -> -inf, 1.19e-52 -> 0.0); generation currently REJECTS those samples, so that
    regime is missing from …",
      "lastProgressAt": 1787843012850,
      "tokens": 148808,
      "toolCalls": 13,
      "durationMs": 496111,
      "resultPreview": "# f64 + byte-token migration — one ordered plan

Repos: `FA` = /home/psaegert/Projects/flash-ansr, `SD` = /home/psaegert/Projects/symbolic-data, `SR` = /home/psaegert/Projects/srbf. Line numbers verified against the working tree on `feat/ac-core` unless marked \"per map\".

---

## 0. Corrections — things in the maps that are wrong or that two maps contradict

**C1. `p_nibbles` is not the expanded/c…"
    }
  ],
  "totalTokens": 1079540,
  "totalToolCalls": 373
}