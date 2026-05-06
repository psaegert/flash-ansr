# System-Effect Analysis Plan (SimpliPy vs SymPy vs Unsimplified)

Camera-ready strengthening plan for §8.1, building on the proof-of-concept
investigation completed 2026-05-05.

Audience: future-self picking this up tomorrow morning. Assumes the
investigation context from `experimental/eval/training_scale_analysis.ipynb`
and the candidate-distribution probe at `/tmp/candidate_distribution_probe.py`.

---

## 1. The hook (paper-facing claim)

**Reframe §8.1 from a metric race to a system-component analysis.**

Current §8.1 promises "SimpliPy-trained models exhibit a scaling law the
others do not" (Av9g Q1.1). That phrasing is fragile because at matched
test-time compute, vNRR for S vs U is within ~2 pp at all training scales —
the existing pickles already show this. A win is not robustly defendable.

The defendable version: **SimpliPy is a system component that enforces
canonical hypothesis representation; removing it manifests as measurable
distributional and computational effects, while a sound symbolic alternative
(SymPy) is infeasible at training scale.** This addresses the same reviewer
ask but with claims that are robust to noise.

Three independent, falsifiable claims, each with its own figure:

1. **Training-side**: SimpliPy compresses the surface-form distribution while
   preserving the hypothesis distribution. SymPy does the same (more
   aggressively) but at infeasible cost. Removing simplification (U) shifts
   the training distribution measurably.
2. **Inference-side / breadth**: At fixed test-time compute, U dedicates a
   measurable share of refinement to syntactic rewrites of already-explored
   canonical hypotheses; SymPy avoids this but has runtime barriers; S sits
   in the favorable middle.
3. **Best-of-N quality**: Despite the asymmetric distribution and the
   refinement waste, vNRR at matched fit_time is comparable across S, U,
   and (where feasible) Y — SimpliPy's contribution lands in compute breadth
   and surface-form parsimony, not in numerical fit. State this honestly;
   it strengthens the paper rather than weakening it.

---

## 2. POC inventory (what's already in hand on 2026-05-05)

### Methodological findings
- Asymmetric test-time pipeline: S evals run with `simplify: true`, U with
  `simplify: false` ([configs/evaluation/scaling/v23.0-20M-A-{S,U}100_fastsrb.yaml:38](../configs/evaluation/scaling/v23.0-20M-A-S100_fastsrb.yaml#L38)).
  This biases length / SRR / Recall(vars) / log-prob comparisons in ways
  unrelated to model quality.
- GT skeleton is SimpliPy-simplified before all structural metrics
  ([result_processing.py:236-240](../src/flash_ansr/eval/result_processing.py#L236)).
- `predicted_log_prob` is total token log-prob sum
  ([flash_ansr.py:329](../src/flash_ansr/flash_ansr.py#L329)); cross-model
  comparisons are confounded by per-token entropy of the training
  distribution.

### Training-data analysis (5K skeletons each)
- U raw length 25.5 vs S raw length 22.9; canonicalizing U recovers 22.8
  (≈ S's). 60% of U skeletons get shortened by ≥1 token under SimpliPy.
- 67% of U sits in [25, 35) length bucket vs 47% for S — pile-up at the
  `max_length=35` ceiling.
- Operator drift: `pow2`, `mult2` 2.5× more frequent in S; `*`, `/` more
  frequent in U. SimpliPy actively rewrites `* x x → pow2(x)` etc.
- SimpliPy adds ~50% to skeleton-generation time; SymPy adds ~100×.
- Canonical-form overlap between two independent 5K samples: 1/5000 — sample
  space is much larger than 5K.

### Candidate-distribution probe (N=20, choices=2048)
- U mean bloat factor 1.556 (vs 1.001 for S) — every U candidate carries
  ~50% redundant tokens.
- U mean rewrite fraction 4.9% (vs 0.2% for S); scales from 2.8% at
  choices=256 to 4.9% at choices=2048.
- Largest canonical group: U max 161 (I.12.5: `* x1 x2`), S max 3.
- Hypothesis count: S finds 94 perfect / 228 good per sample; U finds 73 /
  211. S has ~30% more perfect-fit hypotheses at fixed `choices`.
- fit_time per sample: U 8.2s vs S 11.2s — U is faster because it skips
  test-time SimpliPy.
- Net unique-canonical hypothesis discovery rate per second of compute:
  U 6.6/s vs S 5.1/s — U covers more hypothesis space per second despite
  the rewrite waste.

### Artifacts
- `/tmp/cand_dist_choices00256_n20.pkl`, `/tmp/cand_dist_choices02048_n20.pkl`
  (candidate dumps, will need to be moved to a versioned location).
- `/tmp/training_data_analysis.py`, `/tmp/candidate_distribution_probe.py`,
  `/tmp/analyze_candidates.py` (scripts, also need to move).

---

## 3. Tomorrow's plan (one-day, executable)

Goal: lift the three POC findings to camera-ready statistical rigor without
disturbing the planned B1/B2/B4 schedule.

### Sequencing (8h budget)

#### Phase 0 (15 min) — Versioning & cleanup

- Move `/tmp/training_data_analysis.py`, `/tmp/candidate_distribution_probe.py`,
  `/tmp/analyze_candidates.py` to `experimental/system_effect/` (new dir).
- Move `/tmp/cand_dist_choices*_n20.pkl` to `results/system_effect/`.
- Add a one-line README in each describing the experiment.

#### Phase 1 (90 min) — Methodological fix: parallel U-with-simplify-true eval

The cleanest disambiguation is to evaluate **the same U model** under both
test-time pipelines, so we can attribute differences to (a) training
asymmetry vs (b) test-time pipeline asymmetry.

- Duplicate
  [configs/evaluation/scaling/v23.0-20M-A-U100_fastsrb.yaml](../configs/evaluation/scaling/v23.0-20M-A-U100_fastsrb.yaml)
  → `v23.0-20M-A-U100-with-simplify_fastsrb.yaml`. Change the single line
  `simplify: false` → `simplify: true`. Adjust output paths to a parallel
  results directory.
- Same for U10 and U1.
- Run on Solomon. Cost estimate: 6 choices points × 30 min × 3 model sizes =
  ~9h. **Run in background overnight** — kick off before paper writing.

This gives us a clean 2×2: training {simplify, no-simplify} × test-time
{simplify, no-simplify}. Most of the asymmetry attributed to "training data
quality" should actually move to "test-time pipeline."

#### Phase 2 (2h) — Scale up candidate-distribution probe

Required for camera-ready:
- N=20 → **N=300** (10 samples per fastsrb eq_id × 30 eq_ids; deterministic
  selection by `benchmark_sample_index ∈ {0..9}`).
- Pipelines: S100, U100, **and U100-with-simplify** (so we can decompose the
  bloat / rewrite signal into "training" vs "test-time" contributions).
- Compute points: choices ∈ {64, 256, 2048} — so we can plot rewrite
  fraction vs `choices` as a scaling curve, not a single number.

Cost estimate (choices=2048, N=300, 3 pipelines):
- choices=2048: ~10s/sample × 300 × 3 = 2.5h GPU
- choices=256: ~3s/sample × 300 × 3 = 45 min
- choices=64: ~1.5s/sample × 300 × 3 = 22 min

Total: ~3.5h GPU. Run with `run_in_background=true`. Save dumps with full
candidate tuples to `results/system_effect/cand_dist_*.pkl`.

#### Phase 3 (90 min) — Statistical rigor on the probe

Add to the analysis script:
- Bootstrap CIs (1000 resamples, 95%) on rewrite fraction, mean bloat,
  hypothesis count, fit_time.
- Paired-by-eq_id Wilcoxon signed-rank tests for S vs U on each metric.
- Stratified means: split eq_ids into "small canonical" (canonical len ≤ 7)
  and "large canonical" (canonical len > 7) — the I.12.5 rewrite-cluster
  pattern likely concentrates in the small bucket.
- Distribution plots: histogram of bloat factor (per-candidate, all pooled),
  log-scale histogram of canonical-group size.

#### Phase 4 (90 min) — Training-data analysis at scale

- Bump 5K → 50K skeletons per pool.
- Add **SymPy pool** as a third arm (use the same skeleton_pool config but
  set `simplify: 'sympy'` if the SkeletonPool supports it; otherwise use the
  existing `v23.0-20M-A-Y50K` config and just sample 50K). Measure per-skeleton
  cost — this directly supports the "infeasible at 100M" argument with
  primary numbers.
- Bootstrap CIs on length distribution, operator-frequency drift, redundancy.
- Per-skeleton timing distribution (mean, p50, p99, max with timeout).

#### Phase 5 (60 min) — Figure design

Three figures for §8.1, sketched:

**Figure A — Training distribution shift.** 4 panels.
- Panel 1: length distribution (raw and canonical) for S, U, Y. Histogram
  with bootstrap-CI bands.
- Panel 2: operator-frequency log-ratio (U/S and Y/S) for the top 15
  operators. Bar chart.
- Panel 3: per-skeleton SimpliPy and SymPy timing distribution. Show the 100×
  gap.
- Panel 4: cumulative training-time projection at 1M / 10M / 100M for each
  pipeline. Ends with SymPy 100M extrapolated and labeled "infeasible".

**Figure B — Candidate-distribution analysis.** 3 panels at choices=2048.
- Panel 1: bloat factor histogram (per-candidate, pooled), S vs U vs
  Y-if-available. Vertical line at mean. Annotate "1.5× longer surface
  forms" effect.
- Panel 2: canonical-group-size distribution (log-log). Annotate the
  I.12.5 161-rewrite outlier as a labeled point.
- Panel 3: rewrite fraction vs choices (scaling curve). S flat at ~0.2%,
  U scaling linearly. Bootstrap-CI bands.

**Figure C — Apples-to-apples best-of-N.** 4 panels (rerun of existing §8.1).
- vNRR vs fit_time, with **U-with-simplify** added as a fourth curve. Show
  that the as-configured curve attribution decomposes cleanly.
- Length ratio vs fit_time, with U-with-simplify ≈ S.
- log-prob vs fit_time, deprecated as a quality metric (footnote: "this
  measures per-token entropy of the training distribution, not model
  quality").
- Hypothesis count (FVU≤eps32) vs fit_time, the new clean win for S.

#### Phase 6 (60 min) — Writeup hooks

Draft 2 paragraphs for §8.1 (one for the training-side, one for the
inference-side) and 1 paragraph for the appendix scaling discussion (SymPy
infeasibility). Save to `agent/section-8.1-draft.md` for review.

---

## 4. Multi-day expansion (camera-ready scope)

Stretch goals if Phase 1-6 finishes early or if the analysis surfaces
something that demands extension:

### 4.1 Cross-scale robustness
Repeat the candidate-distribution probe at 1M and 10M training scales (S, U,
and U-with-simplify). Question: does the bloat factor change with training
data scale? Hypothesis: bloat is a property of the training distribution,
not training amount, so it should stay roughly constant.

### 4.2 SymPy candidate-distribution
A_Y10 finishes ~2026-05-08. Once available, run the probe on Y10 at
choices=2048 / N=300. Expected: zero rewrites (SymPy enforces a strong
canonical form), low bloat. This anchors the "S sits in the middle"
narrative.

### 4.3 Ablate the I.12.5 pattern
Manually identify the ~5 fastsrb equations with the smallest GT canonical
form. Hypothesize these are the rewrite-cluster hotspots for U. Show that
the rewrite waste is **concentrated** on simple-canonical samples, not
uniform — sharper claim than "U has 5% mean rewrites."

### 4.4 Operator-vocabulary effect on inference
Hypothesis: U undersamples `pow2`/`mult2` at training, so U almost never
*generates* them at inference. Verify by counting `pow2`/`mult2` token
appearances in the candidate sets. If true, this connects the training-side
operator drift directly to the inference-side bloat — a single mechanism
explanation rather than two parallel observations.

### 4.5 Variance under seeds
Re-train one U pipeline with a different RNG seed for the skeleton sampler.
If feasible (only the 1M scale is fast enough), this argues the differences
are not seed artifacts. Probably a "nice-to-have," not blocker.

---

## 5. Risks and fallbacks

- **Solomon eval queue contention.** B1/B2/B4 evals are scheduled May 16-19.
  The U-with-simplify-true reruns must be done **before May 16** or fit into
  the May 19-28 writing window. Tomorrow is May 6 → ~10 days of slack.
  Run them tonight and tomorrow morning.
- **A_Y10 not ready by May 8.** Drop SymPy candidate-distribution probe;
  retain the training-data SymPy timing comparison (which uses Y50K and Y1
  pickles already in hand).
- **Stratified analysis (small vs large canonical) shows no concentration.**
  Then the rewrite-waste claim weakens to "uniform 5%" and the I.12.5
  case study becomes a pure example, not a generalization. Still publishable
  but less rhetorically powerful.
- **Phase 1 reveals U-with-simplify ≈ S.** That would mean the as-configured
  U disadvantage is mostly test-time, not training. This is a *positive*
  outcome — it sharpens the framing to "the simplification component matters
  at test time more than at training time." Adjust the §8.1 narrative
  accordingly; do not suppress the finding.
- **Probe at N=300 takes longer than estimated.** Cut to N=150 (5/eq_id) for
  choices=2048 and keep N=300 only for the cheaper choices=64 / 256 sweeps.

---

## 6. Open decisions (for tomorrow morning self)

1. **Do we keep the as-configured asymmetric eval as the "system-level"
   primary comparison, or replace it?** Recommendation: keep the as-configured
   curves as the headline (matches user's framing of "with vs without
   simplification as a whole system"), and add the U-with-simplify curve
   as the controlled comparison that decomposes the effect.
2. **Where does the appendix discussion of `predicted_log_prob` deprecation
   live?** Recommendation: a footnote in §8.1, plus a paragraph in the
   appendix ("On the limitations of cross-model log-probability
   comparisons"). Frame it as a methodology note, not a retraction.
3. **What's the headline number?** Candidates: "U dedicates 5% of test-time
   compute to syntactic rewrites at choices=2048, scaling with `choices`"
   *or* "U candidates carry 1.5× redundant tokens over their canonical
   form" *or* "S finds 30% more perfect-fit hypotheses per fixed
   `choices`." Pick the one that survives Phase 3's bootstrap CIs cleanest.
4. **§8.1 length budget.** Currently §8.1 is short. Two figures + 3
   paragraphs is ~1 page. Three figures + 4 paragraphs is ~1.5 pages.
   Camera-ready ICML allows expansion; check final page budget before
   committing.

---

## 7. Cross-references

- Existing camera-ready plan: [camera-ready-todo.md](camera-ready-todo.md)
  item 8.1.
- Existing analysis plan (currently 1M-only provisional findings):
  [analysis-plan.md](analysis-plan.md). **Update its §8.1 section** after
  tomorrow's results land.
- Ablation/training plan: [ablation-plan.md](ablation-plan.md).
- Notebooks:
  [training_scale_analysis.ipynb](../experimental/eval/training_scale_analysis.ipynb)
  (the §8.1 figure source) and
  [test_time_compute_scaling.ipynb](../experimental/eval/test_time_compute_scaling.ipynb).
- New experiment dir to create: `experimental/system_effect/`.
- New results dir to create: `results/system_effect/`.
