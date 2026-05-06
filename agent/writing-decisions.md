# Camera-Ready Writing Decisions (deferred)

Notes-to-future-self capturing the structural and stylistic decisions made for
the camera-ready ablations writeup so we can pick up prose drafting later
without re-relitigating any of these. Companion to
[system-effect-analysis-plan.md](system-effect-analysis-plan.md) and
[camera-ready-todo.md](camera-ready-todo.md).

The actual prose is **not** drafted here — that's deferred. This is the
"planning record" so prose drafting is a transcription exercise, not a
re-decision exercise.

---

## Section 8 structure (locked)

- **§8 lead-in** (~0.2 page main): introduces a single methodological frame —
  *hypothesis-space coverage analysis* — and forward-references three
  applications below.
- **§8.1** Training-data ablation (A-track: SimpliPy / SymPy / Unsimplified at
  1M, 10M, 100M). Av9g Q1.1, meta-review.
- **§8.2** Architecture ablation (B-track: post-norm, 16-bit, LayerNorm on
  120M). Av9g Q4.1, meta-review.
- **§8.3** Inference-strategy ablation (Z-track: beam vs softmax sampling on
  120M; possibly Z1 BFGS vs LM). Av9g Q3, Q5.

---

## §8.1 — Training-data ablation

### Main text (~0.5 page)
- One figure, three panels:
  - **Panel A** — vNRR vs fit_time scaling sweep. Three pipelines × three
    training sizes (S/U at 1M/10M/100M, Y at 1M/10M; Y100M annotated
    "infeasible: ~40 days"). Bootstrap CI bands.
  - **Panel B** — distinct hypotheses (FVU ≤ ε₃₂) per fit_time. S vs U at
    100M. The "S finds ~30% more perfect-fit hypotheses at fixed compute"
    point.
  - **Panel C** — I.12.5 rewrite-cluster case study. GT canonical
    `* x₁ x₂` (length 3) at top; below it 5 of U's 161 raw rewrites at
    lengths 9–18, all FVU ≈ 2e-15. Annotate "U: 161 raw rewrites at
    choices=2048; S: 1." This is the rhetorical knockout panel.
- One paragraph wrapping the figure. Beats:
  1. S/U comparable on numerical fit at matched compute.
  2. S finds ~30% more distinct perfect-fit hypotheses (panel B).
  3. U commits compute to syntactic rewrites — case study (panel C).
  4. Y infeasible at 100M training scale (extrapolated from Fig. 4).

### Appendix N — "Anatomy of the SimpliPy effect"
- **N.0** Hypothesis-space coverage analysis: definitions of the five
  dimensions; canonical-form-via-SimpliPy oracle; self-application caveat;
  related work pointers (sampling diversity in LM literature, Pareto-front
  diversity in symbolic regression).
- **N.1** Training-data distribution shift (length distribution histogram;
  operator-frequency drift table; length-ceiling pile-up). From
  `training_data_analysis.py` at N=50K samples per pool. **Total Nestedness
  lives here.**
- **N.2** Candidate-distribution analysis at A-track: bloat-factor histogram
  (per-candidate, all pooled), rewrite-fraction-vs-choices scaling curve,
  canonical-group-size distribution (log-log), paired Wilcoxon S vs U,
  length-stratified breakdown (small vs large canonical). Bootstrap CIs
  throughout.
- **N.3** Full multi-metric plot from `training_scale_analysis.ipynb`: vNRR,
  log_prob, length ratio, F1, Recall (Variables), **Total Nestedness**, ZSS
  edit, n_constants_delta, Fit Success — at apples-to-apples completed scales.
- **N.4** Methodology note: cross-model `predicted_log_prob` is
  per-token-entropy of the training distribution, not model quality. Frames
  the rebuttal's "diluted posterior" wording correctly. Reframes the
  candidate-distribution evidence as the substantive backing for that claim.
- **N.5** Candidate-distribution analysis at Z-track (decoder axis), parallel
  template to N.2.

---

## §8.2 — Architecture ablation

### Main text (~0.4 page)
- Small table: baseline 120M vs B1 (post-norm) / B2 (16-bit) / B4 (LayerNorm)
  at choices=32 (rebuttal Fig. 14 anchor). Columns: vNRR, ΔvNRR vs baseline.
- One paragraph: which architectural change matters most for stability /
  quality at scale. Connect to Av9g Q4.1's "isolate architecture from
  SimpliPy" ask.

### Appendix M
- §3.2 architecture comparison table vs Biggio et al. (already a committed
  deliverable).
- Full per-ablation choices sweeps (vNRR vs fit_time per B-variant).
- Optional: candidate-distribution fingerprint per B-variant if results show
  meaningful differences. Otherwise, report a clean null.

---

## §8.3 — Inference-strategy ablation

### Main text (~0.2-0.3 page)
- Headline: matched-axis decoder comparison on the same 120M baseline.
  Beam_width vs choices on the x-axis, vNRR + a fingerprint summary on the y.
- Pair the candidate-distribution fingerprint with the §8.1 fingerprint for
  the cross-axis comparison: "training-axis affects concentration and
  surface-form; decoding-axis affects coverage."

### Z1 (BFGS vs LM)
- One sentence in §3.5 referencing Z1 numbers ("LM is 30% faster than BFGS at
  equal accuracy on our problems, supporting our refiner choice"). Full
  numbers in App.

---

## Cross-cutting decisions (locked)

- **No U-with-`simplify: true` parallel evaluation.** Treat simplification as
  a system-level toggle (whole pipeline including training and inference).
  The asymmetric as-configured eval is the comparison the paper makes.
- **`predicted_log_prob` deprecation** — call it out as a methodology
  limitation in App N.4, not in the main text.
- **The 161-way I.12.5 case study goes in main text** — Panel C of §8.1
  figure, not buried in appendix. Strongest single piece of rhetorical
  evidence.
- **Total Nestedness lives in App N.3**, not main text; bloat factor is
  sharper.
- **Honest framing on the "scaling law" claim**: do not over-claim "U
  flat-lines" — our data shows U comparable on vNRR and worse on
  hypothesis-space coverage. Phrase the §8.1 paragraph accordingly:
  "comparable on numerical fit; diverges sharply on hypothesis-space
  coverage and surface-form parsimony."

## Cross-cutting decisions (deferred — to revisit later)

- How prominent the SBI/TRS framing should be in the introduction (currently
  promised as a §3 lead-in tweak; possibly worth elevating to §1).
- Whether to include a numerical-equivalence (FVU+constants clustering)
  oracle as a secondary check alongside the SimpliPy canonical-form oracle
  in App N.
- §2 Problem Formulation page budget — currently estimated at ~0.5 page,
  could compress further if §3.3 cleanup runs over.
- Whether to add B-track candidate-distribution probes (depends on what the
  B1/B2/B4 fastsrb evals show; if architecture differences are large in
  vNRR, probe them; if not, skip).

---

## Drafting order (when prose work resumes)

1. App N.0 (diagnostic definition) — methodological keystone; everything
   else points to it.
2. §8 lead-in — one paragraph forward-referencing §8.1, §8.2, §8.3.
3. §8.1 main paragraph (with placeholder for Y10 point) — landing the A-track
   story.
4. §8.3 main paragraph — landing the parallel Z-track story.
5. App N.4 methodology note — short, can be drafted any time.
6. §2 Problem Formulation — draft from zDuW rebuttal verbatim as starting
   point.
7. §3.1 formal TRS, refined Kruskal, abstract Alg 1 — draft from zDuW
   rebuttal wording bank.
8. §3.2 architecture comparison table vs Biggio — draft from oUGq rebuttal.
9. §3.5 inference clarifications (Av9g Q3, Q4, Q5) — draft from Av9g
   rebuttal.
10. §3.3 cleanup + running example — relocate, do not rewrite.
11. §8.2 paragraph — once B-track lands.
12. App N.1, N.2, N.3, N.5 — once probe data lands.

Steps 1-9 can be drafted with no new experimental data. Steps 10-12 are
data-dependent.

---

## Figures inventory

| ID | Section | Status | Source | Blocking on |
|---|---|---|---|---|
| §8.1 main | three-panel scaling + hypothesis count + I.12.5 cluster | designed | new | A_Y10 (panel A); probe data (panels B, C) |
| §8.2 main | small table only | designed | new | B-track training |
| §8.3 main | matched-decoder fingerprint | designed | new | 120M softmax+beam probe |
| App N.0 | diagram of the diagnostic | not designed | new | none — can sketch now |
| App N.1 | training-data length / operator drift | designed | `training_data_analysis.py` | none |
| App N.2 | bloat-factor histogram + rewrite-vs-choices curve | designed | `analyze_candidates.py` | running probes |
| App N.3 | big multi-metric plot | exists in notebook | `training_scale_analysis.ipynb` | A_Y10 (cleanup pass) |
| App N.5 | Z-track fingerprint | designed | `analyze_candidates.py` | 120M softmax+beam probe |
| App M.* | per-B-variant choices sweep | designed | existing scaling pipeline | B-track training |

All §8 figures consume the same fingerprint utility. Define style once
(`fingerprint_plot.py`), apply uniformly.

---

## Page-budget sketch (worry about for real later)

| Block | Net main-text delta |
|---|---|
| §2 Problem Formulation | +0.5 |
| §3.1 formal TRS / Kruskal / Alg 1 | +0.3 |
| §3.2 architecture comparison table | +0.3 |
| §3.3 cleanup + running example | ±0 |
| §3.5 inference clarifications | +0.1 |
| §8 lead-in | +0.2 |
| §8.1 ablation main | +0.5 |
| §8.2 ablation main | +0.4 |
| §8.3 ablation main | +0.2 |
| SBI/TRS framing | +0.1 |
| FastSRB footnote, NRR clarification, etc. | ±0 |
| **Total estimated delta** | **+2.6** |

If we need to fit +1 page net, ~1.6 pages of compression has to come from
§3.3 cleanup overshooting target plus tighter §8.1, §8.3 paragraphs and
relocation of detailed text to appendix. Not impossible — many of the
"+0.5" estimates are upper bounds.
