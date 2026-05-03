# Camera-Ready TODO

Comprehensive task list for the camera-ready revision of the Flash-ANSR /
SimpliPy paper. Synthesized from:

- Decision + reviews ([decision](review-threads/decision.md),
  [Av9g](review-threads/Av9g.md), [oUGq](review-threads/oUGq.md),
  [yyY8](review-threads/yyY8.md), [zDuW](review-threads/zDuW.md))
- The current paper at [paper/_paper.tex](../paper/_paper.tex)
- The codebase under [src/flash_ansr/](../src/flash_ansr) and
  [configs/](../configs)

**Conventions**

- **[committed]** — explicitly promised in a rebuttal reply.
- **Source** — review pointer (e.g. `Av9g Q1`) or `paper:` / `code:`.
- **Anchor** — paper label or section + line-range from current
  `_paper.tex` (line numbers are approximate; prefer `\label{...}`).
- **Code** — file + symbol when an experiment or code change is implied.
- **P** — priority: `must` (camera-ready blocker), `should`, `nice`.
- **E** — effort: `S` (≤1 day), `M` (≤1 week), `L` (multi-week, e.g. ablations).

---

## Critical path (camera-ready blockers)

These are the items the meta-review or reviewers explicitly flagged as
outstanding *and* that require new results/figures, so they have the longest
lead time:

1. **§8.1** SimpliPy vs SymPy vs Unsimplified scaling sweep
   (Av9g Q1.1, meta-review). `must`, `L`. Configs already exist at
   [configs/v23.0-20M-A-S1/](../configs/v23.0-20M-A-S1),
   [configs/v23.0-20M-A-Y1/](../configs/v23.0-20M-A-Y1),
   [configs/v23.0-20M-A-Y50K/](../configs/v23.0-20M-A-Y50K),
   [configs/v23.0-20M-A-U1/](../configs/v23.0-20M-A-U1);
   sanity-check exists at
   [experimental/eval/rebuttal_sanity_check.ipynb](../experimental/eval/rebuttal_sanity_check.ipynb).
   Extend training-set sizes to 1M / 10M / 100M and integrate into camera-ready.
2. **§8.2** Architecture ablations on 120M model (Av9g Q4.1, meta-review).
   `must`, `L`. Configs exist at
   [configs/v23.0-120M-ablation-1/](../configs/v23.0-120M-ablation-1) and
   [configs/v23.0-120M-ablation-2/](../configs/v23.0-120M-ablation-2);
   author rebuttal states they are "currently in progress". Finalize, isolate
   each architectural change (pre-RMSNorm, masked RMSSetNorm, RoPE, 32-bit
   input encoding) and integrate into Section 3.2 / Appendix.
3. **§4** Major presentation overhaul committed to zDuW (new Section 2 problem
   formulation, running example in Sec 3, implementation details moved to
   appendix). `must`, `M`. Touches large parts of `_paper.tex`.

Everything below is text-only ("[committed]" items in particular) and can land
in parallel with the experiments above.

---

## 1. Methodological framing

Goal: address the recurring concern that the contribution looks like
engineering rather than ML.

| # | Task | Source | Anchor | P | E |
|---|---|---|---|---|---|
| 1.1 | **[committed]** Add explicit framing of SimpliPy as a Term Rewriting System (TRS) that enforces algebraic invariance within the simulation-based-inference (SBI) pipeline. Position the simulator/data-gen pipeline as a first-class methodological component. | oUGq Q1, oUGq comment, yyY8 W1, decision | Section 3 lead-in (`paper/_paper.tex` ~L278), and Section 3.1 (`sec:simplipy`, ~L289) | must | S |
| 1.2 | Sharpen the contribution statement in the introduction: removing the simplification bottleneck in amortized neural SR; SBI-style enforcement of prior algebraic knowledge through the training distribution. | oUGq Q1 | Introduction (~L168) | must | S |
| 1.3 | Add a brief note that SimpliPy is **architecture-agnostic** — only requires a compatible tokenization. | oUGq Q2 | Section 3.1 closing or Section 3.2 opening | should | S |

## 2. Formal problem statement & definitions

| # | Task | Source | Anchor | P | E |
|---|---|---|---|---|---|
| 2.1 | **[committed]** Add a new **Section 2: Problem Formulation** *before* Related Work. Define dataset, hypothesis space $\mathcal{F}$, fit-quality vs. complexity trade-off, parsimony score $S(\boldsymbol\tau)$. Use the formulation drafted in the [zDuW rebuttal](review-threads/zDuW.md). Renumber existing Sec 2 (Related Work) → Sec 3, etc. | zDuW W3.i | Insert new section before current `sec:related_work` (~L214); update all section refs | must | M |
| 2.2 | **[committed]** In Section 3.1, add a **formal TRS definition** with well-formedness conditions (no new variables, strict length reduction) and the **termination guarantee**. | zDuW W3 | `sec:simplipy` (~L289–353) | must | S |
| 2.3 | **[committed]** Define the **normalized form** explicitly (shortest representative of the equivalence class) with a small worked example. | zDuW W3, Q4 | `sec:simplipy` (~L325) | must | S |
| 2.4 | **[committed]** Refine the MSF / Kruskal claim: clarify that rule discovery is **greedy and length-stratified**, not globally optimal; explain why the 16.2% redundant rules introduced by cancellation/folding have negligible runtime impact (O(1) hash lookup, length/root-op-indexed patterns). | zDuW W2 | `sec:simplipy` text around the Kruskal/MSF mention (currently ~L305 / L325) and `alg:simplipy_rule_discovery` (~L961–991) | must | S |

## 3. Algorithmic descriptions

| # | Task | Source | Anchor | P | E |
|---|---|---|---|---|---|
| 3.1 | **[committed]** Add an **abstract algorithm** for online SimpliPy simplification to Section 3.1 Phase 2. Use the rebuttal's draft (Alg 1 simplification loop). | zDuW Q3 | New algorithm in Section 3.1 (~L325) | must | S |
| 3.2 | **[committed]** Move/expand full algorithms into a renamed appendix **"Algorithms"** (currently [`app:equivalence_checking`](../paper/_paper.tex#L928), Appendix E): Alg 1 SimpliPy simplification loop; Alg 2 term cancellation (add/mult classes, polarity tracking); Alg 3 rule application (Rewrite using `R_E` hash table + `R_P` length/root-op indexed patterns). Existing `alg:equivalence` and `alg:simplipy_rule_discovery` stay. | zDuW Q3, W3 | Appendix E (~L928–991), retitle | must | M |
| 3.3 | **[committed]** Report **complexities** for both phases: offline `O(α^(L_pattern + L_target))`, ~100 h on 32 threads; online `O(i · n² · R · P + n² log n)`. Include the offline runtime in the runtime section so the comparison with SymPy is fair. | zDuW Q2 | New paragraph in Section 3.1 (~L325) and Section 5.2 `sec:symbolic_simplification_efficiency` (~L603) | must | S |

## 4. Presentation cleanup (zDuW W3)

| # | Task | Source | Anchor | P | E |
|---|---|---|---|---|---|
| 4.1 | **[committed]** Move excessive implementation details out of the main text. Specifically migrate the sampling hyperparameters and itemized parameter lists currently in Section 3.3 into Appendix I (or a new appendix subsection). | zDuW W3.iii | Section 3.3 (`sec:data_generation`, ~L366–383) → Appendix I (`app:data_gen_example`, ~L1075–1120) | must | M |
| 4.2 | **[committed]** Move the data-generation example currently in Appendix I into Section 3.3 as a **running example** woven through Section 3 (sampling → simplification → guarding → evaluation). | zDuW W3.ii | Appendix I (~L1075–1120) → inline in Section 3.3 | must | M |
| 4.3 | **[committed]** Clarify the FastSRB **footnote at line 428**: 5 expressions (B4, B7, B17, II.24.17, III.14.14) excluded because the prescribed sampling regimes induce sqrt-domain / exp-overflow issues that 100× resampling cannot resolve. Mark this as a flaw of the SRSD/FastSRB benchmark, not Flash-ANSR. | zDuW Q5 | Footnote at `_paper.tex` ~L428 (`sec:exp_test_time_compute`) | must | S |
| 4.4 | Resolve outstanding TODO comments: `% TODO: captions` at ~L1217 (above the detailed test-time-compute figures in Appendix L). Sweep ~L184, L248 for stale TODOs. | paper | `_paper.tex` ~L184, L248, L1217 | must | S |
| 4.5 | Remove or annotate commented-out blocks left in source (~L316, L344–347, L397–400) so they don't accidentally re-activate during edits. | paper | various | nice | S |
| 4.6 | Decide whether the `todonotes` package import (~L79) should remain enabled in the camera-ready or be commented out / replaced with `\usepackage[disable]{todonotes}`. | paper | `_paper.tex` ~L79 | should | S |

## 5. Architecture & inference design

| # | Task | Source | Anchor | Code | P | E |
|---|---|---|---|---|---|---|
| 5.1 | **[committed]** Add an explicit **comparison table** vs Biggio et al. (2021) in Section 3.2 (decoder norm, encoder norm, positional encoding, input encoding, optimizer, decoding). Use the table drafted in the [oUGq rebuttal](review-threads/oUGq.md). | oUGq Q2, Av9g Q6 | Section 3.2 (~L354–365); cross-link to [`app:masked_rmsnorm`](../paper/_paper.tex#L896), [`app:model_architecture`](../paper/_paper.tex#L913) | — | must | S |
| 5.2 | Verify each comparison-table entry is reflected in the code (no drift). Code already verified to match: pre-norm decoder, masked RMSSetNorm, RoPE, 32-bit IEEE-754, LM optimizer, top-k/p sampling. | code | — | [decoders/components.py](../src/flash_ansr/model/decoders/components.py), [common/components.py](../src/flash_ansr/model/common/components.py), [pre_encoder.py](../src/flash_ansr/model/pre_encoder.py), [refine.py](../src/flash_ansr/refine.py), [generation/beam.py](../src/flash_ansr/generation/beam.py) | must | S |

## 6. Inference / sampling clarifications

| # | Task | Source | Anchor | Code | P | E |
|---|---|---|---|---|---|---|
| 6.1 | Clarify in Section 3.5 that inference uses **K independent stochastic samples** (top-k / top-p, temperature-scaled), **not** beam search; motivate by multi-modal posterior. | Av9g Q3 | Section 3.5 (~L393–425) | [generation/beam.py: `run_softmax_sampling`](../src/flash_ansr/generation/beam.py#L39) | must | S |
| 6.2 | State the **purpose of the Eq. 3 / Eq. 4 ranking** explicitly (select the parsimony-optimal expression). Note the implementation also exposes optional `constants_penalty` and `likelihood_penalty` terms. | Av9g Q4 | `eq:parsimony_scoring` (~L404–415) | [flash_ansr.py: `_score_from_fvu`](../src/flash_ansr/flash_ansr.py#L260) | must | S |
| 6.3 | Justify **Levenberg–Marquardt** over BFGS (~30% faster at equal accuracy; SR constant fitting is a nonlinear least-squares problem). | Av9g Q5 | Section 3.5 (~L393–425) | [refine.py: `Refiner`](../src/flash_ansr/refine.py#L164) | must | S |
| 6.4 | Decide whether to also document MCTS decoding (present in code but not the default path). If kept secret, leave a one-line note that other decoders are supported. | code | Section 3.5 | [generation/mcts.py](../src/flash_ansr/generation/mcts.py) | nice | S |

## 7. NRR metric clarification

| # | Task | Source | Anchor | Code | P | E |
|---|---|---|---|---|---|---|
| 7.1 | In Section 4.4 / `eq:fvu`, **define NRR** explicitly as the fraction of *test problems* whose best solution achieves $\mathrm{FVU} \le \varepsilon_{32} \approx 1.19 \times 10^{-7}$ (per-problem, not per-point). Contrast with weaker SRBench thresholds (R² > 0.999). | Av9g W1 | `_paper.tex` ~L508–525 (Sec 4.4) and `eq:fvu` (~L1158–1159) | [eval/metrics/numeric.py: `is_perfect_fit`](../src/flash_ansr/eval/metrics/numeric.py#L72) | must | S |

## 8. New experiments / ablations

| # | Task | Source | Anchor | Code | P | E |
|---|---|---|---|---|---|---|
| 8.1 | **SimpliPy vs SymPy vs Unsimplified scaling sweep** (Av9g Q1.1, partially outstanding per meta-review). Train the 20M model on 1M / 10M / 100M expressions in each of {SimpliPy, SymPy, Unsimplified}; report NRR / wall-clock / fit metrics. Test the hypothesis that SimpliPy-trained models exhibit a scaling law the others do not. | Av9g Q1.1, decision | New figure/table; reference from Section 5.2 (~L603) | Configs at [configs/v23.0-20M-A-S1/](../configs/v23.0-20M-A-S1), [configs/v23.0-20M-A-Y1/](../configs/v23.0-20M-A-Y1), [configs/v23.0-20M-A-Y50K/](../configs/v23.0-20M-A-Y50K), [configs/v23.0-20M-A-U1/](../configs/v23.0-20M-A-U1); driver script [scripts/rerun_ablation2.sh](../scripts/rerun_ablation2.sh); analysis at [experimental/eval/rebuttal_sanity_check.ipynb](../experimental/eval/rebuttal_sanity_check.ipynb) | must | L |
| 8.2 | **Architecture ablations** on the 120M model (Av9g Q4.1, partially outstanding per meta-review). Isolate each architectural change (pre-RMSNorm decoder, masked RMSSetNorm encoder, RoPE, 32-bit IEEE-754 input encoding) vs Biggio et al. baseline; integrate results into Section 3.2 and Appendix. | Av9g Q4.1, decision | Section 3.2 + new appendix subsection | Configs at [configs/v23.0-120M-ablation-1/](../configs/v23.0-120M-ablation-1) and [configs/v23.0-120M-ablation-2/](../configs/v23.0-120M-ablation-2); also [configs/v23.1-120M/](../configs/v23.1-120M), [configs/v23.2-120M/](../configs/v23.2-120M) | must | L |

## 9. Memory / scalability discussion

| # | Task | Source | Anchor | P | E |
|---|---|---|---|---|---|
| 9.1 | **[committed]** Add an appendix subsection on **memory requirements** of SimpliPy vs. SymPy based on Experiment 4.5 / Fig. 4. Note RSS at import (412 MB SimpliPy vs 58 MB SymPy) and that this overhead is negligible at training/inference scale; document successful operation up to 17 variables and 35 symbols. | yyY8 W3 | New subsection in Section 5.2 (~L603) or new appendix subsection | should | S |
| 9.2 | **[committed]** Expand discussion of why **execution time is prioritized over memory** for large-scale SBI training (6h vs 40 days), while noting SymPy's incompleteness in this context (38–52% length increase, ~9% timeouts). | yyY8 L1 | Section 5.2 (~L603) | should | S |

## 10. Open / lower-priority

| # | Task | Source | Anchor | P | E |
|---|---|---|---|---|---|
| 10.1 | Mention formal alternatives (K, egglog) as future work for a more principled rule system. | yyY8 author comment | Conclusion (~L726) | nice | S |
| 10.2 | Note that the **35-symbol cap** is a practical choice; clarify that recursive rule application + meta-variable subtree binding makes effective coverage much larger than "7 symbols" suggests. | zDuW W1, Q1 | Section 3.1 (~L289–353) | should | S |
| 10.3 | Consider a **lightweight pruning pass** for the 16.2% redundant rules (mentioned but not implemented; not expected to materially affect speed). Defer to follow-up unless trivially feasible. | zDuW W2 | code (SimpliPy package, external) | nice | M |
| 10.4 | Cross-link the SimpliPy GitHub repo (~L189) and the Flash-ANSR repo (~L190) prominently in the abstract or introduction, given several reviewers' concern about reusability. | yyY8 L1 | `_paper.tex` ~L189–190, abstract | nice | S |

---

## Provenance check

Every reviewer ask is covered:

- **Av9g** W1 → §7.1; Q1 → §1.1, §1.2; Q2 → existing rebuttal text (no new task — already in current paper); Q3 → §6.1; Q4 → §6.2; Q5 → §6.3; Q6 → §5.1; Q1.1 → §8.1; Q4.1 → §8.2.
- **oUGq** Q1 → §1.1, §1.2; Q2 → §1.3, §5.1.
- **yyY8** W1 → §1.1; W2 → no action (rebuttal sufficient, reviewer accepted); W3 → §9.1; L1 → §9.2, §10.4.
- **zDuW** W2 → §2.4; W3.i → §2.1; W3.ii → §4.2; W3.iii → §4.1, §4.4; Q1 → §10.2; Q2 → §3.3; Q3 → §3.1, §3.2; Q4 → §2.3; Q5 → §4.3.
- **decision / meta-review** → §1.1 (TRS framing), §8.1, §8.2 (open ablations), §4 (presentation overhaul).

All `[committed]` items from rebuttals are flagged and present.
