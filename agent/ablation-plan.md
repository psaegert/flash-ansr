# Camera-Ready Plan (May 1 – May 28, 2026)

Working reference for the ICML 2026 camera-ready of Flash-ANSR / SimpliPy.
Hard freeze: **May 28**.

## Hardware
- **Solomon** (RTX 4090): **time-critical evaluation only** (and Z1/Z2 while A100s train).
- **Valkyrie** (RTX 4090): Track A 20 M scaling sweep training.
- **A100 cluster** (3 × A100, ssh + tmux): B1/B2/B4 training in parallel. **370 h** wall clock.
- 120 M model: 370 h training on A100, ~1 d evaluation on Solomon (6-pt reduced sweep).
- 20 M model measured: SimpliPy/Unsimplified 10 M ≈ 4 h, 100 M ≈ 40 h;
  SymPy 10 M ≈ 60 h; SymPy 100 M infeasible (~600 h).

## Status (May 1)
- **B1/B2/B4**: training launched on A100 cluster, all three in parallel.
  Expected completion: **~May 16** (day 15.4, 370 h).
- **Solomon**: free until A100 checkpoints arrive → running Z1/Z2 now.
- **Valkyrie**: Track A chain running: S10(4 h) → U10(4 h) → S100(40 h) → U100(40 h) → Y10(60 h).
  Expected completion: **~May 9** (day 8.8, 148 h total).

## Binding constraint
**Solomon's eval queue** is the bottleneck, and the window is tight.
B1/B2/B4 arrive May 16. Each needs ~1 d eval (6-pt sweep) → all done **~May 19**.
Track A arrives May 9; 5 × ~30 min = ~2.5 h total — negligible.
Z1/Z2 run now; done before May 3.
Paper integration window: **May 19–28 = 9 days**. No slip allowed on evals.
Drop B4 first if any eval overruns.

## Shortlist

### Track Z — inference-only re-evaluations on existing 120 M checkpoint
No training cost. Running on **Solomon** now (while A100s train).
The rebuttal already claims "prior experiments" for both; these runs
produce the actual numbers to cite.
| ID | Description | Reviewer anchor |
|---|---|---|
| Z1 | Swap LM refiner → BFGS, compare NNR + ZSS | Av9g Q5 |
| Z2 | Swap softmax sampling → beam search, compare NNR | Av9g Q3 |

### Track B — 120 M architecture ablations
Each item is a revert of one architectural change vs Biggio et al. (2021).
56 h train + 3 d eval. Av9g Q4.1, meta-review.
| ID | Component reverted | Code anchor |
|---|---|---|
| B1 | pre-norm → post-norm decoder | `TransformerDecoderBlock` |
| B2 | 32-bit → 16-bit input encoding | `IEEE75432PreEncoder` |
| B4 | masked RMSSetNorm → LayerNorm encoder | `RMSSetNorm` |

*B3 (RoPE → sinusoidal) dropped: RoPE is a standard contemporary component,
not the intended target of Av9g Q4.1's "vs Biggio 2021" framing.*

### Track A — 20 M scaling sweep
Reuse the existing 1 M rebuttal numbers (SimpliPy / SymPy / Unsimplified).
Av9g Q1.1, meta-review. Three points per curve (1M / 10M / 100M) are
required to test the scaling-law hypothesis; A_S10 and A_U10 are therefore
**firm**, not stretch.
| ID | Pipeline | Expressions | Train (V) | Eval (S) |
|---|---|---|---|---|
| A_S100 | SimpliPy      | 100 M | ~40 h | ~30 min |
| A_U100 | Unsimplified  | 100 M | ~40 h | ~30 min |
| A_Y10  | SymPy         |  10 M | ~60 h | ~30 min |
| A_S10  | SimpliPy      |  10 M | ~4 h  | ~30 min |
| A_U10  | Unsimplified  |  10 M | ~4 h  | ~30 min |
| ~~A_Y100~~ | ~~SymPy 100 M~~ | dropped — infeasible (~25 d simplify) |

## Calendar (S = Solomon, V = Valkyrie, C = A100 cluster)

| Day | Solomon (S) | Valkyrie (V) | A100 cluster (C) |
|---|---|---|---|
| 0–2 | **Z1 + Z2** (5-pt sweeps, ~8 h total) | A_S10(4h) + A_U10(4h) running | **B1 + B2 + B4 in parallel** |
| 2–8.8 | idle / paper writing | A_S100(40h) → A_U100(40h) → A_Y10(60h) | training continuing |
| ~8.8 | **eval Track A** (5 × 30 min ≈ 2.5 h) | done | training continuing |
| 8.8–15.4 | idle / paper writing | — | training continuing |
| ~15.4 | rsync A100 checkpoints; start B eval queue | — | **done** |
| 15.4–16.4 | **eval B1** | — | — |
| 16.4–17.4 | **eval B2** | — | — |
| 17.4–18.4 | **eval B4** | — | — |
| 18.4–27 | **paper integration**, figures, captions, TODO sweep | — | — |

## Risks and fallbacks
- Solomon eval overruns 3 d → drop **B4** first; cite \citet{Zhang2022SetNA}
  and the derivation in App. E ([`app:masked_rmsnorm`](../paper/_paper.tex))
  and explicitly defer the full empirical LayerNorm comparison to future work.
- Thermal / availability slip → drop A_Y10 (use 1 M number only).
- A_Y10 simplification > 60 h → report as "infeasible at scale". This
  *supports* the paper's argument about SymPy scaling.
- A100 training extends further → B eval window shrinks to <9 d; drop B4.
- Eval hang → re-launchable checkpoint/seed protocol; abort any single eval
  after 2 d to protect the 9-day paper-writing window.

## Existing artifacts to reuse
- 1 M rebuttal results (SimpliPy / SymPy / Unsimplified).
- [configs/rebuttal-20M-simplipy/](../configs/rebuttal-20M-simplipy)
- [configs/rebuttal-20M-sympy/](../configs/rebuttal-20M-sympy)
- [configs/rebuttal-20M-sympy-50K/](../configs/rebuttal-20M-sympy-50K)
- [configs/rebuttal-20M-unsimplified/](../configs/rebuttal-20M-unsimplified)
- [configs/v23.0-120M-ablation-1/](../configs/v23.0-120M-ablation-1)
- [configs/v23.0-120M-ablation-2/](../configs/v23.0-120M-ablation-2)
- [configs/v23.0-120M/](../configs/v23.0-120M) — base for B1/B2/B4 diffs
- [scripts/rerun_ablation2.sh](../scripts/rerun_ablation2.sh)
- [experimental/eval/rebuttal_sanity_check.ipynb](../experimental/eval/rebuttal_sanity_check.ipynb)

## Open decisions
1. ~~Include B4 (LayerNorm encoder)?~~ Resolved: firm (B3 dropped instead).
2. 20 M eval scope — full FastSRB (~2 d each) vs reduced suite (~1 d each)?
   Recommended: reduced.
