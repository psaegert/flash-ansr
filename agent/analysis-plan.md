# Ablation Analysis Plan

Tracks what to do in `experimental/eval/test_time_compute_scaling.ipynb` as
results come in.  The notebook's `results` dict already has stubs — just
uncomment the relevant lines at each milestone.

---

## Already available (May 2)

### §8.2 — Inference-method ablation (Z1/Z2)

**Action**: Run `test_time_compute_scaling.ipynb` as-is.
Z1 (`v23.0-120M-Z1-bfgs`) and Z2 (`v23.0-120M-Z2-beam`) are live in the
`results` dict.

**Expected output**: vNNR vs test-time compute (choices / beam_width),
overlaid on the baseline `v23.0-120M` softmax curve.

**Paper target**: §6.3 (LM justification) and §6.1 (softmax vs beam).
Cite the choices=32 vNNR values from Z1/Z2 vs baseline.

---

## ~May 3 evening — A_S100 finishes on Valkyrie

1. `scp` model to Solomon.
2. Run: `flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-20M-A-S100_fastsrb.yaml -v`
3. `./scripts/get_results.sh`
4. In `test_time_compute_scaling.ipynb` → uncomment `v23.0-20M-A-S100`.
5. **Don't make a §8.1 figure yet** — wait for U100 so SimpliPy and Unsimplified
   are both complete at 100M.

---

## ~May 4 evening — A_U100 finishes

1. `scp` + eval + `get_results.sh` (same as above for A_U100).
2. Uncomment `v23.0-20M-A-S10`, `v23.0-20M-A-U10`, `v23.0-20M-A-S100`,
   `v23.0-20M-A-U100` in the notebook.
3. **Produce the §8.1 SimpliPy vs Unsimplified figure** (3-point curve:
   1M / 10M / 100M).  The 1M point comes from the existing rebuttal results
   under `results/evaluation/rebuttal/{simplipy,unsimplified}/fastsrb/` at
   `choices_00032.pkl` (or whichever single point best matches choices=32).
4. Draft §8.1 text based on this 2-line plot.  SymPy 10M is still missing —
   leave a `\todo` placeholder.

---

## ~May 7 — A_Y10 finishes

1. `scp` + eval + `get_results.sh`.
2. Uncomment `v23.0-20M-A-Y10`.
3. Complete the §8.1 figure with the SymPy 10M point.
4. Finalize §8.1 text (3 lines: SimpliPy / Unsimplified / SymPy at
   1M / 10M / 100M).

**Note**: SymPy 100M is dropped as infeasible (~25 d simplification).
State this explicitly in the paper: "SymPy simplification at 100M expressions
is infeasible (estimated >600 h), which itself supports the argument for
SimpliPy."

---

## ~May 16 — B1/B2/B4 checkpoints arrive from A100 cluster

1. `rsync` from A100 cluster to Solomon:
   ```bash
   rsync -av <a100>:/path/to/models/v23.0-120M-B{1-postnorm,2-16bit,4-layernorm}/ \
     ~/Projects/flash-ansr/models/ansr-models/
   ```
2. Chain evals on Solomon:
   ```bash
   flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-B1-postnorm_fastsrb.yaml -v \
     && flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-B2-16bit_fastsrb.yaml -v \
     && flash_ansr evaluate-run -c configs/evaluation/scaling/v23.0-120M-B4-layernorm_fastsrb.yaml -v
   ```
3. `./scripts/get_results.sh`
4. In `test_time_compute_scaling.ipynb` → uncomment B1/B2/B4.
5. Produce the §8.2 architecture-ablation figure (6-pt sweep per model) and
   summary table (vNNR @ choices=32 for baseline vs B1 vs B2 vs B4).
6. Finalize §8.2 / Section 3.2 text.

---

## Summary table (fill in as numbers arrive)

| Model | Pipeline | Training exprs | vNNR @ choices=32 | Status |
|---|---|---|---|---|
| v23.0-20M (baseline) | SimpliPy | 2M\* | — | existing |
| v23.0-20M-A-S1 | SimpliPy | 1M | — | existing |
| v23.0-20M-A-S10 | SimpliPy | 10M | **TBD** | eval running |
| v23.0-20M-A-U10 | Unsimplified | 10M | **TBD** | eval running |
| v23.0-20M-A-S100 | SimpliPy | 100M | **TBD** | ~May 3 |
| v23.0-20M-A-U100 | Unsimplified | 100M | **TBD** | ~May 4 |
| v23.0-20M-A-Y10 | SymPy | 10M | **TBD** | ~May 7 |
| v23.0-120M (baseline) | SimpliPy | — | — | existing |
| v23.0-120M-Z1-bfgs | SimpliPy+BFGS | — | **TBD** | done |
| v23.0-120M-Z2-beam | SimpliPy+Beam | — | **TBD** | done |
| v23.0-120M-B1-postnorm | SimpliPy | — | **TBD** | ~May 16 |
| v23.0-120M-B2-16bit | SimpliPy | — | **TBD** | ~May 16 |
| v23.0-120M-B4-layernorm | SimpliPy | — | **TBD** | ~May 16 |

\* v23.0-20M trained on 2M expressions (its full training run); rebuttal-20M
   was a shorter 1M-expression run for the rebuttal.

---

## Notebook file

`experimental/eval/test_time_compute_scaling.ipynb` — all models have stubs.
The §8.1 training-scale plot requires a **separate cell** plotting
vNNR @ choices=32 vs log10(training expressions) — this does not exist yet
and needs to be added when the full set of A results is in.
