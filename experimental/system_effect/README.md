# System-effect analysis (§8.1)

Scripts and dumps for the SimpliPy vs SymPy vs Unsimplified system-effect
analysis. Companion to [agent/system-effect-analysis-plan.md](../../agent/system-effect-analysis-plan.md).

## Scripts

| Script | Runs on | Purpose |
|---|---|---|
| `training_data_analysis.py` | CPU | Sample N skeletons from each pool; compare length distributions, operator-frequency drift, canonical compression. |
| `candidate_distribution_probe.py` | GPU | For each fastsrb test point, dump the full post-refinement candidate set for a single (model, simplify) configuration. |
| `analyze_candidates.py` | CPU | Summarize one or more candidate dumps with bootstrap CIs and paired stats. |

The probe is parameterized by ``--model-path``, ``--simplify``, ``--choices``,
and ``--n-samples``; it sources the (X, y) test data from a previously-evaluated
fastsrb pickle (any choices point works — choices=256 is the default for
sourcing because every model has it).

## Driver

`scripts/run_candidate_probe.sh` runs the probe across the system-level
configurations (S100, U100; Y10 once available) at choices ∈ {64, 256, 2048},
N=300. Outputs go to `results/system_effect/`. Resume-safe (skips outputs that
already exist).

## Dumps

All dumps land under `results/system_effect/` (gitignored). Naming:

```
results/system_effect/cand_dist_<model>_choices<NNNNN>_n<N>.pkl
results/system_effect/training_data_summary_n<N>.pkl
```
