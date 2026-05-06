#!/bin/bash
# Run the full set of evaluations required for the camera-ready paper.
#
#   Phase 1: candidate-distribution probes at c=4096 (§8.1 main-text table).
#            Delegates to scripts/run_candidate_probe.sh; resume-safe via
#            existing-output checks.
#
#   Phase 2: standard fastsrb scaling evals filling in the gaps for the
#            per-track appendix plots. Each YAML carries `resume: true`, so
#            re-running an already-complete config is a no-op. Configs whose
#            model checkpoints are not yet on disk (Y10 still training on
#            Valkyrie; B-track on the A100 cluster) are skipped with a
#            message --- re-run this script after each new checkpoint lands.
#
# Usage:  ./scripts/run_full_evaluation.sh
# Resume: yes (idempotent; safe to re-run).

set -e
ROOT=$(pwd)

run_eval_if_model_exists() {
    local config_path="$1"
    local model_dir="$2"
    if [ ! -d "$model_dir" ]; then
        echo "[skip] ${config_path}: model not found at ${model_dir}"
        return 0
    fi
    echo ""
    echo "=== flash_ansr evaluate-run -c ${config_path} ==="
    flash_ansr evaluate-run -c "$config_path" -v
}

echo ""
echo "================================================================"
echo " Phase 1: candidate-distribution probes (§8.1 main-text table)"
echo "================================================================"
bash "${ROOT}/scripts/run_candidate_probe.sh"

echo ""
echo "================================================================"
echo " Phase 2: standard fastsrb scaling evals (appendix per-track plots)"
echo "================================================================"

# --- A-track ---------------------------------------------------------------
# S10, S100, U10 already cover c up to 8192 (~10-20s).
# U100 has c=8192 in its config but the run is still pending.
run_eval_if_model_exists \
    "configs/evaluation/scaling/v23.0-20M-A-U100_fastsrb.yaml" \
    "${ROOT}/models/ansr-models/v23.0-20M-A-U100"

# Y1 already covers up to c=1024 (already past 20s with SymPy at training scale).
# Y10 will skip until its checkpoint arrives from Valkyrie (~May 8); re-run
# this script once the model lands.
run_eval_if_model_exists \
    "configs/evaluation/scaling/v23.0-20M-A-Y10_fastsrb.yaml" \
    "${ROOT}/models/ansr-models/v23.0-20M-A-Y10"

# --- Z-track ---------------------------------------------------------------
# 120M baseline + Z2-beam already cover the full compute range.
# Z1-bfgs has c=8192 in its config but only ran up to c=2048; the BFGS refiner
# is ~30% slower than LM, so c=8192 is needed to reach the 10-20s range.
# Z1-bfgs uses the same v23.0-120M checkpoint.
run_eval_if_model_exists \
    "configs/evaluation/scaling/v23.0-120M-Z1-bfgs_fastsrb.yaml" \
    "${ROOT}/models/ansr-models/v23.0-120M"

# --- B-track ---------------------------------------------------------------
# All three architecture variants are still training on the A100 cluster; they
# will skip cleanly here until their checkpoints land in models/ansr-models/.
# Re-run this script after each B-variant arrives.
for variant in B1-postnorm B2-16bit B4-layernorm; do
    run_eval_if_model_exists \
        "configs/evaluation/scaling/v23.0-120M-${variant}_fastsrb.yaml" \
        "${ROOT}/models/ansr-models/v23.0-120M-${variant}"
done

echo ""
echo "================================================================"
echo " All evaluations complete (skipping any pending-checkpoint rows)."
echo " Sync results back via scripts/get_results.sh."
echo "================================================================"
