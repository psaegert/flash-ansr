#!/bin/bash
# Run the candidate-distribution probe across system-level configurations.
#
# Two probe sweeps:
#  (A) Training-data axis (§8.1): S100 (simplify=true), U100 (simplify=false).
#      Add Y10 once the A_Y10 checkpoint is available.
#  (Z) Inference-strategy axis (§8.3): the same 120M baseline checkpoint with
#      softmax_sampling and beam_search decoders, for matched-axis diversity
#      comparison.
#
# Choices/beam_width points: {64, 256, 2048}. Resume-safe.

set -e

ROOT=$(pwd)
PROBE="${ROOT}/experimental/system_effect/candidate_distribution_probe.py"
SOURCE_PKL_TEMPLATE="${ROOT}/results/evaluation/scaling/__MODEL__/fastsrb/choices_00256.pkl"
N_SAMPLES=300

mkdir -p "${ROOT}/results/system_effect"

run_probe() {
    local label="$1"
    local model="$2"
    local simplify="$3"
    local decoder="$4"
    local source_pkl="$5"

    if [ ! -f "$source_pkl" ]; then
        echo "WARNING: source pickle ${source_pkl} not found, skipping ${label}"
        return 0
    fi

    for choices in 64 256 2048; do
        local choices_padded
        choices_padded=$(printf "%05d" "$choices")
        local out="${ROOT}/results/system_effect/cand_dist_${label}_choices${choices_padded}_n${N_SAMPLES}.pkl"
        if [ -f "$out" ]; then
            echo "=== SKIP ${out} (exists) ==="
            continue
        fi
        echo ""
        echo "=== ${label} (model=${model} decoder=${decoder} simplify=${simplify} choices=${choices}) ==="
        python "$PROBE" \
            --model-path "${ROOT}/models/ansr-models/${model}" \
            --decoder "$decoder" \
            --simplify "$simplify" \
            --choices "$choices" \
            --n-samples "$N_SAMPLES" \
            --source-pkl "$source_pkl" \
            --output "$out"
    done
}

# --- (A) Training-data axis: §8.1 -------------------------------------------
# S100 / U100 (system-level treatment: training and inference simplify match).
# Add: run_probe "v23.0-20M-A-Y10"  ... once Y10 checkpoint is available.

run_probe "v23.0-20M-A-S100" \
          "v23.0-20M-A-S100" \
          "true" \
          "softmax_sampling" \
          "${SOURCE_PKL_TEMPLATE/__MODEL__/v23.0-20M-A-S100}"

run_probe "v23.0-20M-A-U100" \
          "v23.0-20M-A-U100" \
          "false" \
          "softmax_sampling" \
          "${SOURCE_PKL_TEMPLATE/__MODEL__/v23.0-20M-A-U100}"

# --- (Z) Inference-strategy axis: §8.3 --------------------------------------
# Same 120M baseline checkpoint, two decoders. Source pickle is the existing
# v23.0-120M fastsrb eval (choices=256). simplify is recorded for parity but
# only affects the softmax_sampling code path; beam_search does not run the
# post-processing simplify step.

run_probe "v23.0-120M-softmax" \
          "v23.0-120M" \
          "true" \
          "softmax_sampling" \
          "${SOURCE_PKL_TEMPLATE/__MODEL__/v23.0-120M}"

run_probe "v23.0-120M-beam" \
          "v23.0-120M" \
          "true" \
          "beam_search" \
          "${SOURCE_PKL_TEMPLATE/__MODEL__/v23.0-120M}"

# --- 10s inference-time anchor for the 20M models (main-text table) ----------
# 20M S100/U100 reach ~5s at c=2048; we extend to c=4096 to land at ~10s,
# matching the standard test-time-compute scaling evaluation's ~10s target.
# 120M models already hit ~10s at c=2048 so they are not extended here.

run_extra_choices() {
    # Single-config-and-choice probe; smaller 'run_probe' for ad-hoc additions.
    local label="$1"
    local model="$2"
    local simplify="$3"
    local choices="$4"
    local source_pkl="$5"

    if [ ! -f "$source_pkl" ]; then
        echo "WARNING: source pickle ${source_pkl} not found, skipping ${label} c=${choices}"
        return 0
    fi

    local choices_padded
    choices_padded=$(printf "%05d" "$choices")
    local out="${ROOT}/results/system_effect/cand_dist_${label}_choices${choices_padded}_n${N_SAMPLES}.pkl"
    if [ -f "$out" ]; then
        echo "=== SKIP ${out} (exists) ==="
        return 0
    fi
    echo ""
    echo "=== ${label} (c=${choices}, 10s inference-time anchor) ==="
    python "$PROBE" \
        --model-path "${ROOT}/models/ansr-models/${model}" \
        --decoder softmax_sampling \
        --simplify "$simplify" \
        --choices "$choices" \
        --n-samples "$N_SAMPLES" \
        --source-pkl "$source_pkl" \
        --output "$out"
}

run_extra_choices "v23.0-20M-A-S100" \
                  "v23.0-20M-A-S100" \
                  "true" \
                  4096 \
                  "${SOURCE_PKL_TEMPLATE/__MODEL__/v23.0-20M-A-S100}"

run_extra_choices "v23.0-20M-A-U100" \
                  "v23.0-20M-A-U100" \
                  "false" \
                  4096 \
                  "${SOURCE_PKL_TEMPLATE/__MODEL__/v23.0-20M-A-U100}"
