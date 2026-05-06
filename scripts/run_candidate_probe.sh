#!/bin/bash
# Run the candidate-distribution probe across system-level configurations.
#
# Pipelines: S100 (simplify=true), U100 (simplify=false). Add Y10 to MODELS
# once the A_Y10 checkpoint is available — its system-level treatment is
# simplify=true.
#
# Choices points: {64, 256, 2048}. Resume-safe (skips outputs that exist).

set -e

ROOT=$(pwd)
PROBE="${ROOT}/experimental/system_effect/candidate_distribution_probe.py"
SOURCE_PKL_TEMPLATE="${ROOT}/results/evaluation/scaling/__MODEL__/fastsrb/choices_00256.pkl"

declare -A SIMPLIFY
SIMPLIFY[v23.0-20M-A-S100]=true
SIMPLIFY[v23.0-20M-A-U100]=false
SIMPLIFY[v23.0-20M-A-Y10]=true

MODELS=( v23.0-20M-A-S100 v23.0-20M-A-U100 )
CHOICES_LIST=( 64 256 2048 )
N_SAMPLES=300

mkdir -p "${ROOT}/results/system_effect"

for model in "${MODELS[@]}"; do
    simplify="${SIMPLIFY[${model}]}"
    if [ -z "$simplify" ]; then
        echo "ERROR: no simplify flag configured for ${model}"
        exit 1
    fi

    source_pkl="${SOURCE_PKL_TEMPLATE/__MODEL__/${model}}"
    if [ ! -f "$source_pkl" ]; then
        echo "WARNING: source pickle ${source_pkl} not found, skipping ${model}"
        continue
    fi

    for choices in "${CHOICES_LIST[@]}"; do
        choices_padded=$(printf "%05d" "$choices")
        out="${ROOT}/results/system_effect/cand_dist_${model}_choices${choices_padded}_n${N_SAMPLES}.pkl"
        if [ -f "$out" ]; then
            echo "=== SKIP ${out} (exists) ==="
            continue
        fi
        echo ""
        echo "=== ${model} simplify=${simplify} choices=${choices} ==="
        python "$PROBE" \
            --model-path "${ROOT}/models/ansr-models/${model}" \
            --simplify "$simplify" \
            --choices "$choices" \
            --n-samples "$N_SAMPLES" \
            --source-pkl "$source_pkl" \
            --output "$out"
    done
done
