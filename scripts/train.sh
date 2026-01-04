#!/bin/bash

set -euo pipefail

# Check if an argument was passed
if [ $# -eq 0 ]; then
    echo "Usage: train.sh <CONFIG>"
    exit 1
else
    CONFIG=$1
fi

cmd=(
    flash_ansr train
    -c "{{ROOT}}/configs/${CONFIG}/train.yaml"
    -o "{{ROOT}}/models/ansr-models/${CONFIG}"
    -v
    -ci 250000
    -vi 10000
)

if [ -n "${OMP_NUM_THREADS:-}" ]; then
    echo "Using OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    cmd+=(-w "${OMP_NUM_THREADS}")
fi

"${cmd[@]}"