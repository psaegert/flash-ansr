#!/bin/bash

set -euo pipefail

# Check if an argument was passed
if [ $# -eq 0 ]; then
    echo "Usage: train.sh <CONFIG> [extra flash_ansr train args...]"
    echo "  e.g. train.sh v23.0-120M-B2-16bit --resume-from models/ansr-models/v23.0-120M-B2-16bit/checkpoint_2000000"
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
    echo "train.sh: Using OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    cmd+=(-w "${OMP_NUM_THREADS}")
fi

"${cmd[@]}" "${@:2}"