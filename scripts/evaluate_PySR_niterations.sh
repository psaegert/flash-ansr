#!/bin/bash

# Get the first argument and store it in TEST_SET
if [ $# -eq 0 ]; then
    echo "Usage: evaluate_PySR_niterations.sh <TEST_SET>"
    exit 1
else
    TEST_SET=$1
fi

# 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768

niterations_values=(
    1 2 4 8 16 32 64
)

for niterations in "${niterations_values[@]}"; do
    output_dir="{{ROOT}}/results/evaluation/PySR_niterations/evaluation_niterations_${niterations}"

    flash_ansr evaluate-pysr \
        -c "{{ROOT}}/configs/PySR_niterations/evaluation_niterations_${niterations}.yaml" \
        -d "{{ROOT}}/data/ansr-data/test_set/$TEST_SET/dataset.yaml" \
        -e "dev_7-3" \
        -n 512 \
        -o "$output_dir/$TEST_SET.pickle" \
        -v
done