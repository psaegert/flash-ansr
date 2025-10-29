# !/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model> <config_dir> <test_set> <max_n_support>"
    exit 1
fi

MODEL=$1
CONFIG_DIR=$2
TEST_SET=$3
MAX_N_SUPPORT=$4

n_support_values=(
    1 2 4 8 16 32 64 128 256 512 1024
)

for n_support in "${n_support_values[@]}"; do
    if [ "$n_support" -gt "$MAX_N_SUPPORT" ]; then
        break
    fi
    flash_ansr evaluate \
        -c "{{ROOT}}/configs/evaluation/${CONFIG_DIR}/evaluation_n_support_${n_support}.yaml" \
        -m "{{ROOT}}/models/ansr-models/${MODEL}" \
        -d "{{ROOT}}/data/ansr-data/test_set/${TEST_SET}/dataset.yaml" \
        -o "{{ROOT}}/results/evaluation/${MODEL}_n_support/evaluation_n_support_${n_support}/${TEST_SET}.pickle" \
        -n 2048 \
        -v
done