# !/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model> <config_dir> <test_set>"
    exit 1
fi

MODEL=$1
CONFIG_DIR=$2
TEST_SET=$3

choices_values=(
    1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
)

for choices in "${choices_values[@]}"; do
    flash_ansr evaluate \
        -c "{{ROOT}}/configs/evaluation/${CONFIG_DIR}/evaluation_choices_${choices}.yaml" \
        -m "{{ROOT}}/models/ansr-models/${MODEL}" \
        -d "{{ROOT}}/data/ansr-data/test_set/${TEST_SET}/dataset.yaml" \
        -o "{{ROOT}}/results/evaluation/${MODEL}_choices/evaluation_choices_${choices}/${TEST_SET}.pickle" \
        -n 2048 \
        -v
done