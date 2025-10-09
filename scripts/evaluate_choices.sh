# !/bin/bash

# Check if an argument was passed
if [ $# -eq 0 ]; then
    echo "Usage: run.sh <CONFIG>"
    exit 1
else
    CONFIG=$1
fi

# If two arguments were passed, store the second one in MODEL, otherwise set MODEL=CONFIG
if [ $# -eq 2 ]; then
    MODEL=$2
else
    MODEL=$CONFIG
fi

# 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
choices_values=(
    1 2 4 8 16 32 64 128 256 512
)

for choices in "${choices_values[@]}"; do
    flash_ansr evaluate \
        -c "{{ROOT}}/configs/${CONFIG}_choices/evaluation_choices_${choices}.yaml" \
        -m "{{ROOT}}/models/ansr-models/${MODEL}" \
        -d "{{ROOT}}/data/ansr-data/test_set/pool_15_10/dataset.yaml" \
        -n 256 \
        -o "{{ROOT}}/results/evaluation/${CONFIG}_choices/evaluation_choices_${choices}/pool_15_10.pickle" \
        -v
done