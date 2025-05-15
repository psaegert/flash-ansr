#!/bin/bash

# Get the first argument and store it in TEST_SET
if [ $# -eq 0 ]; then
    echo "Usage: evaluate_nesymres.sh <TEST_SET>"
    exit 1
else
    TEST_SET=$1
fi

flash_ansr evaluate-nesymres \
    -c "{{ROOT}}/configs/nesymres-100M/evaluation.yaml" \
    -ce "{{ROOT}}/configs/nesymres-100M/eq_config.json" \
    -cm "{{ROOT}}/configs/nesymres-100M/config.yaml" \
    -e "{{ROOT}}/configs/test_set/expression_space.yaml" \
    -m "{{ROOT}}/models/nesymres/100M.ckpt" \
    -d "{{ROOT}}/data/ansr-data/test_set/$TEST_SET/dataset.yaml" \
    -n 1000 \
    -o {{ROOT}}/results/evaluation/nesymres-100M/$TEST_SET.pickle \
    -v