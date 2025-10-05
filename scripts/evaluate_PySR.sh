#!/bin/bash

# Get the first argument and store it in TEST_SET
if [ $# -eq 0 ]; then
    echo "Usage: evaluate_pysr.sh <TEST_SET>"
    exit 1
else
    TEST_SET=$1
fi

flash_ansr evaluate-pysr \
    -c "{{ROOT}}/configs/PySR-default/evaluation.yaml" \
    -d "{{ROOT}}/data/ansr-data/test_set/$TEST_SET/dataset.yaml" \
    -e "dev_7-3" \
    -n 1000 \
    -o {{ROOT}}/results/evaluation/PySR-default/$TEST_SET.pickle \
    -v