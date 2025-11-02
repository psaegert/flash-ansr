# !/bin/bash

# Check if an argument was passed
if [ $# -eq 0 ]; then
    echo "Usage: run.sh <CONFIG>"
    exit 1
else
    CONFIG=$1
fi

flash_ansr train -c {{ROOT}}/configs/${CONFIG}/train.yaml -o {{ROOT}}/models/ansr-models/${CONFIG} -v -ci 250000 -vi 10000