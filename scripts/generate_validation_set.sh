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

flash_ansr generate-skeleton-pool -c {{ROOT}}/configs/${CONFIG}/skeleton_pool_val.yaml -o {{ROOT}}/data/ansr-data/${CONFIG}/skeleton_pool_val -s 1000 -v