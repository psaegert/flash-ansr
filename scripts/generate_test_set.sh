# !/bin/bash

# Check if an argument was passed
if [ $# -eq 0 ]; then
    echo "Usage: run.sh <CONFIG>"
    exit 1
else
    CONFIG=$1
fi

flash_ansr generate-skeleton-pool -c {{ROOT}}/configs/${CONFIG}/skeleton_pool.yaml -e "{{ROOT}}/configs/test_set/simplipy_engine.yaml" -o {{ROOT}}/data/ansr-data/test_set/${CONFIG}/skeleton_pool -s 5000 -v