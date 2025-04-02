# !/bin/bash

# Check if an argument was passed
if [ $# -eq 0 ]; then
    echo "Usage: run.sh <CONFIG>"
    exit 1
else
    CONFIG=$1
fi

echo "Finding simplifications with ${CONFIG}"

flash_ansr find-simplifications -e "{{ROOT}}/configs/${CONFIG}/expression_space.yaml" -n 10000 -t 10800 -m 5 -r 5 -s 1000 -o {{ROOT}}/data/ansr-data/simplification_rules/${CONFIG}.json --reset-rules -v