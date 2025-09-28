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

echo "Running ${CONFIG} with model ${MODEL}"

flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/soose_nc/nc.csv" -p "soose" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/soose_nc/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/soose_nc/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/feynman/FeynmanEquations.csv" -p "feynman" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/feynman/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/feynman/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/nguyen/nguyen.csv" -p "nguyen" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/nguyen/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/nguyen/skeleton_pool" -v

flash_ansr generate-skeleton-pool -c {{ROOT}}/configs/${CONFIG}/skeleton_pool_val.yaml -o {{ROOT}}/data/ansr-data/${CONFIG}/skeleton_pool_val -s 5000 -v

flash_ansr dino -c {{ROOT}}/configs/${CONFIG}/train.yaml -o {{ROOT}}/models/ansr-models/${CONFIG} -v -ci 250000 -vi 50000