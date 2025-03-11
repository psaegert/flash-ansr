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

# flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/soose_nc/nc.csv" -p "soose" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/soose_nc/skeleton_pool" -v
# flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/feynman/FeynmanEquations.csv" -p "feynman" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/feynman/skeleton_pool" -v
# flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/nguyen/nguyen.csv" -p "nguyen" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/nguyen/skeleton_pool" -v

# flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/pool_15/dataset.yaml" -n 1000 -o {{ROOT}}/results/evaluation/${CONFIG}/pool_15.pickle -v
# flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/soose_nc/dataset.yaml" -n 1000 -o {{ROOT}}/results/evaluation/${CONFIG}/soose_nc.pickle -v
flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/feynman/dataset.yaml" -n 1000 -o {{ROOT}}/results/evaluation/${CONFIG}/feynman.pickle -v
# flash_ansr evaluate -c {{ROOT}}/configs/${CONFIG}/evaluation.yaml -m "{{ROOT}}/models/ansr-models/${MODEL}" -d "{{ROOT}}/data/ansr-data/test_set/nguyen/dataset.yaml" -n 1000 -o {{ROOT}}/results/evaluation/${CONFIG}/nguyen.pickle -v