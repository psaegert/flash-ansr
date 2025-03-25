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

echo "Running checkpoint analysis with config ${CONFIG} and model ${MODEL}"

flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/soose_nc/nc.csv" -p "soose" -e "{{ROOT}}/configs/test_set/expression_space.yaml" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/soose_nc/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/feynman/FeynmanEquations.csv" -p "feynman" -e "{{ROOT}}/configs/test_set/expression_space.yaml" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/feynman/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/nguyen/nguyen.csv" -p "nguyen" -e "{{ROOT}}/configs/test_set/expression_space.yaml" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/nguyen/skeleton_pool" -v

for CKPT in {100000..1500000..100000}; do
    echo "Processing checkpoint $CKPT"

    for DATASET in soose_nc feynman nguyen pool_15; do
        flash_ansr evaluate -c "{{ROOT}}/configs/${CONFIG}/evaluation.yaml" \
                            -m "{{ROOT}}/models/ansr-models/${MODEL}/checkpoint_${CKPT}" \
                            -d "{{ROOT}}/data/ansr-data/test_set/$DATASET/dataset.yaml" \
                            -n 1000 \
                            -o "{{ROOT}}/results/evaluation/analysis_checkpoints_${MODEL}/${CONFIG}_checkpoint-${CKPT}/$DATASET.pickle" \
                            -v
    done

    flash_ansr evaluate -c "{{ROOT}}/configs/${CONFIG}/evaluation.yaml" \
                        -m "{{ROOT}}/models/ansr-models/${MODEL}/checkpoint_${CKPT}" \
                        -d "{{ROOT}}/configs/${CONFIG}/dataset_val.yaml" \
                        -n 1000 \
                        -o "{{ROOT}}/results/evaluation/analysis_checkpoints_${MODEL}/${CONFIG}_checkpoint-${CKPT}/val.pickle" \
                        -v

    flash_ansr evaluate -c "{{ROOT}}/configs/${CONFIG}/evaluation.yaml" \
                        -m "{{ROOT}}/models/ansr-models/${MODEL}/checkpoint_${CKPT}" \
                        -d "{{ROOT}}/configs/${CONFIG}/dataset_train.yaml" \
                        -n 1000 \
                        -o "{{ROOT}}/results/evaluation/analysis_checkpoints_${MODEL}/${CONFIG}_checkpoint-${CKPT}/train.pickle" \
                        -v

    echo "Finished processing checkpoint $CKPT"
done