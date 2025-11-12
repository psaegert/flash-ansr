# ./scripts/evaluate_choices.sh v22.0-60M v22.x 256 pool_15_10

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model> <config_dir> <choices> <test_set>"
    exit 1
fi

MODEL=$1
CONFIG_DIR=$2
CHOICES=$3
TEST_SET=$4

flash_ansr evaluate \
        -c "{{ROOT}}/configs/evaluation/${CONFIG_DIR}/evaluation_${CHOICES}.yaml" \
        -m "{{ROOT}}/models/ansr-models/${MODEL}" \
        -d "{{ROOT}}/data/ansr-data/test_set/${TEST_SET}/dataset.yaml" \
        -o "{{ROOT}}/results/evaluation/${MODEL}/evaluation_${CHOICES}/${TEST_SET}.pickle" \
        -n 8192 \
        -s 100 \
        -v