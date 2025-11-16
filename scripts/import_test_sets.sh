# !/bin/bash

# flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/soose_nc/nc.csv" -p "soose" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/soose_nc/skeleton_pool" -v
# flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/feynman/FeynmanEquations.csv" -p "feynman" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/feynman/skeleton_pool" -v
# flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/nguyen/nguyen.csv" -p "nguyen" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/nguyen/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/fastsrb/expressions.yaml" -p "fastsrb" -e "dev_7-3" -b "{{ROOT}}/configs/test_set/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/fastsrb/skeleton_pool" -v
