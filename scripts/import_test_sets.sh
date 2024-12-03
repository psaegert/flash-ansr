# !/bin/bash

flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/soose_nc/nc.csv" -p "soose" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/soose_nc/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/feynman/FeynmanEquations.csv" -p "feynman" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/feynman/skeleton_pool" -v
flash_ansr import-data -i "{{ROOT}}/data/ansr-data/test_set/nguyen/nguyen.csv" -p "nguyen" -e "{{ROOT}}/configs/test_set_base/expression_space.yaml" -b "{{ROOT}}/configs/test_set_base/skeleton_pool.yaml" -o "{{ROOT}}/data/ansr-data/test_set/nguyen/skeleton_pool" -v
