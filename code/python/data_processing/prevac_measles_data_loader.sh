#!/bin/bash

# Loading Conda environment
eval "$(conda shell.bash hook)"
conda activate ml_dl_env

# Assuming you want to run the script for an array job manually
# Here we simulate the array job locally by iterating over a range
for k in $(seq 1 4 12 20 34 52); do
    python3 prevac_measles_data_loader.py \
        --k=${k} \
        --t-lag=130 \
        --test-size=0.3 \
        --susc-data-loc="../../../data/tsir_susceptibles/tsir_susceptibles.csv" \
        --birth-data-loc="../../../data/births/ewBu4464.csv" \
        --write-to-file \
        --write-loc= "../../output/data/basic_nn/train_test_k_tlag/" \
        --include-nbc-cases \
        --verbose
done

