#!/bin/bash

# Conda environment setup
eval "$(conda shell.bash hook)"
conda activate finalmlenv

# Define an array of 'k' values
k_values=(1 4 12 20 34 52)

# Loop over each value of 'k' and run the Python script
for k in "${k_values[@]}"; do
    echo "Running hyperparameter tuning for k=$k"
    python full_basic_raytune.py --k $k --num-samples 20 --max-num-epochs 20 --gpus-per-trial 1
done

echo "All tuning processes are complete."

