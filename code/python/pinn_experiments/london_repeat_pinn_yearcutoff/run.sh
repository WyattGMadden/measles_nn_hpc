#!/bin/bash

# Define the conda environment
ENV_NAME="finalmlenv"

# Activate the environment
echo "Activating the conda environment: $ENV_NAME"
source activate $ENV_NAME

# Define the script directory
SCRIPT_DIR=$(pwd)

# Define the values for k and tlag
#k_values=(1 4 12 20 34 52)
#k_values=(52 34 20 12 4 1)

# Directory to save outputs
write_loc="../../../../output/models/pinn_experiments/london_repeat_pinn_yearcutoff/"

# Maximum number of concurrent jobs
MAX_JOBS=4
current_jobs=0

# 10 iterations
for ((i=1; i<=100; i++)); do
    echo "Iteration $i"
    # Run naivepinn.py
    echo "Running naivepinn.py for city=$city, k=$k and tlag=$tlag"
    python3 "$SCRIPT_DIR/naivepinn.py" --run-num $i --city "London" --k 52 --tlag 104 --write-loc "$write_loc" --year-test-cutoff 61 --num-epochs 2500 --wd-fnn 0.025 &
    current_jobs=$((current_jobs+1))

    # Run tsirpinn.py
    echo "Running tsirpinn.py for city=$city, k=$k and tlag=$tlag"
    python3 "$SCRIPT_DIR/tsirpinn.py" --run-num $i --city "London" --k 52 --tlag 104 --write-loc "$write_loc" --year-test-cutoff 61 --num-epochs 2500 --wd-fnn 0.025 &
    current_jobs=$((current_jobs+1))

    # Check if we need to wait for jobs to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait  # Wait for all background jobs to finish
        current_jobs=0
    fi
done

wait # Wait for the last batch of background jobs to finish
echo "All models processed."

