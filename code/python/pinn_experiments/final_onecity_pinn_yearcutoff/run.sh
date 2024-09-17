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
k_values=(52)
#tlag_values=(26 52 78 104 130)
tlag_values=(130 104 78 52 26)
#tlag_values=(130)
#cities=("London" "Manchester" "Birmingham" "Liverpool")
cities=("Manchester" "Birmingham" "Liverpool")

# Directory to save outputs
write_loc="../../../../output/models/pinn_experiments/final_onecity_pinn_yearcutoff/"

# Maximum number of concurrent jobs
MAX_JOBS=4
current_jobs=0

# Iterate over each city, k, and tlag
for city in "${cities[@]}"; do
    for k in "${k_values[@]}"; do
        for tlag in "${tlag_values[@]}"; do
            if [ "$k" -lt "$tlag" ]; then  # Only run scripts if k is less than tlag
                # Run naivepinn.py
                echo "Running naivepinn.py for city=$city, k=$k and tlag=$tlag"
                python3 "$SCRIPT_DIR/naivepinn.py" --city "$city" --k $k --tlag $tlag --write-loc "$write_loc" --year-test-cutoff 61 --num-epochs 2000 --wd-fnn 0.025 &
                current_jobs=$((current_jobs+1))

                # Run tsirpinn.py
                echo "Running tsirpinn.py for city=$city, k=$k and tlag=$tlag"
                python3 "$SCRIPT_DIR/tsirpinn.py" --city "$city" --k $k --tlag $tlag --write-loc "$write_loc" --year-test-cutoff 61 --num-epochs 2000 --wd-fnn 0.025 &
                current_jobs=$((current_jobs+1))

                # Check if we need to wait for jobs to finish
                if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
                    wait  # Wait for all background jobs to finish
                    current_jobs=0
                fi
            else
                echo "Skipping run for city=$city, k=$k, tlag=$tlag as k is not less than tlag"
            fi
        done
    done
done

wait # Wait for the last batch of background jobs to finish
echo "All models processed."

