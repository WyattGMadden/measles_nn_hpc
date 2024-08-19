#!/bin/bash

# Job name equivalent
job_name="testbasicnn"
log_dir="./logs"
mkdir -p $log_dir  # Ensure log directory exists

timestamp=$(date +%Y%m%d-%H%M%S)

echo "Starting job: $job_name"

# Conda environment setup
eval "$(conda shell.bash hook)"
conda activate finalmlenv

# Define job indices
#job_indices=({26} {12..26..2})
#job_indices=({1..2})
#job_indices=({50..52})
# just 26 and 52
#job_indices=({26..52..26})
#just 52
job_indices=(1 4 12 20 34)

# Maximum number of concurrent jobs
max_jobs=2
current_jobs=0

for SLURM_ARRAY_TASK_ID in "${job_indices[@]}"
do
    echo "Running task index: $SLURM_ARRAY_TASK_ID"
    python3 full_random_forest.py --n-estimators=200 \
        --save-model \
        --k=$SLURM_ARRAY_TASK_ID \
        --save-data-loc="../../../output/models/random_forest_100/" \
        --susc-data-loc="../../../data/tsir_susceptibles/tsir_susceptibles.csv" \
        --birth-data-loc="../../../data/births/ewBu4464.csv" \
        --test-size=0.251197 \
        --verbose 
    
    # Increment and manage job count
    ((current_jobs++))
    if [[ $current_jobs -ge $max_jobs ]]; then
        wait -n
        ((current_jobs--))
    fi

    echo "Task $SLURM_ARRAY_TASK_ID started"
done

# Wait for all jobs to complete
wait
echo "All tasks completed"

