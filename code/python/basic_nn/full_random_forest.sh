#!/bin/bash

# Job name equivalent
job_name="randomforest100"
log_dir="./logs"
mkdir -p $log_dir  # Ensure log directory exists

timestamp=$(date +%Y%m%d-%H%M%S)

echo "Starting job: $job_name"

# Conda environment setup
eval "$(conda shell.bash hook)"
conda activate finalmlenv

# Define job indices
job_indices=(1 4 12 20 34 52)

# Maximum number of concurrent jobs
max_jobs=3
current_jobs=0

for SLURM_ARRAY_TASK_ID in "${job_indices[@]}"
do
    echo "Running task index: $SLURM_ARRAY_TASK_ID"
    python3 full_random_forest.py --n-estimators=100 \
        --save-model \
        --k=$SLURM_ARRAY_TASK_ID \
        --save-data-loc="../../../output/models/random_forest_100/" \
        --cases-data-loc="../../../data/data_from_measles_competing_risks/inferred_cases_urban.csv" \
        --pop-data-loc="../../../data/data_from_measles_competing_risks/inferred_pop_urban.csv" \
        --coords-data-loc="../../../data/data_from_measles_competing_risks/coordinates_urban.csv" \
        --susc-data-loc="../../../output/data/tsir_susceptibles/tsir_susceptibles.csv" \
        --birth-data-loc="../../../data/data_from_measles_competing_risks/ewBu4464.csv" \
        --year-test-cutoff=61 \
        --verbose &
    
    # Increment and manage job count
    ((current_jobs++))
    if [[ $current_jobs -ge $max_jobs ]]; then
        wait -n  # Wait for at least one job to finish before continuing
        ((current_jobs--))
    fi

    echo "Task $SLURM_ARRAY_TASK_ID started"
done

# Wait for all jobs to complete
wait
echo "All tasks completed"

