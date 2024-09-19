#!/bin/bash

# Job name equivalent
job_name="explain"

timestamp=$(date +%Y%m%d-%H%M%S)

echo "Starting job: $job_name" 

# Conda environment setup
eval "$(conda shell.bash hook)"
conda activate finalmlenv

#job_indices=({1..10} {15..50..5})
#job_indices=({1..10} {12..52..2})
#job_indices=(1 4 12 20 34)
job_indices=(52)

# Maximum number of concurrent jobs
max_jobs=1
current_jobs=0

for SLURM_ARRAY_TASK_ID in "${job_indices[@]}"
do
    echo "$(date "+%Y-%m-%d %H:%M:%S") - Running task index: $SLURM_ARRAY_TASK_ID" 
    python3 basic_nn_explain.py \
        --k=$SLURM_ARRAY_TASK_ID \
        --t-lag=130 \
        --hidden-dim=721 \
        --num-hidden-layers=2 \
        --data-read-loc="../../../../output/data/basic_nn_optimal/explain/" \
        --write-data-loc="../../../../output/data/basic_nn_optimal/explain/" \
        --model-read-loc="../../../../output/models/basic_nn_optimal/" \
        --susc-data-loc="../../../../output/data/tsir_susceptibles/tsir_susceptibles.csv" \
        --birth-data-loc="../../../../data/data_from_measles_competing_risks/ewBu4464.csv" \
        --verbose 
    
    # Increment and manage job count
    ((current_jobs++))
    if [[ $current_jobs -ge $max_jobs ]]; then
        wait -n
        ((current_jobs--))
    fi

    echo "$(date "+%Y-%m-%d %H:%M:%S") - Task $SLURM_ARRAY_TASK_ID started"
done

# Wait for all jobs to complete
wait
echo "$(date "+%Y-%m-%d %H:%M:%S") - All tasks completed"

