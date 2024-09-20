#!/bin/bash

# Job name equivalent
job_name="dataprocessexplain"

timestamp=$(date +%Y%m%d-%H%M%S)

echo "Starting job: $job_name" 

# Conda environment setup
eval "$(conda shell.bash hook)"
conda activate finalmlenv

# job_indices=({1..10} {12..52..2})
job_indices=(52)

# Maximum number of concurrent jobs
max_jobs=1
current_jobs=0

for SLURM_ARRAY_TASK_ID in "${job_indices[@]}"
do
    echo "$(date "+%Y-%m-%d %H:%M:%S") - Running task index: $SLURM_ARRAY_TASK_ID" 
    python3 data_process_explain.py \
        --k=$SLURM_ARRAY_TASK_ID \
        --t-lag=130 \
        --save-data-loc="../../../../output/data/basic_nn_optimal/explain/" \
        --cases-data-loc="../../../../output/data/basic_nn_optimal/prefit/" \
        --year-test-cutoff=61 \
    
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

