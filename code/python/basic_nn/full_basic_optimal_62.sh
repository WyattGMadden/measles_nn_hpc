#!/bin/bash

# Conda environment setup
eval "$(conda shell.bash hook)"
conda activate finalmlenv

# Directory containing the CSV files with hyperparameters
hyperparams_dir="../../../output/data/basic_nn_optimal/raytune_hp_optim/"

# Get the directory where the Bash script is running from
script_dir=$(pwd)

# Python script to run with optimal parameters
python_script_path="$script_dir/full_basic.py"

# Navigate to the directory with hyperparameter files
cd "$hyperparams_dir"

# Find all CSV files and process each one in the background
for file in raytune_hp_optim_k_*.csv; do
    if [[ -f "$file" ]]; then
        (
            # Extract 'k' value from the filename
            k=$(echo $file | grep -oP '(?<=k_)\d+')

            echo "Processing file: $file for k=$k"

            # Extract best hyperparameters from CSV, assuming the best trial is always in the second line after the header
            best_params=$(awk -F',' 'NR==2 && $3 == "True" {print "--num-hidden-layers="$14" --hidden-dim="$12" --lr="$10" --weight-decay="$13" --t-lag="$11}' $file)
            
            if [ -z "$best_params" ]; then
                echo "No best parameters found for k=$k in file $file."
                exit 1
            fi

            echo "Extracted parameters: $best_params"  # Debugging output

            # Return to the original script directory before running Python script
            cd "$script_dir"

            # Run the Python script with extracted parameters
            echo "Running model for k=$k with parameters: $best_params"
            python $python_script_path --k $k $best_params --save-model --num-epochs 200 --year-test-cutoff 62 --save-data-loc "../../../output/models/basic_nn_optimal_62/"
        ) &
    else
        echo "No files found to process."
        break
    fi
done

# Wait for all background processes to finish
wait

echo "All models processed."

