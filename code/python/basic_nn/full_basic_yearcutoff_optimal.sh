#!/bin/bash

# Conda environment setup
eval "$(conda shell.bash hook)"
conda activate finalmlenv

# Directory containing the CSV files with hyperparameters
hyperparams_dir="../../../output/figures/basic_nn/raytune_hp_optim/"

# Get the directory where the Bash script is running from
script_dir=$(pwd)

# Python script to run with optimal parameters
python_script_path="$script_dir/full_basic.py"

# Navigate to the directory with hyperparameter files
cd "$hyperparams_dir"

# Find all CSV files and process each one
for file in raytune_hp_optim_k_*.csv; do
    if [[ -f "$file" ]]; then
        # Extract 'k' value from the filename
        k=$(echo $file | grep -oP '(?<=k_)\d+')

        echo "Processing file: $file for k=$k"

        # Extract best hyperparameters from CSV, assuming the best trial is always in the second line after the header
        best_params=$(awk -F',' 'NR==2 && $3 == "True" {print "--num-hidden-layers="$18" --hidden-dim="$16" --lr="$14" --weight-decay="$17" --t-lag="$15}' $file)
        
        if [ -z "$best_params" ]; then
            echo "No best parameters found for k=$k in file $file."
            continue
        fi

        echo "Extracted parameters: $best_params"  # Debugging output

        # Return to the original script directory
        cd "$script_dir"

        # Run the Python script with extracted parameters
        echo "Running model for k=$k with parameters: $best_params"
        python $python_script_path --k $k $best_params --save-data-loc "./output/"

        # Navigate back to the hyperparameters directory to continue processing
        cd "$hyperparams_dir"
    else
        echo "No files found to process."
        break
    fi
done

# Return to the original directory just in case
cd "$script_dir"

echo "All models processed."

