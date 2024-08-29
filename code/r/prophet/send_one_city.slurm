#!/bin/bash
#SBATCH --job-name=prophet_fit
#SBATCH --partition=lau
#SBATCH --ntasks=1
#SBATCH --output=../../../output/data/prophet/logs/prophet_city_%A.out
#SBATCH --error=../../../output/data/prophet/logs/prophet_city_%A.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Load the R module
module load R/4.2.2

# The city index is passed as the first command line argument
Rscript -e "source('prophet_uk_run.R'); fit_prophet($1)"

