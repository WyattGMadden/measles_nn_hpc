#!/bin/bash
#SBATCH --job-name=prophet_fit
#SBATCH --output=/path/to/output/logs/prophet_city_%A_%a.out
#SBATCH --error=/path/to/output/logs/prophet_city_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3
# #SBATCH --array=1-1459

module load R/4.2.2

Rscript -e "source('/path/to/your/script/prophet_functions.R'); fit_prophet(${SLURM_ARRAY_TASK_ID})"

