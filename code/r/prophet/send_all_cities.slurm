#!/bin/bash
#SBATCH --job-name=prophet_fit
#SBATCH --partition=lau
#SBATCH --ntasks=1
#SBATCH --output=../../../output/data/prophet/logs/prophet_city_%A_%a.out
#SBATCH --error=../../../output/data/prophet/logs/prophet_city_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-1451

module load R/4.2.2

Rscript -e "source('prophet_uk_run.R'); fit_prophet(${SLURM_ARRAY_TASK_ID})"

