#!/bin/bash

# Loop from 1 to 1451
for i in {1..1451}
do
    # Submit the Slurm job script with the current index as an argument
    sbatch send_one_city.slurm $i
done

