#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate finalmlenv

python3 naivepinn.py

python3 tsirpinn.py


