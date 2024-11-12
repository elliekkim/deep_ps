#!/bin/bash

#SBATCH --job-name=test_ps_pymc
#SBATCH --output=test_ps_pymc.out
#SBATCH --error=test_ps_pymc.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate pymc

python test_ps_pymc.py \
    --test \
    --file_path ./data/july2023_eve.npy \
    --target_path ./results/

