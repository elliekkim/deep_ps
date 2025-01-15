#!/bin/bash

#SBATCH --job-name=gp_pymc
#SBATCH --output=gp_pymc.out
#SBATCH --error=gp_pymc.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=64G

source ~/.bashrc
conda activate pymc_env

python scripts/gp_pymc.py \
    --file_path ./data/july2023_eve.npy \
    --target_path ./results/
