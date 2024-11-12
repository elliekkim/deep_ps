#!/bin/bash

#SBATCH --job-name=ps_pymc
#SBATCH --output=ps_pymc.out
#SBATCH --error=ps_pymc.err
#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate pymc_env

python scripts/ps_pymc.py \                                                                            
    --file_path ./data/july2023_eve.npy \
    --target_path ./results/

