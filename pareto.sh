#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --partition=mit_normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kunimune@mit.edu

module purge
module load miniforge
conda activate geant_env

cd $HOME/MERGS-detector

timeout 11.9h python -u pareto.py
if [ $? -eq 123 ]; then
	sbatch pareto.sh
fi

