#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:59:00
#SBATCH --partition=mit_normal
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=kunimune@mit.edu

echo "running on the $SLURM_JOB_PARTITION partition, on node $SLURM_JOB_NODELIST"

module load miniforge
conda activate grasshoppenv

cd $HOME/MERGS-detector

python -u compare.py $@
