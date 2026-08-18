#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=11:59:00
#SBATCH --partition=sched_mit_psfc
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kunimune@mit.edu

module purge
module load miniforge
conda activate geant_env

cd $HOME/MERGS-detector

EXIT_CODE=124

for i in $(seq 1 23); do
	timeout 0.5h python -u pareto.py
	EXIT_CODE=$?
	if [ $EXIT_CODE -eq 124 ]; then
		echo "Killing and restarting to clear memory."
	else
		echo "Python terminated with exit code $EXIT_CODE"
		exit $EXIT_CODE
	fi
done

echo "Our time is up but there's more work to do.  Recuing..."
sbatch pareto.sh
exit 0
