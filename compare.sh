#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=11:59:00
#SBATCH --partition=mit_normal
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=kunimune@mit.edu

echo "running on the $SLURM_JOB_PARTITION partition, on node $SLURM_JOB_NODELIST"

module load miniforge
conda activate grasshoppenv

cd $HOME/MERGS-detector

EXIT_CODE=124

for i in $(seq 1 11); do
	# delete old files
  if [ -d run ]; then
	  find run -mmin +60 -delete
  fi
	# run the script
	timeout 1.0h python -u compare.py $@
	# decide whether to keep going
	EXIT_CODE=$?
	if [ $EXIT_CODE -eq 124 ]; then
		echo "Killing and restarting to clear memory."
	else
		echo "Python terminated with exit code $EXIT_CODE"
		exit $EXIT_CODE
	fi
done

echo "Our time is up but there's more work to do.  Recuing..."
sbatch compare.sh $@
exit 0
