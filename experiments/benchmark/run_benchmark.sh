#!/bin/bash -e
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err



NPZ_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" datasets.txt)

echo "Processing file: $NPZ_FILE"

. $HOME/micromamba/etc/profile.d/micromamba.sh

micromamba activate gp
echo "Running SegCPGP with parameters:"
cat $4

python run_segcpgp.py -d $NPZ_FILE -p $4 -r $2
