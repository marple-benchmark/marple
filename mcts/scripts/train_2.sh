#!/bin/bash
#SBATCH --account=vision
#SBATCH --partition=svl --qos=normal
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --job-name="marple_train"
#SBATCH --output=marple_train_%A.out
#SBATCH --error=marple_train_%A.err
####SBATCH --mail-user=emilyjin@stanford.edu
####SBATCH --mail-type=ALL

# vision/u/emilyjin/marple_long/experiments/${experiment_name}/
# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

##########################################
# Setting up virtualenv / conda / docker #
##########################################
module load anaconda3
source activate marpl
echo "conda env activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/usr/lib/x86_64-linux-gnu
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd /vision/u/emilyjin/marple_long/

experiment_name="$1"
model_name="$2"

echo "training model: ${model_name}"
python utils/train_2.py experiment.experiment_name=${experiment_name} model.model_name=${model_name} 
