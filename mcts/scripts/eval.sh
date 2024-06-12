#!/bin/bash
#SBATCH --account=vision
#SBATCH --partition=svl --qos=normal
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres=gpu:1

#SBATCH --job-name="marple_inference"
#SBATCH --output=marple_inference_%A.out
#SBATCH --error=marple_inference_%A.err
####SBATCH --mail-user=emilyjin@stanford.edu
####SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE
export HYDRA_FULL_ERROR=1

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

##########################################
# Setting up virtualenv / conda / docker #
##########################################
source /sailhome/weiyul/.bashrc
conda activate /vision/u/emilyjin/anaconda3/envs/marpl

echo "conda env activated"


##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/usr/lib/x86_64-linux-gnu
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

cd /vision/u/emilyjin/marple_long/

experiment_name="$1" 
checkpoint_name="$2"
split=$3
data_level=$4

echo "eval agent: " ${experiment_name}
echo "python utils/eval.py experiment.experiment_name=${experiment_name} model.checkpoint_name=${checkpoint_name} data.split=${split}" model.dirpath=/vision/u/emilyjin/marple_long/final_checkpoints/${data_level}

python utils/eval.py experiment.experiment_name=${experiment_name} model.checkpoint_name=${checkpoint_name} data.split=${split} model.dirpath=${data_level}
