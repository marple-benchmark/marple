#!/bin/bash
#SBATCH --account=vision
#SBATCH --partition=svl --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1

#SBATCH --job-name="marple_eval"
#SBATCH --output=marple_eval_%A.out
#SBATCH --error=marple_eval_%A.err
####SBATCH --mail-user=emilyjin@stanford.edu
####SBATCH --mail-type=ALL

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

a_mission=$1
b_mission=$2
a_pref=$3
b_pref=$4
data_level=$5

python utils/eval_inference.py rollout.a_mission=$a_mission rollout.b_mission=$b_mission rollout.a_pref=$a_pref rollout.b_pref=$b_pref model.dirpath=${data_level}