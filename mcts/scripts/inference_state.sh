#!/bin/bash
#SBATCH --account=vision
#SBATCH --partition=svl --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
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
inference_answer="$2" 
a_mission="$3"
b_mission="$4"
a_pref="$5"
b_pref="$6"
room_config=$7
traj_name=$8
model_name=$9
simulator=${10}
data_level=${11}

if [ "$model_name" == "low_policy" ]; then
        python utils/inference_low_policy.py rollout.room_config=${room_config} rollout.traj_name=${traj_name} rollout.a_mission=${a_mission} rollout.b_mission=${b_mission} rollout.a_pref=${a_pref} rollout.b_pref=${b_pref}  experiment.agent=${inference_answer} experiment.experiment_name=${experiment_name}  model.dirpath=${data_level}
elif [ "$model_name" == "subgoal_low_policy" ]; then
        if [ "$simulator" == 'False' ]; then
                echo "python utils/inference_subgoal_low_policy.py"
                python utils/inference_subgoal_low_policy.py rollout.room_config=${room_config} rollout.traj_name=${traj_name} rollout.a_mission=${a_mission} rollout.b_mission=${b_mission} rollout.a_pref=${a_pref} rollout.b_pref=${b_pref} experiment.agent=${inference_answer} experiment.experiment_name=${experiment_name}  model.dirpath=${data_level}
        else
                echo "python utils/inference_subgoal_low_policy_simulator.py"
                python utils/inference_subgoal_low_policy_simulator.py rollout.room_config=${room_config} rollout.traj_name=${traj_name} rollout.a_mission=${a_mission} rollout.b_mission=${b_mission} rollout.a_pref=${a_pref} rollout.b_pref=${b_pref} experiment.agent=${inference_answer} experiment.experiment_name=${experiment_name} model.dirpath=${data_level}
        fi
else
        echo "Model name not recognized"
        exit 1
fi  
