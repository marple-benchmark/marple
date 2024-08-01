# MARPLE: A Benchmark for Long-Horizon Inference
This repository contains the simulator and experiment code for **MARPLE: A Benchmark for Long-Horizon Inference**.

The project website is: https://marple-benchmark.github.io/.
The data is uploaded [here](https://drive.google.com/drive/folders/1zXsErNVOMYjBMWzTnmZS4e4aIljWlRce?usp=sharing).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Train and Evaluate Policy Models](#train-and-evaluate-policy-models)
- [Experiments](#experiments)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
Before Migration gym 0.21.0 to 0.26.0
```bash
conda create -n marple python==3.10
conda activate marple
# install gym 0.21.0
pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40

pip install -r requirements.txt
pip install -e .
# if encountered seed issue, delete seed=seed in reset() funcion in minigrid.py
# no need to run pip install inside of marple_mini_behavior
``` 

## Folder Structure
Below is the folder structure of our project:

- `/MARPLE/`
  -  `/gpt/` Code for GPT-4 
    - `azure.py`
    - `/config/`	
    - `gpt4.py`
    - `helpers.py` 
    - `main.py`
    - `prompts.py` 
  - `/mcts/` Code for Mental Simulation with Learned Agent Models 
    - `/config/`
    - `/src/`
      - `/mini_behavior/` Simulator Code
      - `/models/` Model Code
    - `arguments.py`
    - `eval.py`
    - `inference.py`
    - `main.py`
    - `train.py`
  - `/figures/`
  - `setup.py`
  - `requirements.txt`
  - `.gitignore`
  - `datasheet.md`
  - `README.md`

## Data
### Downloading Data
The data is uploaded at a [Google Drive link](https://drive.google.com/drive/folders/1c4ncerbpZMyWxhDs-ysPze6LxQ4lgW3e?usp=drive_link). You can download the zip files and add them to the "data" directory. The scripts will assume that the data is located in that directory.
<!-- ### Generating Data -->

## Train and Evaluate Policy Models
You can train a policy model on any dataset of agent trajectories. All of the policy models can be trained using the following command:
```bash
python mcts/train.py experiment.mission_1=${mission_1} experiment.mission_1_pref=${mission_1_pref} experiment.mission_2=${mission_2} experiment.mission_2_pref=${mission_2_pref} model.model_name=${model_name}
```

For example, to train a low-level policy model for an agent that performs the mission 'feed dog' 60% of the time and 'do laundry' 40% of the time, run the following command:
```bash
python mcts/train.py experiment.mission_1=feed_dog experiment.mission_1_pref=0.6 experiment.mission_2=take_shower experiment.mission_2_pref=0.4 model.model_name=low_policy
```

To evaluate the  model, use the following command: 
```bash
python mcts/eval.py experiment.mission_1=${mission_1} experiment.mission_1_pref=${mission_1_pref} experiment.mission_2=${mission_2} experiment.mission_2_pref=${mission_2_pref} model.model_name=${model_name} model.checkpoint_name=${checkpoint_name} data.split=${split} model.dirpath=${checkpoint_path} experiment.results_dir=${results_dir}
```

## Experiments
To run an inference experiment using a trained policy model, use the following command:
```bash
python mcts/main.py rollout.room_config=${room_config} rollout.traj_name=${traj_name} rollout.a_mission=${a_mission} rollout.b_mission=${b_mission} rollout.a_pref=${a_pref} rollout.b_pref=${b_pref}  experiment.agent=${inference_answer} experiment.experiment_name=${experiment_name}  model.dirpath=${data_level}
```

## GPT-4 Experiments
TO run an inference experiment using GPT-4, use the following command:
```bash
python gpt/main.py data.mission=$config 
```
## Contributing

## License

## Acknowledgements
